#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:
    IsotonicRegression = None
    LogisticRegression = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

EPS = 1e-12
COINBASE_BASE = "https://api.exchange.coinbase.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
FRED_BASE = "https://api.stlouisfed.org/fred"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"
OKX_BASE = "https://www.okx.com"
BYDFI_BASE = "https://open-api.bydoxe.com"


class BTCProbabilityAlertApp:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.artifacts_dir = self.base_dir / "artifacts"
        self.logs_dir = self.artifacts_dir / "logs"
        self.latest_dir = self.artifacts_dir / "latest"
        self.charts_dir = self.artifacts_dir / "charts"

        for d in [self.artifacts_dir, self.logs_dir, self.latest_dir, self.charts_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.query_log_path = self.logs_dir / "query_log.csv"
        self.futures_log_path = self.logs_dir / "futures_query_log.csv"
        self._calibrator: object = {"method": "none", "model": None}
        self._calibrator_source: str = "none"
        self._calibrator_ready = False

    def run(
        self,
        target_price: float,
        timeframe_minutes: int,
        bankroll: Optional[float] = None,
        stake: Optional[float] = None,
        market_yes: Optional[float] = None,
        market_no: Optional[float] = None,
        edge_threshold: float = 0.03,
        kelly_fraction: float = 0.10,
        plot: bool = False,
    ) -> Dict[str, Any]:
        self._validate_inputs(
            target_price=target_price,
            timeframe_minutes=timeframe_minutes,
            bankroll=bankroll,
            stake=stake,
            edge_threshold=edge_threshold,
            market_yes=market_yes,
            market_no=market_no,
            kelly_fraction=kelly_fraction,
        )
        self._resolve_past_queries()
        result = self._compute(
            target_price=target_price,
            timeframe_minutes=timeframe_minutes,
            bankroll=bankroll,
            stake=stake,
            market_yes=market_yes,
            market_no=market_no,
            edge_threshold=edge_threshold,
            kelly_fraction=kelly_fraction,
        )
        self._print_summary(result)

        if plot:
            chart_path = self._plot_distribution(result)
            if chart_path is not None:
                print(f"Saved distribution chart: {chart_path}")

        self._write_latest(result)
        self._append_query_log(result)

        webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        if webhook:
            self._send_discord_alert(webhook, result)
        else:
            print("ALERT_WEBHOOK_URL not set; skipped Discord alert.")

        self._print_accuracy_summary()
        return result

    def run_auto(
        self,
        timeframe_minutes: int,
        poll_seconds: int = 60,
        strike_step: float = 100.0,
        strikes_per_side: int = 5,
        market_prob_min: float = 0.03,
        market_prob_max: float = 0.97,
        max_spread: float = 0.12,
        min_expiry_minutes: float = 2.0,
        max_expiry_minutes: float = 180.0,
        bankroll: Optional[float] = None,
        stake: Optional[float] = None,
        edge_threshold: float = 0.03,
        kelly_fraction: float = 0.10,
        plot: bool = False,
        max_cycles: Optional[int] = None,
    ) -> Dict[str, Any]:
        import time

        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be > 0")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")
        if strike_step <= 0:
            raise ValueError("strike_step must be > 0")
        if strikes_per_side < 1:
            raise ValueError("strikes_per_side must be >= 1")
        if not (0.0 <= market_prob_min < market_prob_max <= 1.0):
            raise ValueError("market_prob_min/max must satisfy 0<=min<max<=1")

        webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        cycle = 0
        last_result: Dict[str, Any] = {}

        print(
            f"[btcnew-auto] started | poll={poll_seconds}s timeframe={timeframe_minutes}m "
            f"edge_threshold={edge_threshold:.4f} market_prob_range=[{market_prob_min:.2f},{market_prob_max:.2f}]"
        )
        while True:
            cycle += 1
            if max_cycles is not None and cycle > max_cycles:
                print(f"[btcnew-auto] reached max_cycles={max_cycles}; stopping.")
                break

            cycle_started = datetime.now(timezone.utc)
            try:
                spot = self._fetch_spot_price([])
                markets = self._fetch_kalshi_btc_markets()
                candidates: List[Dict[str, Any]] = []
                filtered_extreme = 0
                filtered_spread = 0
                filtered_expiry = 0
                filtered_parse = 0
                for m in markets:
                    strike = self._extract_strike_from_market(m)
                    if strike is None:
                        filtered_parse += 1
                        continue
                    mins_left = self._extract_expiry_minutes(m, cycle_started)
                    if mins_left is None:
                        mins_left = float(timeframe_minutes)
                    if mins_left < min_expiry_minutes or mins_left > max_expiry_minutes:
                        filtered_expiry += 1
                        continue
                    tf = int(max(1, round(mins_left)))
                    yes_prob = float(np.clip(float(m.get("yes_prob", np.nan)), 0.0, 1.0))
                    no_prob = float(np.clip(float(m.get("no_prob", np.nan)), 0.0, 1.0))
                    if not np.isfinite(yes_prob) or not np.isfinite(no_prob):
                        filtered_parse += 1
                        continue
                    yes_is_above = self._infer_yes_is_above(m)
                    market_yes_for_above = yes_prob if yes_is_above else no_prob
                    market_no_for_above = 1.0 - market_yes_for_above
                    if market_yes_for_above <= market_prob_min or market_yes_for_above >= market_prob_max:
                        filtered_extreme += 1
                        continue
                    spread_raw = m.get("spread")
                    if spread_raw is not None:
                        spread = float(spread_raw)
                        if spread > 1.0:
                            spread /= 100.0
                        if spread > max_spread:
                            filtered_spread += 1
                            continue

                    result = self._compute(
                        target_price=float(strike),
                        timeframe_minutes=tf,
                        bankroll=bankroll,
                        stake=stake,
                        market_yes=float(market_yes_for_above),
                        market_no=float(market_no_for_above),
                        edge_threshold=float(edge_threshold),
                        kelly_fraction=float(kelly_fraction),
                    )
                    result["market_ticker"] = m.get("market_ticker")
                    result["market_expiry_minutes"] = float(mins_left)
                    result["market_yes_is_above"] = bool(yes_is_above)
                    candidates.append(result)

                if not candidates:
                    print(
                        f"[btcnew-auto] cycle={cycle} no usable markets "
                        f"raw={len(markets)} parse={filtered_parse} expiry={filtered_expiry} "
                        f"extreme={filtered_extreme} spread={filtered_spread}"
                    )
                    elapsed = (datetime.now(timezone.utc) - cycle_started).total_seconds()
                    sleep_s = max(0.0, float(poll_seconds) - float(elapsed))
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue

                def _rank(r: Dict[str, Any]) -> tuple:
                    signal = 1 if str(r.get("bet_signal")) == "BET_YES" else 0
                    ev = float(r.get("expected_value_per_1", 0.0))
                    edge = float(r.get("edge", 0.0))
                    prob = float(r.get("model_prob_yes", 0.0))
                    return (signal, ev, edge, prob)

                best = max(candidates, key=_rank)
                last_result = best

                # Always persist latest best candidate for local monitoring.
                self._print_summary(best)
                if plot:
                    chart_path = self._plot_distribution(best)
                    if chart_path is not None:
                        print(f"Saved distribution chart: {chart_path}")
                self._write_latest(best)

                # Scan all markets and alert/log only those with edge above threshold.
                qualified = [r for r in candidates if float(r.get("edge", 0.0)) >= float(edge_threshold)]
                qualified = sorted(qualified, key=lambda r: float(r.get("edge", 0.0)), reverse=True)
                if not qualified:
                    print(f"[btcnew-auto] no market met edge threshold >= {edge_threshold:.4f} this cycle")
                for q in qualified:
                    self._append_query_log(q)
                    if webhook:
                        self._send_discord_alert(webhook, q)
                    else:
                        print("ALERT_WEBHOOK_URL not set; skipped Discord alert.")

                self._print_accuracy_summary()

                print(
                    f"[btcnew-auto] cycle={cycle} "
                    f"spot={spot:.2f} markets_raw={len(markets)} markets_used={len(candidates)} "
                    f"selected_ticker={best.get('market_ticker')} selected_target={best['target_price']:.2f} "
                    f"signal={best['bet_signal']} edge={best['edge']:+.4f} ev={best['expected_value_per_1']:+.4f} "
                    f"qualified={len(qualified)} filtered(extreme/spread/expiry/parse)="
                    f"{filtered_extreme}/{filtered_spread}/{filtered_expiry}/{filtered_parse}"
                )
            except KeyboardInterrupt:
                print("[btcnew-auto] interrupted by user; stopping.")
                break
            except Exception as exc:
                print(f"[btcnew-auto] cycle={cycle} error: {exc}")

            elapsed = (datetime.now(timezone.utc) - cycle_started).total_seconds()
            sleep_s = max(0.0, float(poll_seconds) - float(elapsed))
            if sleep_s > 0:
                time.sleep(sleep_s)

        return last_result

    def run_futures(
        self,
        timeframe_minutes: int,
        leverage: float = 10.0,
        expected_return_threshold: float = 0.0005,
        min_confidence: float = 0.52,
        min_signal_strength: float = 0.10,
        prob_threshold: float = 0.55,
        take_profit_mult: float = 1.20,
        stop_loss_mult: float = 0.70,
        contract_check: bool = False,
        maintenance_margin_rate: float = 0.005,
        taker_fee_bps: float = 5.0,
        plot: bool = False,
    ) -> Dict[str, Any]:
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be > 0")
        if leverage <= 0:
            raise ValueError("leverage must be > 0")
        if expected_return_threshold < 0:
            raise ValueError("expected_return_threshold must be >= 0")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if not (0.0 < prob_threshold < 1.0):
            raise ValueError("prob_threshold must be in (0, 1)")
        if min_signal_strength < 0:
            raise ValueError("min_signal_strength must be >= 0")
        if take_profit_mult <= 0 or stop_loss_mult <= 0:
            raise ValueError("take_profit_mult/stop_loss_mult must be > 0")
        if maintenance_margin_rate < 0 or maintenance_margin_rate >= 1:
            raise ValueError("maintenance_margin_rate must be in [0, 1)")
        if taker_fee_bps < 0:
            raise ValueError("taker_fee_bps must be >= 0")

        result = self._compute_futures(
            timeframe_minutes=timeframe_minutes,
            leverage=leverage,
            expected_return_threshold=expected_return_threshold,
            min_confidence=min_confidence,
            min_signal_strength=min_signal_strength,
            prob_threshold=prob_threshold,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            contract_check=contract_check,
            maintenance_margin_rate=maintenance_margin_rate,
            taker_fee_bps=taker_fee_bps,
        )
        self._print_futures_summary(result)

        if plot:
            chart_path = self._plot_distribution(result)
            if chart_path is not None:
                print(f"Saved distribution chart: {chart_path}")

        self._write_latest(result)
        self._append_futures_log(result)

        webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        if webhook:
            self._send_discord_alert_futures(webhook, result)
        else:
            print("ALERT_WEBHOOK_URL not set; skipped Discord alert.")

        return result

    @staticmethod
    def _validate_inputs(
        target_price: float,
        timeframe_minutes: int,
        bankroll: Optional[float],
        stake: Optional[float],
        edge_threshold: float,
        market_yes: Optional[float],
        market_no: Optional[float],
        kelly_fraction: float,
    ) -> None:
        if target_price <= 0:
            raise ValueError("target_price must be > 0")
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be > 0")
        if bankroll is not None and bankroll <= 0:
            raise ValueError("bankroll must be > 0")
        if stake is not None and stake <= 0:
            raise ValueError("stake must be > 0")
        if edge_threshold < 0:
            raise ValueError("edge_threshold must be >= 0")
        if market_yes is not None and market_yes <= 0:
            raise ValueError("market_yes must be > 0")
        if market_no is not None and market_no <= 0:
            raise ValueError("market_no must be > 0")
        if kelly_fraction <= 0 or kelly_fraction > 1:
            raise ValueError("kelly_fraction must be in (0, 1]")

    @staticmethod
    def _prob_input_to_unit(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        p = float(value)
        if p > 1.0:
            p = p / 100.0
        return float(np.clip(p, 1e-6, 1 - 1e-6))

    @staticmethod
    def _safe_get_json(
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "8"))
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _kalshi_base_candidates(self) -> List[str]:
        env_base = os.getenv("KALSHI_API_BASE", "").strip()
        cands = [
            env_base,
            "https://api.kalshi.com/trade-api/v2",
            "https://api.elections.kalshi.com/trade-api/v2",
            "https://trading-api.kalshi.com/trade-api/v2",
        ]
        out: List[str] = []
        seen = set()
        for b in cands:
            if b and b not in seen:
                out.append(b)
                seen.add(b)
        return out

    def _kalshi_headers(self) -> Dict[str, str]:
        key = os.getenv("KALSHI_API_KEY", "").strip()
        token = os.getenv("KALSHI_API_TOKEN", "").strip()
        headers = {"accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if key:
            headers["Authorization"] = f"Bearer {key}"
            headers["KALSHI-ACCESS-KEY"] = key
        return headers

    @staticmethod
    def _normalize_price_prob(v: object) -> Optional[float]:
        if v is None:
            return None
        x = float(v)
        if x > 1.0:
            x /= 100.0
        return float(np.clip(x, 0.0, 1.0))

    def _extract_kalshi_market_fields(self, data: Dict[str, object], fallback_ticker: Optional[str] = None) -> Dict[str, object]:
        ticker = data.get("ticker") or data.get("market_ticker") or data.get("marketTicker") or fallback_ticker
        yes_ask_raw = data.get("yes_ask") or data.get("yesAsk") or data.get("yes_ask_price") or data.get("yesAskPrice")
        yes_bid_raw = data.get("yes_bid") or data.get("yesBid") or data.get("yes_bid_price") or data.get("yesBidPrice")
        no_ask_raw = data.get("no_ask") or data.get("noAsk") or data.get("no_ask_price") or data.get("noAskPrice")
        no_bid_raw = data.get("no_bid") or data.get("noBid") or data.get("no_bid_price") or data.get("noBidPrice")
        yes_last_raw = data.get("yes_price") or data.get("yesPrice")
        no_last_raw = data.get("no_price") or data.get("noPrice")
        spread = data.get("spread") or data.get("bid_ask_spread") or data.get("bidAskSpread")
        liq = data.get("volume") or data.get("open_interest") or data.get("openInterest") or data.get("liquidity") or 0
        strike = data.get("strike") or data.get("subtitle") or data.get("strike_price") or data.get("strikePrice")
        expiry = (
            data.get("close_time")
            or data.get("closeTime")
            or data.get("expiration_time")
            or data.get("expirationTime")
            or data.get("expiration_date")
            or data.get("expirationDate")
            or data.get("settlement_time")
            or data.get("settlementTime")
        )

        yes_ask = self._normalize_price_prob(yes_ask_raw)
        yes_bid = self._normalize_price_prob(yes_bid_raw)
        no_ask = self._normalize_price_prob(no_ask_raw)
        no_bid = self._normalize_price_prob(no_bid_raw)
        yes_last = self._normalize_price_prob(yes_last_raw)
        no_last = self._normalize_price_prob(no_last_raw)

        yes_p = yes_ask
        no_p = no_ask
        if yes_p is None and no_p is not None:
            yes_p = float(np.clip(1.0 - float(no_p), 0.0, 1.0))
        if no_p is None and yes_p is not None:
            no_p = float(np.clip(1.0 - float(yes_p), 0.0, 1.0))
        if yes_p is None and yes_bid is not None and yes_ask is not None:
            yes_p = float(np.clip((yes_bid + yes_ask) / 2.0, 0.0, 1.0))
        if no_p is None and no_bid is not None and no_ask is not None:
            no_p = float(np.clip((no_bid + no_ask) / 2.0, 0.0, 1.0))
        if yes_p is None and yes_last is not None:
            yes_p = yes_last
        if no_p is None and no_last is not None:
            no_p = no_last
        if yes_p is None and yes_bid is not None:
            yes_p = yes_bid
        if no_p is None and no_bid is not None:
            no_p = no_bid
        if yes_p is None and no_p is None:
            raise ValueError("Kalshi yes/no price missing")
        if yes_p is None:
            yes_p = float(np.clip(1.0 - float(no_p), 0.0, 1.0))
        if no_p is None:
            no_p = float(np.clip(1.0 - float(yes_p), 0.0, 1.0))

        return {
            "market_ticker": str(ticker) if ticker is not None else None,
            "yes_prob": float(yes_p),
            "no_prob": float(no_p),
            "yes_ask_prob": yes_ask,
            "yes_bid_prob": yes_bid,
            "no_ask_prob": no_ask,
            "no_bid_prob": no_bid,
            "spread": float(spread) if spread is not None else None,
            "liquidity": float(liq) if liq is not None else 0.0,
            "strike": strike,
            "expiry": expiry,
            "raw": data,
        }

    def _fetch_kalshi_markets_with_params(self, params: Dict[str, str]) -> List[Dict[str, object]]:
        headers = self._kalshi_headers()
        out_rows: List[Dict[str, object]] = []
        page_cap = max(1, int(os.getenv("KALSHI_TARGETED_MAX_PAGES", "2")))
        for base in self._kalshi_base_candidates():
            url = f"{base}/markets"
            for h in (headers, {"accept": "application/json"}):
                cursor = None
                for _ in range(page_cap):
                    q = dict(params)
                    if cursor:
                        q["cursor"] = str(cursor)
                    try:
                        payload = self._safe_get_json(url, params=q, headers=h)
                    except Exception:
                        break
                    if isinstance(payload, dict):
                        rows = payload.get("markets") or payload.get("data") or payload.get("results") or []
                        cursor = payload.get("cursor") or payload.get("next_cursor") or payload.get("nextCursor")
                    elif isinstance(payload, list):
                        rows = payload
                        cursor = None
                    else:
                        break
                    if not isinstance(rows, list):
                        break
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        rr = r.get("market") if isinstance(r.get("market"), dict) else r
                        try:
                            out_rows.append(self._extract_kalshi_market_fields(rr))
                        except Exception:
                            continue
                    if not cursor:
                        break

        uniq: Dict[str, Dict[str, object]] = {}
        for m in out_rows:
            t = m.get("market_ticker")
            if t:
                uniq[str(t)] = m
        return list(uniq.values())

    @staticmethod
    def _is_supported_threshold_market(market: Dict[str, object]) -> bool:
        raw = market.get("raw", {}) if isinstance(market.get("raw"), dict) else {}
        txt = " ".join(
            str(x).upper()
            for x in [
                market.get("market_ticker", ""),
                raw.get("title", ""),
                raw.get("subtitle", ""),
                raw.get("event_ticker", ""),
                raw.get("eventTicker", ""),
            ]
        )
        if any(k in txt for k in ["RANGE", "BETWEEN", " TO ", "-TO-", " BRACKET", " BAND "]):
            return False
        if ("OR ABOVE" in txt) or ("OR BELOW" in txt) or ("ABOVE" in txt) or ("BELOW" in txt):
            return True
        if re.search(r"-T\d{4,8}(\.\d+)?$", str(market.get("market_ticker", "")).upper()):
            return True
        return False

    @staticmethod
    def _infer_yes_is_above(market: Dict[str, object]) -> bool:
        raw = market.get("raw", {}) if isinstance(market.get("raw"), dict) else {}
        txt = " ".join(
            str(x).upper()
            for x in [
                market.get("market_ticker", ""),
                raw.get("title", ""),
                raw.get("subtitle", ""),
                raw.get("rules_primary", ""),
                raw.get("rulesPrimary", ""),
            ]
        )
        if any(k in txt for k in ["BELOW", "UNDER", "AT OR BELOW", "LESS THAN"]):
            return False
        return True

    @staticmethod
    def _extract_strike_from_market(m: Dict[str, object]) -> Optional[float]:
        s = m.get("strike")
        if isinstance(s, (int, float)):
            v = float(s)
            if v > 300_000:
                v /= 100.0
            if 5_000 <= v <= 300_000:
                return v
        raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
        for key in ["strike_price", "strikePrice", "floor_strike", "floorStrike", "cap_strike", "capStrike"]:
            v = raw.get(key)
            if isinstance(v, (int, float)):
                fv = float(v)
                if fv > 300_000:
                    fv /= 100.0
                if 5_000 <= fv <= 300_000:
                    return fv
        txt = " ".join(str(m.get(k, "")) for k in ["market_ticker", "raw", "strike"])
        mm = re.findall(r"(?<!\d)(\d{4,7})(?!\d)", txt)
        vals = [float(x) for x in mm]
        plausible = [v for v in vals if 5_000 <= v <= 300_000]
        if plausible:
            return float(plausible[-1])
        return None

    @staticmethod
    def _extract_expiry_minutes(m: Dict[str, object], now: datetime) -> Optional[float]:
        raw = m.get("expiry")
        if raw is None:
            rr = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
            raw = (
                rr.get("close_time")
                or rr.get("closeTime")
                or rr.get("expiration_time")
                or rr.get("expirationTime")
                or rr.get("expiration_date")
                or rr.get("expirationDate")
                or rr.get("settlement_time")
                or rr.get("settlementTime")
            )
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return float((ts.to_pydatetime() - now).total_seconds() / 60.0)

    def _fetch_kalshi_btc_markets(self) -> List[Dict[str, object]]:
        queries = [
            {"status": "open", "limit": "1000", "search": "bitcoin"},
            {"status": "open", "limit": "1000", "search": "btc"},
            {"status": "active", "limit": "1000", "search": "bitcoin"},
            {"status": "active", "limit": "1000", "search": "btc"},
        ]
        raw = os.getenv("KALSHI_BTC_SERIES", "KXBTCD,KXBTC,KXBT")
        for series in [s.strip().upper() for s in re.split(r"[,\s;]+", raw) if s.strip()]:
            queries.extend(
                [
                    {"status": "open", "limit": "1000", "series_ticker": series},
                    {"status": "active", "limit": "1000", "series_ticker": series},
                ]
            )

        markets: List[Dict[str, object]] = []
        for q in queries:
            try:
                rows = self._fetch_kalshi_markets_with_params(q)
                if rows:
                    markets.extend(rows)
            except Exception:
                continue

        keywords = ["BTC", "BITCOIN", "XBT", "KXBTC", "KXBT", "KXBTCD", "BTCUSD", "XBTUSD"]
        uniq: Dict[str, Dict[str, object]] = {}
        for m in markets:
            t = str(m.get("market_ticker", "")).upper()
            rawm = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
            txt = " ".join(
                str(x).upper()
                for x in [
                    t,
                    rawm.get("title", ""),
                    rawm.get("subtitle", ""),
                    rawm.get("event_ticker", ""),
                    rawm.get("series_ticker", ""),
                ]
            )
            if not any(k in txt for k in keywords):
                continue
            if not self._is_supported_threshold_market(m):
                continue
            if t:
                uniq[t] = m
        return list(uniq.values())

    @staticmethod
    def _http_status(exc: Exception) -> Optional[int]:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return int(exc.response.status_code)
        return None

    def _compute(
        self,
        target_price: float,
        timeframe_minutes: int,
        bankroll: Optional[float],
        stake: Optional[float],
        market_yes: Optional[float],
        market_no: Optional[float],
        edge_threshold: float,
        kelly_fraction: float,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        now_utc = datetime.now(timezone.utc)
        self._ensure_calibrator_loaded(warnings)

        spot = self._fetch_spot_price(warnings)
        factors = self._collect_factors(timeframe_minutes, warnings)
        spot, factors = self._sanitize_market_inputs(spot, factors, warnings)
        annual_vol = max(float(factors.get("realized_vol_annual", 0.55)), 0.08)
        t_years = max(timeframe_minutes, 1) / (365.0 * 24.0 * 60.0)
        expected_move = max(spot * annual_vol * math.sqrt(t_years), EPS)
        distance_vol_norm = float((spot - target_price) / expected_move)
        strike_distance_pct = self._strike_distance_pct(target_price=target_price, spot_price=spot)
        minutes_remaining = float(max(timeframe_minutes, 1))
        factors["distance_vol_normalized_signed"] = float(distance_vol_norm)
        factors["distance_vol_normalized_abs"] = float(abs(distance_vol_norm))
        factors["minutes_remaining"] = minutes_remaining
        factors["sqrt_time_remaining"] = float(math.sqrt(minutes_remaining))
        factors["minutes_remaining_scaled"] = float(min(minutes_remaining / 60.0, 1.0))
        factors["time_adjusted_distance"] = float(strike_distance_pct / minutes_remaining)

        market_above_model = self._market_probability(spot, target_price, timeframe_minutes, annual_vol, "above")
        market_below_model = self._market_probability(spot, target_price, timeframe_minutes, annual_vol, "below")
        market_above, market_below, market_prob_source = self._resolve_market_probabilities(
            market_yes=market_yes,
            market_no=market_no,
            fallback_yes=market_above_model,
            fallback_no=market_below_model,
            warnings=warnings,
        )
        model_above_raw, model_meta = self._model_probability(
            spot, target_price, timeframe_minutes, annual_vol, "above", factors, return_details=True
        )
        model_below_raw = float(np.clip(1.0 - model_above_raw, 0.01, 0.99))

        model_above = float(self.apply_calibrator(self._calibrator, np.array([model_above_raw], dtype=float))[0])
        model_below = float(np.clip(1.0 - model_above, 0.01, 0.99))

        edge_above = model_above - market_above
        edge_below = model_below - market_below

        action_above = "YES" if edge_above >= edge_threshold else "NO"
        action_below = "YES" if edge_below >= edge_threshold else "NO"

        decision = self._decision_table(
            model_yes=model_above,
            model_no=model_below,
            market_yes=market_above,
            market_no=market_below,
            bankroll=bankroll,
            stake=stake,
            kelly_fraction=kelly_fraction,
            warnings=warnings,
        )

        self._check_model_market_divergence(
            model_yes=model_above,
            model_no=model_below,
            market_yes=market_above,
            market_no=market_below,
            warnings=warnings,
        )
        self._check_model_probability_spike(model_yes=model_above, now_utc=now_utc, warnings=warnings)

        staking_above = decision["rows"]["YES"]["staking"]
        staking_below = decision["rows"]["NO"]["staking"]
        if decision["recommended_side"] == "YES":
            recommended_ev = float(decision["rows"]["YES"]["ev"])
            recommended_stake = decision["rows"]["YES"]["staking"]["recommended_stake"]
        elif decision["recommended_side"] == "NO":
            recommended_ev = float(decision["rows"]["NO"]["ev"])
            recommended_stake = decision["rows"]["NO"]["staking"]["recommended_stake"]
        else:
            recommended_ev = 0.0
            recommended_stake = None

        moneyness_category = self._moneyness_category(strike_distance_pct)
        confidence_meta = self._confidence_reliability(
            model_prob_yes=model_above,
            distance_vol_normalized=distance_vol_norm,
            model_meta=model_meta,
            factors=factors,
            warnings=warnings,
        )
        confidence_score = float(confidence_meta["confidence_score"])
        confidence_tier = self._confidence_tier_from_score(confidence_score)
        bet_signal = self._validation_bet_signal(model_prob_yes=model_above, edge=edge_above, strike_distance_pct=strike_distance_pct)
        if bet_signal == "BET_YES":
            expected_value_per_1 = float(decision["rows"]["YES"]["ev"])
            kelly_fraction_signal = float(decision["rows"]["YES"]["staking"]["kelly_fraction_used"])
            suggested_stake_signal = decision["rows"]["YES"]["staking"]["recommended_stake"]
        else:
            expected_value_per_1 = 0.0
            kelly_fraction_signal = 0.0
            suggested_stake_signal = None

        result = {
            "timestamp_utc": now_utc.isoformat(),
            "target_price": float(target_price),
            "timeframe_minutes": int(timeframe_minutes),
            "spot_price": float(spot),
            "model_probability_above": float(model_above),
            "model_probability_below": float(model_below),
            "model_probability_above_raw": float(model_above_raw),
            "model_probability_below_raw": float(model_below_raw),
            "market_probability_above": float(market_above),
            "market_probability_below": float(market_below),
            "market_probability_above_model_implied": float(market_above_model),
            "market_probability_below_model_implied": float(market_below_model),
            "market_probability_source": market_prob_source,
            "edge_above": float(edge_above),
            "edge_below": float(edge_below),
            "suggested_action_above": action_above,
            "suggested_action_below": action_below,
            "recommended_side": decision["recommended_side"],
            "recommended_action": decision["recommended_action"],
            "decision_reasoning_steps": decision["reasoning_steps"],
            "recommended_ev": recommended_ev,
            "recommended_stake": recommended_stake,
            "model_prob_yes": float(model_above),
            "market_prob_yes": float(market_above),
            "edge": float(edge_above),
            "strike_distance_pct": float(strike_distance_pct),
            "distance_vol_normalized": float(distance_vol_norm),
            "moneyness_category": moneyness_category,
            "confidence_tier": confidence_tier,
            "confidence_score": confidence_score,
            "confidence_components": confidence_meta,
            "bet_signal": bet_signal,
            "expected_value_per_1": float(expected_value_per_1),
            "kelly_fraction_signal": float(kelly_fraction_signal),
            "suggested_stake_signal": suggested_stake_signal,
            "expected_final_price": float(model_meta.get("expected_final_price", spot)),
            "expected_variance": float(model_meta.get("expected_variance", expected_move * expected_move)),
            "near_strike_model_used": bool(model_meta.get("near_strike_model_used", False)),
            "p_above_from_distribution": float(model_meta.get("p_above_from_distribution", model_above_raw)),
            "ev_table": decision["rows"],
            "market_factor_summary": {
                "source": market_prob_source,
                "market_yes": float(market_above),
                "market_no": float(market_below),
                "payout_yes": float(decision["rows"]["YES"]["payout"]),
                "payout_no": float(decision["rows"]["NO"]["payout"]),
                "market_sum": float(market_above + market_below),
                "calibration_method": str(self._calibrator.get("method", "none")) if isinstance(self._calibrator, dict) else "none",
            },
            "calibration_method": str(self._calibrator.get("method", "none")) if isinstance(self._calibrator, dict) else "none",
            "calibration_source": self._calibrator_source,
            "edge_threshold": float(edge_threshold),
            "kelly_fraction": float(kelly_fraction),
            "staking_above": staking_above,
            "staking_below": staking_below,
            "factors": factors,
            "warnings": warnings,
            "resolution_due_utc": (now_utc + timedelta(minutes=timeframe_minutes)).isoformat(),
            "resolved": False,
            "resolved_price": None,
            "actual_hit_above": None,
            "actual_hit_below": None,
        }
        return result

    def _compute_futures(
        self,
        timeframe_minutes: int,
        leverage: float,
        expected_return_threshold: float,
        min_confidence: float,
        min_signal_strength: float,
        prob_threshold: float,
        take_profit_mult: float,
        stop_loss_mult: float,
        contract_check: bool,
        maintenance_margin_rate: float,
        taker_fee_bps: float,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        now_utc = datetime.now(timezone.utc)
        self._ensure_calibrator_loaded(warnings)

        spot = self._fetch_spot_price(warnings)
        factors = self._collect_factors(timeframe_minutes, warnings)
        spot, factors = self._sanitize_market_inputs(spot, factors, warnings)

        annual_vol = max(float(factors.get("realized_vol_annual", 0.55)), 0.08)
        t_years = max(timeframe_minutes, 1) / (365.0 * 24.0 * 60.0)
        sigma = max(annual_vol * math.sqrt(t_years), 1e-6)
        expected_move = max(spot * sigma, EPS)

        factors["distance_vol_normalized_signed"] = 0.0
        factors["distance_vol_normalized_abs"] = 0.0
        factors["minutes_remaining"] = float(max(timeframe_minutes, 1))
        factors["sqrt_time_remaining"] = float(math.sqrt(max(timeframe_minutes, 1)))
        factors["minutes_remaining_scaled"] = float(min(max(timeframe_minutes, 1) / 60.0, 1.0))
        factors["time_adjusted_distance"] = 0.0

        prob_up_raw, model_meta = self._model_probability(
            spot, spot, timeframe_minutes, annual_vol, "above", factors, return_details=True
        )
        prob_up = float(self.apply_calibrator(self._calibrator, np.array([prob_up_raw], dtype=float))[0])
        prob_down = float(np.clip(1.0 - prob_up, 0.01, 0.99))

        expected_final_price = float(model_meta.get("expected_final_price", spot))
        expected_return = float((expected_final_price - spot) / max(spot, EPS))
        realized_volatility = float(sigma)
        signal_strength = float(expected_return / (realized_volatility + EPS))

        confidence_meta = self._confidence_reliability(
            model_prob_yes=prob_up,
            distance_vol_normalized=signal_strength,
            model_meta=model_meta,
            factors=factors,
            warnings=warnings,
        )
        confidence_score = float(confidence_meta["confidence_score"])

        direction = "NO_TRADE"
        if (
            prob_up >= prob_threshold
            and expected_return > expected_return_threshold
            and abs(signal_strength) >= min_signal_strength
        ):
            direction = "LONG"
        elif (
            prob_up <= (1.0 - prob_threshold)
            and expected_return < -expected_return_threshold
            and abs(signal_strength) >= min_signal_strength
        ):
            direction = "SHORT"

        vol_pct = float(np.clip(realized_volatility, 1e-6, 1.0))
        abs_ret = float(abs(expected_return))
        edge_move = float(max(abs_ret, vol_pct * 0.50))
        take_profit_pct = float(max(edge_move * take_profit_mult, vol_pct * 0.35))
        stop_loss_pct = float(max(edge_move * stop_loss_mult, vol_pct * 0.25))

        entry_price = float(spot)
        if direction == "LONG":
            take_profit = float(entry_price * (1.0 + take_profit_pct))
            stop_loss = float(entry_price * (1.0 - stop_loss_pct))
        elif direction == "SHORT":
            take_profit = float(entry_price * (1.0 - take_profit_pct))
            stop_loss = float(entry_price * (1.0 + stop_loss_pct))
        else:
            take_profit = None
            stop_loss = None

        fee_roundtrip_bps = float(max(taker_fee_bps, 0.0) * 2.0)
        fee_roundtrip_fraction = fee_roundtrip_bps / 10000.0

        estimated_liquidation_price: Optional[float] = None
        distance_to_liquidation_pct: Optional[float] = None
        take_profit_unlevered_pct: Optional[float] = None
        stop_loss_unlevered_pct: Optional[float] = None
        take_profit_pnl_pct_levered_net: Optional[float] = None
        stop_loss_pnl_pct_levered_net: Optional[float] = None
        stop_within_liquidation_buffer: Optional[bool] = None

        if direction == "LONG" and take_profit is not None and stop_loss is not None:
            estimated_liquidation_price = float(
                entry_price * (1.0 - (1.0 / max(leverage, EPS)) + float(maintenance_margin_rate))
            )
            distance_to_liquidation_pct = float(
                max((entry_price - estimated_liquidation_price) / max(entry_price, EPS), 0.0)
            )
            take_profit_unlevered_pct = float((take_profit - entry_price) / max(entry_price, EPS))
            stop_loss_unlevered_pct = float((stop_loss - entry_price) / max(entry_price, EPS))
            take_profit_pnl_pct_levered_net = float(take_profit_unlevered_pct * leverage - fee_roundtrip_fraction)
            stop_loss_pnl_pct_levered_net = float(stop_loss_unlevered_pct * leverage - fee_roundtrip_fraction)
            stop_within_liquidation_buffer = bool(stop_loss > estimated_liquidation_price)
        elif direction == "SHORT" and take_profit is not None and stop_loss is not None:
            estimated_liquidation_price = float(
                entry_price * (1.0 + (1.0 / max(leverage, EPS)) - float(maintenance_margin_rate))
            )
            distance_to_liquidation_pct = float(
                max((estimated_liquidation_price - entry_price) / max(entry_price, EPS), 0.0)
            )
            take_profit_unlevered_pct = float((entry_price - take_profit) / max(entry_price, EPS))
            stop_loss_unlevered_pct = float((entry_price - stop_loss) / max(entry_price, EPS))
            take_profit_pnl_pct_levered_net = float(take_profit_unlevered_pct * leverage - fee_roundtrip_fraction)
            stop_loss_pnl_pct_levered_net = float(stop_loss_unlevered_pct * leverage - fee_roundtrip_fraction)
            stop_within_liquidation_buffer = bool(stop_loss < estimated_liquidation_price)

        if contract_check and leverage >= 50:
            warnings.append(
                "High-leverage contract check enabled. Verify exchange maintenance margin tiers and liquidation formula before live trading."
            )
        if contract_check and distance_to_liquidation_pct is not None and distance_to_liquidation_pct < 0.02:
            warnings.append(
                f"Liquidation buffer is thin: {distance_to_liquidation_pct * 100:.2f}% from entry at {leverage:.1f}x."
            )
        if contract_check and stop_within_liquidation_buffer is False:
            warnings.append("Configured stop-loss is beyond estimated liquidation price.")

        result = {
            "mode": "futures",
            "timestamp_utc": now_utc.isoformat(),
            "timeframe_minutes": int(timeframe_minutes),
            "timeframe_min": int(timeframe_minutes),
            "spot_price": float(spot),
            "target_price": float(expected_final_price),
            "expected_return": float(expected_return),
            "expected_price": float(expected_final_price),
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "confidence_score": float(confidence_score),
            "signal": direction,
            "entry_price": float(entry_price),
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "signal_strength": float(signal_strength),
            "leverage": float(leverage),
            "expected_return_threshold": float(expected_return_threshold),
            "min_confidence": float(min_confidence),
            "min_signal_strength": float(min_signal_strength),
            "prob_threshold": float(prob_threshold),
            "take_profit_mult": float(take_profit_mult),
            "stop_loss_mult": float(stop_loss_mult),
            "contract_check_enabled": bool(contract_check),
            "maintenance_margin_rate": float(maintenance_margin_rate),
            "fee_roundtrip_bps": float(fee_roundtrip_bps),
            "estimated_liquidation_price": estimated_liquidation_price,
            "distance_to_liquidation_pct": distance_to_liquidation_pct,
            "take_profit_unlevered_pct": take_profit_unlevered_pct,
            "stop_loss_unlevered_pct": stop_loss_unlevered_pct,
            "take_profit_pnl_pct_levered_net": take_profit_pnl_pct_levered_net,
            "stop_loss_pnl_pct_levered_net": stop_loss_pnl_pct_levered_net,
            "stop_within_liquidation_buffer": stop_within_liquidation_buffer,
            "realized_volatility": float(realized_volatility),
            "expected_move": float(expected_move),
            "confidence_components": confidence_meta,
            "factors": factors,
            "warnings": warnings,
            "resolution_due_utc": (now_utc + timedelta(minutes=timeframe_minutes)).isoformat(),
        }
        return result

    def _sanitize_market_inputs(
        self,
        spot: float,
        factors: Dict[str, float],
        warnings: List[str],
    ) -> tuple[float, Dict[str, float]]:
        if not np.isfinite(spot) or spot <= 0 or spot < 1_000 or spot > 500_000:
            warnings.append("WARNING: Spot price failed sanity checks; using fallback spot=65000.0")
            spot = 65_000.0

        sanitized = dict(factors)
        # (min, max, fallback)
        bounds: Dict[str, tuple[float, float, float]] = {
            "bid_ask_spread": (0.0, 0.10, 0.0004),
            "trade_flow_notional_usd": (0.0, 5e10, 0.0),
            "funding_rate": (-0.25, 0.25, 0.0),
            "open_interest_change": (-3.0, 3.0, 0.0),
            "liquidation_imbalance": (-1.0, 1.0, 0.0),
            "perp_premium": (-0.25, 0.25, 0.0),
            "long_short_ratio": (0.01, 20.0, 1.0),
            "futures_funding_rate": (-0.25, 0.25, 0.0),
            "futures_open_interest_change_1m": (-3.0, 3.0, 0.0),
            "futures_open_interest_change_5m": (-3.0, 3.0, 0.0),
            "futures_liquidation_imbalance": (-1.0, 1.0, 0.0),
            "futures_liquidation_pressure": (0.0, 5e10, 0.0),
            "futures_orderbook_imbalance": (-1.0, 1.0, 0.0),
            "futures_orderbook_ratio": (0.01, 100.0, 1.0),
            "realized_vol_annual": (0.01, 6.0, 0.55),
            "realized_vol_expansion": (-5.0, 5.0, 0.0),
            "vol_1m": (0.0, 8.0, 0.55),
            "vol_5m": (0.0, 8.0, 0.55),
            "vol_15m": (0.0, 8.0, 0.55),
            "volume_spike": (0.0, 50.0, 1.0),
            "return_1m": (-0.2, 0.2, 0.0),
            "return_3m": (-0.3, 0.3, 0.0),
            "return_5m": (-0.4, 0.4, 0.0),
            "return_10m": (-0.6, 0.6, 0.0),
            "momentum_acceleration": (-0.4, 0.4, 0.0),
            "vwap_deviation": (-0.3, 0.3, 0.0),
            "distance_to_vwap": (-0.3, 0.3, 0.0),
            "distance_to_high_15m": (0.0, 0.3, 0.0),
            "distance_to_low_15m": (0.0, 0.3, 0.0),
            "price_minus_ema_3": (-0.2, 0.2, 0.0),
            "price_minus_ema_10": (-0.2, 0.2, 0.0),
            "ema_3_minus_ema_10": (-0.2, 0.2, 0.0),
            "distance_vol_normalized_signed": (-8.0, 8.0, 0.0),
            "distance_vol_normalized_abs": (0.0, 8.0, 0.0),
            "minutes_remaining": (0.0, 1_440.0, 60.0),
            "sqrt_time_remaining": (0.0, 100.0, 7.75),
            "minutes_remaining_scaled": (0.0, 1.0, 1.0),
            "time_adjusted_distance": (0.0, 0.2, 0.0),
            "fear_greed": (-1.0, 1.0, 0.0),
            "social_velocity": (-1.0, 1.0, 0.0),
            "etf_flows_z": (-3.0, 3.0, 0.0),
            "spx_btc_corr": (-1.0, 1.0, 0.0),
            "dxy_return": (-0.2, 0.2, 0.0),
            "bond_yield_change": (-2.0, 2.0, 0.0),
            "cpi_yoy": (-0.1, 0.3, 0.0),
        }
        for k, (lo, hi, fallback) in bounds.items():
            raw = sanitized.get(k, fallback)
            try:
                v = float(raw)
            except Exception:
                warnings.append(f"WARNING: Input '{k}' missing/non-numeric; using fallback {fallback}.")
                sanitized[k] = fallback
                continue
            if not np.isfinite(v):
                warnings.append(f"WARNING: Input '{k}' non-finite; using fallback {fallback}.")
                sanitized[k] = fallback
                continue
            if v == 0.0 and k in {"trade_flow_notional_usd"}:
                warnings.append(f"WARNING: Input '{k}' is zero; flagged as potentially missing.")
            if v < lo or v > hi:
                warnings.append(
                    f"WARNING: Input '{k}' out of range [{lo},{hi}] (value={v}); flagged unrealistic spike and fallback applied."
                )
                sanitized[k] = fallback
            else:
                sanitized[k] = v
        return float(spot), sanitized

    def _divergence_threshold(self) -> float:
        raw = os.getenv("MODEL_MARKET_DIVERGENCE_THRESHOLD", "0.20").strip()
        try:
            return float(np.clip(float(raw), 0.0, 1.0))
        except Exception:
            return 0.20

    def _spike_threshold(self) -> float:
        raw = os.getenv("MODEL_PROB_SPIKE_THRESHOLD", "0.20").strip()
        try:
            return float(np.clip(float(raw), 0.0, 1.0))
        except Exception:
            return 0.20

    def _spike_window_seconds(self) -> int:
        raw = os.getenv("MODEL_PROB_SPIKE_WINDOW_SECONDS", "60").strip()
        try:
            return max(1, int(raw))
        except Exception:
            return 60

    def _check_model_market_divergence(
        self,
        model_yes: float,
        model_no: float,
        market_yes: float,
        market_no: float,
        warnings: List[str],
    ) -> None:
        # Market-implied probability from market "price" is P = 1 / price.
        price_yes = 1.0 / max(float(np.clip(market_yes, 1e-6, 1.0)), 1e-6)
        price_no = 1.0 / max(float(np.clip(market_no, 1e-6, 1.0)), 1e-6)
        p_market_yes = 1.0 / price_yes
        p_market_no = 1.0 / price_no
        diff = max(abs(float(model_yes) - p_market_yes), abs(float(model_no) - p_market_no))
        if diff >= self._divergence_threshold():
            warnings.append("WARNING: Model probability significantly diverges from market data. Check inputs.")

    def _check_model_probability_spike(
        self,
        model_yes: float,
        now_utc: datetime,
        warnings: List[str],
    ) -> None:
        if not self.query_log_path.exists():
            return
        try:
            df = pd.read_csv(self.query_log_path, on_bad_lines="skip", engine="python")
            if df.empty or "timestamp_utc" not in df.columns or "model_probability_above" not in df.columns:
                return
            t = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            p = pd.to_numeric(df["model_probability_above"], errors="coerce")
            valid = pd.DataFrame({"t": t, "p": p}).dropna().sort_values("t")
            if valid.empty:
                return
            last = valid.iloc[-1]
            dt = (now_utc - last["t"].to_pydatetime()).total_seconds()
            if 0 <= dt <= self._spike_window_seconds():
                if abs(float(model_yes) - float(last["p"])) >= self._spike_threshold():
                    warnings.append("WARNING: Model probability significantly diverges from market data. Check inputs.")
        except Exception:
            return

    @staticmethod
    def _strike_distance_pct(target_price: float, spot_price: float) -> float:
        if float(target_price) <= 0:
            return 0.0
        return float(abs(float(target_price) - float(spot_price)) / float(target_price))

    @staticmethod
    def _moneyness_category(strike_distance_pct: float) -> str:
        d = float(strike_distance_pct)
        if d < 0.005:
            return "ATM"
        if d < 0.01:
            return "Near"
        if d < 0.02:
            return "Mid"
        return "Far"

    @staticmethod
    def _confidence_tier(model_prob_yes: float) -> str:
        p = float(model_prob_yes)
        if p < 0.60:
            return "LOW"
        if p < 0.65:
            return "MEDIUM"
        if p < 0.70:
            return "HIGH"
        return "VERY_HIGH"

    @staticmethod
    def _confidence_tier_from_score(score: float) -> str:
        s = float(np.clip(score, 0.0, 1.0))
        if s < 0.45:
            return "LOW"
        if s < 0.60:
            return "MEDIUM"
        if s < 0.75:
            return "HIGH"
        return "VERY_HIGH"

    def _confidence_reliability(
        self,
        model_prob_yes: float,
        distance_vol_normalized: float,
        model_meta: Dict[str, Any],
        factors: Dict[str, float],
        warnings: List[str],
    ) -> Dict[str, float]:
        p = float(np.clip(model_prob_yes, 0.0, 1.0))
        p_dist = float(np.clip(float(model_meta.get("p_above_from_distribution", p)), 0.0, 1.0))
        p_drift = float(np.clip(float(model_meta.get("p_above_from_drift", p)), 0.0, 1.0))
        p_score = float(np.clip(float(model_meta.get("p_above_from_score", p)), 0.0, 1.0))

        # 1) Directional separation from 50/50.
        prob_separation = float(np.clip(abs(p - 0.5) / 0.20, 0.0, 1.0))
        # 2) Vol-normalized distance (higher abs(z) => less near-boundary noise).
        distance_strength = float(np.clip(abs(float(distance_vol_normalized)) / 1.0, 0.0, 1.0))
        # 3) Agreement between independent model paths.
        disagreement = max(abs(p - p_dist), abs(p - p_drift), abs(p - p_score))
        component_agreement = float(1.0 - np.clip(disagreement / 0.25, 0.0, 1.0))
        # 4) Micro-vol noise penalty.
        v1 = float(max(float(factors.get("vol_1m", 0.0)), 0.0))
        v5 = float(max(float(factors.get("vol_5m", 0.0)), 0.0))
        v15 = float(max(float(factors.get("vol_15m", 0.0)), 0.0))
        vol_noise = float(np.clip(0.6 * v1 + 0.3 * v5 + 0.1 * v15, 0.0, 3.0))
        noise_stability = float(1.0 - np.clip(vol_noise / 2.0, 0.0, 1.0))
        # 5) Calibration state bonus/penalty.
        cal_method = str(self._calibrator.get("method", "none")) if isinstance(self._calibrator, dict) else "none"
        calibration_quality = 1.0 if cal_method != "none" else 0.72
        # 6) Warning load penalty.
        warning_penalty = float(np.clip(0.10 * len(warnings), 0.0, 0.35))

        score = (
            0.24 * prob_separation
            + 0.18 * distance_strength
            + 0.28 * component_agreement
            + 0.15 * noise_stability
            + 0.15 * calibration_quality
            - warning_penalty
        )
        score = float(np.clip(score, 0.0, 1.0))
        return {
            "confidence_score": score,
            "prob_separation": prob_separation,
            "distance_strength": distance_strength,
            "component_agreement": component_agreement,
            "noise_stability": noise_stability,
            "calibration_quality": calibration_quality,
            "warning_penalty": warning_penalty,
        }

    @staticmethod
    def _validation_bet_signal(model_prob_yes: float, edge: float, strike_distance_pct: float) -> str:
        # Rule inputs are in percent distance bands: 0.15, 0.30, 0.60 (%).
        p = float(model_prob_yes)
        e = float(edge)
        d_pct = float(strike_distance_pct) * 100.0
        if d_pct < 0.15:
            p_req, e_req = 0.68, 0.07
        elif d_pct < 0.30:
            p_req, e_req = 0.65, 0.05
        elif d_pct < 0.60:
            p_req, e_req = 0.60, 0.04
        else:
            p_req, e_req = 0.57, 0.03

        if p >= p_req and e >= e_req:
            return "BET_YES"
        return "NO_BET"

    def _resolve_market_probabilities(
        self,
        market_yes: Optional[float],
        market_no: Optional[float],
        fallback_yes: float,
        fallback_no: float,
        warnings: List[str],
    ) -> tuple[float, float, str]:
        p_yes = self._prob_input_to_unit(market_yes)
        p_no = self._prob_input_to_unit(market_no)

        if p_yes is not None and p_no is not None:
            s = p_yes + p_no
            if abs(s - 1.0) > 0.08:
                warnings.append(f"Kalshi probs do not sum near 1.0 (sum={s:.4f}); normalizing.")
                p_yes = p_yes / s
                p_no = p_no / s
            return float(np.clip(p_yes, 0.01, 0.99)), float(np.clip(p_no, 0.01, 0.99)), "kalshi_cli_both"

        if p_yes is not None:
            return float(np.clip(p_yes, 0.01, 0.99)), float(np.clip(1.0 - p_yes, 0.01, 0.99)), "kalshi_cli_yes_only"

        if p_no is not None:
            return float(np.clip(1.0 - p_no, 0.01, 0.99)), float(np.clip(p_no, 0.01, 0.99)), "kalshi_cli_no_only"

        warnings.append("Kalshi market probabilities not provided; using model-implied market probability baseline.")
        return float(np.clip(fallback_yes, 0.01, 0.99)), float(np.clip(fallback_no, 0.01, 0.99)), "model_implied_fallback"

    @staticmethod
    def fit_calibrator(raw_pred: np.ndarray, y: np.ndarray, method: str) -> object:
        m = method.lower()
        yp = np.asarray(y).reshape(-1)
        rp = np.asarray(raw_pred, dtype=float).reshape(-1)
        if len(yp) == 0 or len(np.unique(yp)) < 2:
            return {"method": "none", "model": None}
        if m == "none":
            return {"method": "none", "model": None}
        if m == "platt":
            if LogisticRegression is None:
                return {"method": "none", "model": None}
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(rp.reshape(-1, 1), yp)
            return {"method": "platt", "model": lr}
        if m == "isotonic":
            if IsotonicRegression is None:
                return {"method": "none", "model": None}
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(rp, yp)
            return {"method": "isotonic", "model": iso}
        raise ValueError("Unknown calibration method")

    @staticmethod
    def apply_calibrator(cal: object, raw_pred: np.ndarray) -> np.ndarray:
        p = np.asarray(raw_pred, dtype=float).reshape(-1)
        if not isinstance(cal, dict):
            return np.clip(p, 1e-6, 1 - 1e-6)
        m = str(cal.get("method", "none")).lower()
        model = cal.get("model")
        if m == "none" or model is None:
            return np.clip(p, 1e-6, 1 - 1e-6)
        if m == "platt":
            return np.clip(model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        if m == "isotonic":
            return np.clip(model.predict(p), 1e-6, 1 - 1e-6)
        return np.clip(p, 1e-6, 1 - 1e-6)

    def _candidate_calibrator_paths(self) -> List[Path]:
        env_path = os.getenv("CALIBRATOR_ARTIFACT_PATH", "").strip()
        cands: List[Path] = []
        if env_path:
            cands.append(Path(env_path))

        cands.extend(
            [
                self.base_dir / "artifacts" / "models" / "btcnew_calibrator.joblib",
                self.base_dir.parent / "BTC" / "STRUCTURAL_BTC_ENGINE" / "artifacts" / "models" / "btc_mispricing_model.joblib",
                self.base_dir.parent / "BTC" / "model_artifacts" / "btc_prob_model_1m_10m.joblib",
            ]
        )
        return cands

    def _ensure_calibrator_loaded(self, warnings: List[str]) -> None:
        if self._calibrator_ready:
            return
        self._calibrator_ready = True

        if joblib is None:
            warnings.append("joblib unavailable; calibration disabled (install with: pip install -r requirements.txt)")
            self._calibrator = {"method": "none", "model": None}
            self._calibrator_source = "none"
            return

        for path in self._candidate_calibrator_paths():
            try:
                if not path.exists():
                    continue
                obj = joblib.load(path)
                cal = None
                if isinstance(obj, dict) and "calibrator" in obj:
                    cal = obj.get("calibrator")
                elif isinstance(obj, dict) and {"method", "model"}.issubset(set(obj.keys())):
                    cal = obj
                if isinstance(cal, dict):
                    self._calibrator = cal
                    self._calibrator_source = str(path)
                    return
            except Exception as exc:
                warnings.append(f"Failed loading calibrator from {path}: {exc}")

        warnings.append("No calibrator artifact found; using raw model probability")
        self._calibrator = {"method": "none", "model": None}
        self._calibrator_source = "none"

    def _coinbase_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        api_key = os.getenv("COINBASE_API_KEY", "").strip()
        header_name = os.getenv("COINBASE_API_KEY_HEADER", "").strip()
        if api_key and header_name:
            headers[header_name] = api_key
        return headers

    def _fetch_spot_price(self, warnings: List[str]) -> float:
        product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
        headers = self._coinbase_headers() or None

        try:
            data = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/ticker", headers=headers)
            return float(data["price"])
        except Exception as exc:
            warnings.append(f"Coinbase ticker unavailable: {exc}")

        # fallback: use latest 5m candle close
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=2)
            params = {
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "granularity": "300",
            }
            rows = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/candles", params=params, headers=headers)
            df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
            if df.empty:
                raise ValueError("no candles")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])
            if df.empty:
                raise ValueError("all candle closes NaN")
            return float(df["close"].iloc[-1])
        except Exception as exc:
            warnings.append(f"Coinbase candle fallback unavailable: {exc}")
            raise RuntimeError("Cannot compute probabilities without spot price") from exc

    def _collect_factors(self, horizon_minutes: int, warnings: List[str]) -> Dict[str, float]:
        factors: Dict[str, float] = {}
        factors.update(self._fetch_orderbook_trade_factors(warnings))
        factors.update(self._fetch_coinalyze_factors(warnings))
        futures_ex = self._fetch_futures_exchange_factors(warnings)
        if futures_ex:
            factors.update(futures_ex)
            if os.getenv("FUTURES_OVERRIDE_FEATURES", "true").strip().lower() in {"1", "true", "yes"}:
                if "futures_funding_rate" in futures_ex:
                    factors["funding_rate"] = futures_ex["futures_funding_rate"]
                if "futures_open_interest_change_1m" in futures_ex:
                    factors["open_interest_change"] = futures_ex["futures_open_interest_change_1m"]
                if "futures_liquidation_imbalance" in futures_ex:
                    factors["liquidation_imbalance"] = futures_ex["futures_liquidation_imbalance"]
                if "futures_orderbook_imbalance" in futures_ex:
                    factors["orderbook_imbalance"] = futures_ex["futures_orderbook_imbalance"]
        candles, candle_interval_minutes = self._fetch_candles_7d(horizon_minutes, warnings)
        factors["price_action_candle_minutes"] = float(candle_interval_minutes)
        factors.update(self._price_action_factors(candles, horizon_minutes, candle_interval_minutes, warnings))
        factors.update(self._macro_factors(warnings))
        factors.update(self._behavioral_factors(warnings, factors.get("long_short_ratio", 1.0)))
        return factors

    def _fetch_orderbook_trade_factors(self, warnings: List[str]) -> Dict[str, float]:
        product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
        headers = self._coinbase_headers() or None
        out = {
            "orderbook_imbalance": 0.0,
            "bid_ask_spread": 0.0004,
            "trade_flow_imbalance": 0.0,
            "trade_flow_buy_share": 0.5,
            "trade_flow_notional_usd": 0.0,
        }

        try:
            book = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/book", params={"level": "2"}, headers=headers)
            bids = book.get("bids", [])[:200]
            asks = book.get("asks", [])[:200]

            bid_notional = sum(float(px) * float(sz) for px, sz, *_ in bids)
            ask_notional = sum(float(px) * float(sz) for px, sz, *_ in asks)
            out["orderbook_imbalance"] = (bid_notional - ask_notional) / (bid_notional + ask_notional + EPS)

            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2.0
                out["bid_ask_spread"] = (best_ask - best_bid) / (mid + EPS)
        except Exception as exc:
            warnings.append(f"Order book unavailable: {exc}")

        try:
            trades = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/trades", params={"limit": "1000"}, headers=headers)
            df = pd.DataFrame(trades)
            if not df.empty:
                df["size"] = pd.to_numeric(df.get("size"), errors="coerce")
                df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
                maker_side = df.get("side", "").astype(str).str.lower()
                # Coinbase side is maker side; maker sell => taker buy.
                taker_buy = (maker_side == "sell").astype(float)
                notional = (df["size"] * df["price"]).fillna(0.0)
                buy_notional = float((notional * taker_buy).sum())
                sell_notional = float((notional * (1.0 - taker_buy)).sum())
                total = buy_notional + sell_notional
                if total > 0:
                    out["trade_flow_imbalance"] = (buy_notional - sell_notional) / total
                    out["trade_flow_buy_share"] = buy_notional / total
                    out["trade_flow_notional_usd"] = total
        except Exception as exc:
            warnings.append(f"Trade flow unavailable: {exc}")

        return out

    @staticmethod
    def _coinalyze_history_df(payload: Any) -> pd.DataFrame:
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict) and isinstance(payload[0].get("history"), list):
                return pd.DataFrame(payload[0]["history"])
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if isinstance(payload.get("history"), list):
                return pd.DataFrame(payload["history"])
            if isinstance(payload.get("data"), list):
                data = payload["data"]
                if data and isinstance(data[0], dict) and isinstance(data[0].get("history"), list):
                    return pd.DataFrame(data[0]["history"])
                return pd.DataFrame(data)
        return pd.DataFrame()

    @staticmethod
    def _last_numeric(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
        for c in candidates:
            if c in df.columns:
                ser = pd.to_numeric(df[c], errors="coerce").dropna()
                if not ser.empty:
                    return float(ser.iloc[-1])
        return None

    def _fetch_coinalyze_factors(self, warnings: List[str]) -> Dict[str, float]:
        out = {
            "funding_rate": 0.0,
            "open_interest_change": 0.0,
            "liquidation_imbalance": 0.0,
            "perp_premium": 0.0,
            "long_short_ratio": 1.0,
        }

        api_key = os.getenv("COINALYZE_API_KEY", "").strip()
        symbol = os.getenv("COINALYZE_SYMBOL", "BTCUSDT_PERP.A").strip()
        if not api_key:
            warnings.append("COINALYZE_API_KEY missing; using neutral coinalyze factors")
            return out

        headers = {"api_key": api_key}
        now = int(datetime.now(timezone.utc).timestamp())
        start = now - 4 * 3600

        try:
            fr = self._safe_get_json(
                f"{COINALYZE_BASE}/funding-rate-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            fr_df = self._coinalyze_history_df(fr)
            v = self._last_numeric(fr_df, ["c", "value", "funding_rate"])
            if v is not None:
                out["funding_rate"] = v
        except Exception as exc:
            warnings.append(f"Coinalyze funding unavailable: {exc}")

        try:
            oi = self._safe_get_json(
                f"{COINALYZE_BASE}/open-interest-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            oi_df = self._coinalyze_history_df(oi)
            for c in ["c", "value", "open_interest"]:
                if c in oi_df.columns:
                    ser = pd.to_numeric(oi_df[c], errors="coerce").dropna()
                    break
            else:
                ser = pd.Series(dtype=float)
            if not ser.empty:
                if len(ser) >= 2 and ser.iloc[-2] != 0:
                    out["open_interest_change"] = float(ser.iloc[-1] / ser.iloc[-2] - 1.0)
        except Exception as exc:
            warnings.append(f"Coinalyze open interest unavailable: {exc}")

        try:
            liq = self._safe_get_json(
                f"{COINALYZE_BASE}/liquidation-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            liq_df = self._coinalyze_history_df(liq)
            long_col = next((c for c in ["long_liquidation_usd", "longs", "long"] if c in liq_df.columns), None)
            short_col = next((c for c in ["short_liquidation_usd", "shorts", "short"] if c in liq_df.columns), None)
            if long_col is None and "l" in liq_df.columns:
                long_col = "l"
            if short_col is None and "s" in liq_df.columns:
                short_col = "s"
            longs = pd.to_numeric(liq_df[long_col], errors="coerce").fillna(0.0) if long_col else pd.Series(dtype=float)
            shorts = pd.to_numeric(liq_df[short_col], errors="coerce").fillna(0.0) if short_col else pd.Series(dtype=float)
            lsum = float(longs.sum()) if not longs.empty else 0.0
            ssum = float(shorts.sum()) if not shorts.empty else 0.0
            if lsum > 0.0 or ssum > 0.0:
                out["liquidation_imbalance"] = (ssum - lsum) / (ssum + lsum + EPS)
        except Exception as exc:
            warnings.append(f"Coinalyze liquidations unavailable: {exc}")

        premium_loaded = False
        premium_errors: List[str] = []
        symbol_candidates = [symbol, symbol.replace(".A", ""), "BTCUSDT_PERP.A", "BTCUSDT_PERP"]
        dedup_symbols: List[str] = []
        seen = set()
        for s in symbol_candidates:
            k = s.strip()
            if k and k not in seen:
                dedup_symbols.append(k)
                seen.add(k)
        for sym in dedup_symbols:
            try:
                premium = self._safe_get_json(
                    f"{COINALYZE_BASE}/premium-index-history",
                    params={"symbols": sym, "from": start, "to": now, "interval": "1hour"},
                    headers=headers,
                )
                pm_df = self._coinalyze_history_df(premium)
                v = self._last_numeric(pm_df, ["c", "value", "premium", "p"])
                if v is not None:
                    out["perp_premium"] = v
                    premium_loaded = True
                    break
            except Exception as exc:
                premium_errors.append(str(exc))
                continue
        # Premium endpoint is often unavailable on free tiers. Fall back to predicted funding proxy.
        if not premium_loaded:
            for sym in dedup_symbols:
                try:
                    pf = self._safe_get_json(
                        f"{COINALYZE_BASE}/predicted-funding-rate-history",
                        params={"symbols": sym, "from": start, "to": now, "interval": "1hour"},
                        headers=headers,
                    )
                    pf_df = self._coinalyze_history_df(pf)
                    v = self._last_numeric(pf_df, ["c", "value"])
                    if v is not None:
                        out["perp_premium"] = v
                        premium_loaded = True
                        break
                except Exception as exc:
                    premium_errors.append(str(exc))
                    continue
        if not premium_loaded and premium_errors:
            warnings.append(f"Coinalyze perp premium unavailable: {premium_errors[-1]}")

        try:
            ls = self._safe_get_json(
                f"{COINALYZE_BASE}/long-short-ratio-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            ls_df = self._coinalyze_history_df(ls)
            v = self._last_numeric(ls_df, ["r", "c", "value"])
            if v is not None:
                out["long_short_ratio"] = v
        except Exception as exc:
            warnings.append(f"Coinalyze long/short ratio unavailable: {exc}")

        return out

    @staticmethod
    def _orderbook_imbalance_from_depth(bids: List[List[Any]], asks: List[List[Any]]) -> Dict[str, float]:
        bid_notional = 0.0
        ask_notional = 0.0
        for row in bids:
            if len(row) < 2:
                continue
            bid_notional += float(row[0]) * float(row[1])
        for row in asks:
            if len(row) < 2:
                continue
            ask_notional += float(row[0]) * float(row[1])
        total = bid_notional + ask_notional + EPS
        imbalance = (bid_notional - ask_notional) / total
        ratio = bid_notional / (ask_notional + EPS)
        return {"futures_orderbook_imbalance": float(imbalance), "futures_orderbook_ratio": float(ratio)}

    def _fetch_futures_exchange_factors(self, warnings: List[str]) -> Dict[str, float]:
        provider = os.getenv("FUTURES_EXCHANGE", "").strip().lower()
        if provider in {"binance", "binanceusdm"}:
            return self._fetch_binance_futures_factors(warnings)
        if provider == "bybit":
            return self._fetch_bybit_futures_factors(warnings)
        if provider == "okx":
            return self._fetch_okx_futures_factors(warnings)
        if provider in {"bydfi", "bydoxe"}:
            return self._fetch_bydfi_futures_factors(warnings)
        return {}

    def _fetch_binance_futures_factors(self, warnings: List[str]) -> Dict[str, float]:
        symbol = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT").strip().upper()
        out: Dict[str, float] = {}

        try:
            payload = self._safe_get_json(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex", params={"symbol": symbol}
            )
            fr = payload.get("lastFundingRate") or payload.get("fundingRate")
            if fr is not None:
                out["futures_funding_rate"] = float(fr)
        except Exception as exc:
            warnings.append(f"Binance funding unavailable: {exc}")

        try:
            hist = self._safe_get_json(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterestHist",
                params={"symbol": symbol, "period": "1m", "limit": "2"},
            )
            if isinstance(hist, list) and len(hist) >= 2:
                v0 = float(hist[-2].get("sumOpenInterest", 0.0))
                v1 = float(hist[-1].get("sumOpenInterest", 0.0))
                if v0 > 0:
                    out["futures_open_interest_change_1m"] = float(v1 / v0 - 1.0)
        except Exception as exc:
            warnings.append(f"Binance open interest (1m) unavailable: {exc}")

        try:
            hist = self._safe_get_json(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterestHist",
                params={"symbol": symbol, "period": "5m", "limit": "2"},
            )
            if isinstance(hist, list) and len(hist) >= 2:
                v0 = float(hist[-2].get("sumOpenInterest", 0.0))
                v1 = float(hist[-1].get("sumOpenInterest", 0.0))
                if v0 > 0:
                    out["futures_open_interest_change_5m"] = float(v1 / v0 - 1.0)
        except Exception as exc:
            warnings.append(f"Binance open interest (5m) unavailable: {exc}")

        try:
            depth = self._safe_get_json(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/depth", params={"symbol": symbol, "limit": "100"}
            )
            bids = depth.get("bids", []) if isinstance(depth, dict) else []
            asks = depth.get("asks", []) if isinstance(depth, dict) else []
            if bids and asks:
                out.update(self._orderbook_imbalance_from_depth(bids, asks))
        except Exception as exc:
            warnings.append(f"Binance orderbook unavailable: {exc}")

        return out

    def _fetch_bybit_futures_factors(self, warnings: List[str]) -> Dict[str, float]:
        symbol = os.getenv("BYBIT_FUTURES_SYMBOL", "BTCUSDT").strip().upper()
        out: Dict[str, float] = {}

        try:
            payload = self._safe_get_json(
                f"{BYBIT_BASE}/v5/market/funding/history",
                params={"category": "linear", "symbol": symbol, "limit": "1"},
            )
            data = payload.get("result", {}).get("list", []) if isinstance(payload, dict) else []
            if data:
                fr = data[-1].get("fundingRate")
                if fr is not None:
                    out["futures_funding_rate"] = float(fr)
        except Exception as exc:
            warnings.append(f"Bybit funding unavailable: {exc}")

        try:
            payload = self._safe_get_json(
                f"{BYBIT_BASE}/v5/market/open-interest",
                params={"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": "2"},
            )
            data = payload.get("result", {}).get("list", []) if isinstance(payload, dict) else []
            if len(data) >= 2:
                v0 = float(data[-2].get("openInterest", 0.0))
                v1 = float(data[-1].get("openInterest", 0.0))
                if v0 > 0:
                    out["futures_open_interest_change_5m"] = float(v1 / v0 - 1.0)
        except Exception as exc:
            warnings.append(f"Bybit open interest unavailable: {exc}")

        try:
            payload = self._safe_get_json(
                f"{BYBIT_BASE}/v5/market/orderbook",
                params={"category": "linear", "symbol": symbol, "limit": "100"},
            )
            data = payload.get("result", {}) if isinstance(payload, dict) else {}
            bids = data.get("b", []) if isinstance(data, dict) else []
            asks = data.get("a", []) if isinstance(data, dict) else []
            if bids and asks:
                out.update(self._orderbook_imbalance_from_depth(bids, asks))
        except Exception as exc:
            warnings.append(f"Bybit orderbook unavailable: {exc}")

        return out

    def _fetch_okx_futures_factors(self, warnings: List[str]) -> Dict[str, float]:
        inst_id = os.getenv("OKX_FUTURES_SYMBOL", "BTC-USDT-SWAP").strip().upper()
        out: Dict[str, float] = {}

        try:
            payload = self._safe_get_json(
                f"{OKX_BASE}/api/v5/public/funding-rate", params={"instId": inst_id}
            )
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if data:
                fr = data[-1].get("fundingRate")
                if fr is not None:
                    out["futures_funding_rate"] = float(fr)
        except Exception as exc:
            warnings.append(f"OKX funding unavailable: {exc}")

        try:
            payload = self._safe_get_json(
                f"{OKX_BASE}/api/v5/public/open-interest", params={"instId": inst_id}
            )
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if len(data) >= 2:
                v0 = float(data[-2].get("oi", 0.0))
                v1 = float(data[-1].get("oi", 0.0))
                if v0 > 0:
                    out["futures_open_interest_change_1m"] = float(v1 / v0 - 1.0)
        except Exception as exc:
            warnings.append(f"OKX open interest unavailable: {exc}")

        try:
            payload = self._safe_get_json(
                f"{OKX_BASE}/api/v5/market/books", params={"instId": inst_id, "sz": "200"}
            )
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if data:
                bids = data[0].get("bids", [])
                asks = data[0].get("asks", [])
                if bids and asks:
                    out.update(self._orderbook_imbalance_from_depth(bids, asks))
        except Exception as exc:
            warnings.append(f"OKX orderbook unavailable: {exc}")

        try:
            payload = self._safe_get_json(
                f"{OKX_BASE}/api/v5/public/liquidation-orders",
                params={"instType": "SWAP", "instId": inst_id, "limit": "100"},
            )
            data = payload.get("data", []) if isinstance(payload, dict) else []
            long_notional = 0.0
            short_notional = 0.0
            for row in data:
                side = str(row.get("side", "")).lower()
                sz = float(row.get("sz", 0.0))
                px = float(row.get("px", 0.0))
                notional = sz * px
                if side == "buy":
                    short_notional += notional
                elif side == "sell":
                    long_notional += notional
            total = long_notional + short_notional
            if total > 0:
                out["futures_liquidation_imbalance"] = float((short_notional - long_notional) / total)
                out["futures_liquidation_pressure"] = float(total)
        except Exception as exc:
            warnings.append(f"OKX liquidation feed unavailable: {exc}")

        return out

    @staticmethod
    def _bydfi_data(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return None
        code = str(payload.get("code", "")).strip()
        if code and code not in {"00000", "0"}:
            return None
        return payload.get("data")

    @staticmethod
    def _bydfi_pick_symbol_rows(data: Any, symbol: str) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            rows = [data]
        elif isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        else:
            return []
        sym = symbol.strip().upper()
        return [r for r in rows if str(r.get("symbol", "")).strip().upper() == sym] or rows

    def _fetch_bydfi_futures_factors(self, warnings: List[str]) -> Dict[str, float]:
        base = os.getenv("BYDFI_FUTURES_BASE", BYDFI_BASE).strip().rstrip("/")
        symbol = os.getenv("BYDFI_FUTURES_SYMBOL", "BTCUSDT").strip().upper()
        out: Dict[str, float] = {}

        # Funding: latest historical funding rate.
        try:
            payload = self._safe_get_json(
                f"{base}/api/v1/future/market/history-fund-rate",
                params={"symbol": symbol, "limit": "1"},
            )
            data = self._bydfi_data(payload)
            rows = self._bydfi_pick_symbol_rows(data, symbol)
            if rows:
                fr = rows[-1].get("fundingRate")
                if fr is not None:
                    out["futures_funding_rate"] = float(fr)
        except Exception as exc:
            warnings.append(f"BYDFi funding unavailable: {exc}")

        # Open interest: only snapshot is exposed in public endpoint.
        try:
            payload = self._safe_get_json(
                f"{base}/api/v1/future/market/open-interest",
                params={"symbol": symbol},
            )
            data = self._bydfi_data(payload)
            if isinstance(data, dict):
                oi_list = data.get("openInterestList", []) if isinstance(data.get("openInterestList"), list) else []
                if oi_list:
                    row = oi_list[-1]
                    size = row.get("size")
                    if size is not None:
                        out["futures_open_interest_size"] = float(size)
        except Exception as exc:
            warnings.append(f"BYDFi open interest unavailable: {exc}")

        # Orderbook signal from best bid/ask ticker endpoint.
        try:
            payload = self._safe_get_json(
                f"{base}/api/v1/future/market/book-ticker",
                params={"symbol": symbol},
            )
            data = self._bydfi_data(payload)
            rows = self._bydfi_pick_symbol_rows(data, symbol)
            if rows:
                row = rows[-1]
                bid_p = float(row.get("bidPrice", 0.0))
                ask_p = float(row.get("askPrice", 0.0))
                bid_q = float(row.get("bidQty", 0.0))
                ask_q = float(row.get("askQty", 0.0))
                bid_notional = bid_p * bid_q
                ask_notional = ask_p * ask_q
                total = bid_notional + ask_notional + EPS
                out["futures_orderbook_imbalance"] = float((bid_notional - ask_notional) / total)
                out["futures_orderbook_ratio"] = float(bid_notional / (ask_notional + EPS))
        except Exception as exc:
            warnings.append(f"BYDFi orderbook unavailable: {exc}")

        return out

    @staticmethod
    def _select_candle_interval_minutes(horizon_minutes: int) -> int:
        if horizon_minutes < 7:
            return 1
        if horizon_minutes <= 20:
            return 3
        return 5

    def _fetch_coinbase_candles(
        self,
        product: str,
        granularity_seconds: int,
        start: datetime,
        end: datetime,
        headers: Optional[Dict[str, str]],
    ) -> pd.DataFrame:
        max_points = 300
        chunk_seconds = granularity_seconds * max_points
        cur = start
        rows: List[Any] = []

        while cur < end:
            nxt = min(cur + timedelta(seconds=chunk_seconds), end)
            params = {
                "start": cur.isoformat().replace("+00:00", "Z"),
                "end": nxt.isoformat().replace("+00:00", "Z"),
                "granularity": str(granularity_seconds),
            }
            payload = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/candles", params=params, headers=headers)
            if isinstance(payload, list) and payload:
                rows.extend(payload)
            cur = nxt

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        for c in ["low", "high", "open", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().drop_duplicates("time").sort_values("time").set_index("time")
        return df

    def _fetch_candles_7d(self, horizon_minutes: int, warnings: List[str]) -> tuple[pd.DataFrame, int]:
        product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
        headers = self._coinbase_headers() or None
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        interval_minutes = 1

        try:
            df_1m = self._fetch_coinbase_candles(product, 60, start, end, headers)
            if df_1m.empty:
                warnings.append("No 1m candle data returned; using neutral price-action factors")
                return pd.DataFrame(), interval_minutes
            return df_1m, interval_minutes
        except Exception as exc:
            warnings.append(f"Candle fetch unavailable: {exc}")
            return pd.DataFrame(), interval_minutes

    def _price_action_factors(
        self,
        df: pd.DataFrame,
        horizon_minutes: int,
        candle_minutes: int,
        warnings: List[str],
    ) -> Dict[str, float]:
        out = {
            "momentum_1h": 0.0,
            "short_term_return": 0.0,
            "return_1m": 0.0,
            "return_3m": 0.0,
            "return_5m": 0.0,
            "return_10m": 0.0,
            "momentum_acceleration": 0.0,
            "realized_vol_annual": 0.55,
            "realized_vol_expansion": 0.0,
            "vol_1m": 0.0,
            "vol_5m": 0.0,
            "vol_15m": 0.0,
            "vwap_deviation": 0.0,
            "distance_to_vwap": 0.0,
            "distance_to_high_15m": 0.0,
            "distance_to_low_15m": 0.0,
            "volume_spike": 1.0,
            "local_range_break": 0.0,
            "ema_3": 0.0,
            "ema_5": 0.0,
            "ema_10": 0.0,
            "price_minus_ema_3": 0.0,
            "price_minus_ema_10": 0.0,
            "ema_3_minus_ema_10": 0.0,
        }

        if df.empty or len(df) < 50:
            warnings.append("Insufficient candles; using neutral price-action factors")
            return out

        close = df["close"]
        rets = np.log(close / close.shift(1)).dropna()
        if rets.empty:
            return out

        bars_1h = max(1, 60 // candle_minutes)
        bars_horizon = max(1, horizon_minutes // candle_minutes)

        if len(close) > bars_1h:
            out["momentum_1h"] = float(close.iloc[-1] / close.iloc[-bars_1h] - 1.0)
        if len(close) > bars_horizon:
            out["short_term_return"] = float(close.iloc[-1] / close.iloc[-bars_horizon] - 1.0)

        def _ret_n(minutes: int) -> float:
            bars = max(1, int(round(minutes / max(candle_minutes, 1))))
            if len(close) <= bars:
                return 0.0
            return float(close.iloc[-1] / close.iloc[-1 - bars] - 1.0)

        out["return_1m"] = _ret_n(1)
        out["return_3m"] = _ret_n(3)
        out["return_5m"] = _ret_n(5)
        out["return_10m"] = _ret_n(10)
        out["momentum_acceleration"] = float(out["return_1m"] - out["return_5m"])

        scale = math.sqrt((365.0 * 24.0 * 60.0) / float(candle_minutes))
        rv = float(rets.tail(min(len(rets), 288)).std(ddof=0) * scale)
        rv_short = float(rets.tail(min(len(rets), 24)).std(ddof=0) * scale)

        out["realized_vol_annual"] = max(rv, 0.08)
        out["realized_vol_expansion"] = (rv_short / (rv + EPS)) - 1.0
        out["vol_1m"] = float(rets.tail(min(len(rets), max(3, int(round(1 / max(candle_minutes, 1)))))).std(ddof=0) * scale)
        out["vol_5m"] = float(rets.tail(min(len(rets), max(5, int(round(5 / max(candle_minutes, 1)))))).std(ddof=0) * scale)
        out["vol_15m"] = float(rets.tail(min(len(rets), max(15, int(round(15 / max(candle_minutes, 1)))))).std(ddof=0) * scale)

        bars_4h = max(2, 240 // candle_minutes)
        bars_2h = max(2, 120 // candle_minutes)
        bars_15m = max(2, 15 // candle_minutes)
        vwap_num = (df["close"] * df["volume"]).tail(bars_4h).sum()
        vwap_den = df["volume"].tail(bars_4h).sum() + EPS
        vwap = vwap_num / vwap_den
        out["vwap_deviation"] = float((close.iloc[-1] - vwap) / (vwap + EPS))
        out["distance_to_vwap"] = out["vwap_deviation"]

        last_close = float(close.iloc[-1])
        high_15m = float(df["high"].tail(bars_15m).max())
        low_15m = float(df["low"].tail(bars_15m).min())
        out["distance_to_high_15m"] = float((high_15m - last_close) / (last_close + EPS))
        out["distance_to_low_15m"] = float((last_close - low_15m) / (last_close + EPS))

        vol = df["volume"].tail(bars_4h)
        if len(vol) >= 10:
            out["volume_spike"] = float(vol.iloc[-1] / (vol.median() + EPS))

        recent_high = float(df["high"].tail(bars_2h).max())
        recent_low = float(df["low"].tail(bars_2h).min())
        if last_close > recent_high:
            out["local_range_break"] = 1.0
        elif last_close < recent_low:
            out["local_range_break"] = -1.0

        bars_ema3 = max(2, int(round(3 / max(candle_minutes, 1))))
        bars_ema5 = max(2, int(round(5 / max(candle_minutes, 1))))
        bars_ema10 = max(2, int(round(10 / max(candle_minutes, 1))))
        ema3 = float(close.ewm(span=bars_ema3, adjust=False).mean().iloc[-1])
        ema5 = float(close.ewm(span=bars_ema5, adjust=False).mean().iloc[-1])
        ema10 = float(close.ewm(span=bars_ema10, adjust=False).mean().iloc[-1])
        out["ema_3"] = ema3
        out["ema_5"] = ema5
        out["ema_10"] = ema10
        out["price_minus_ema_3"] = float((last_close - ema3) / (last_close + EPS))
        out["price_minus_ema_10"] = float((last_close - ema10) / (last_close + EPS))
        out["ema_3_minus_ema_10"] = float((ema3 - ema10) / (last_close + EPS))

        return out

    def _macro_factors(self, warnings: List[str]) -> Dict[str, float]:
        out = {
            "spx_btc_corr": 0.0,
            "dxy_return": 0.0,
            "bond_yield_change": 0.0,
            "cpi_yoy": 0.0,
            "etf_flows_z": 0.0,
            "fomc_event_risk": 0.0,
        }

        fred_key = os.getenv("FRED_API_KEY", "").strip()
        if fred_key:
            try:
                spx = self._fred_series("SP500", fred_key)
                dxy = self._fred_series("DTWEXBGS", fred_key)
                y10 = self._fred_series("DGS10", fred_key)
                cpi = self._fred_series("CPIAUCSL", fred_key)
                btc = self._daily_btc_series(warnings)

                if len(spx) > 30 and len(btc) > 30:
                    joined = pd.concat([spx.pct_change(), btc.pct_change()], axis=1, join="inner").dropna()
                    if len(joined) > 20:
                        out["spx_btc_corr"] = float(joined.iloc[:, 0].tail(30).corr(joined.iloc[:, 1].tail(30)))
                if len(dxy) > 2:
                    out["dxy_return"] = float(dxy.iloc[-1] / dxy.iloc[-2] - 1.0)
                if len(y10) > 2:
                    out["bond_yield_change"] = float(y10.iloc[-1] - y10.iloc[-2])
                if len(cpi) > 13 and cpi.iloc[-13] != 0:
                    out["cpi_yoy"] = float(cpi.iloc[-1] / cpi.iloc[-13] - 1.0)
            except Exception as exc:
                warnings.append(f"FRED macro factors unavailable: {exc}")
        else:
            warnings.append("FRED_API_KEY missing; using neutral macro factors")

        out["etf_flows_z"] = self._etf_flow_factor(warnings)
        out["fomc_event_risk"] = self._fomc_event_risk()
        return out

    def _daily_btc_series(self, warnings: List[str]) -> pd.Series:
        product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
        headers = self._coinbase_headers() or None
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=140)
        params = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "granularity": "86400",
        }
        try:
            rows = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/candles", params=params, headers=headers)
            df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
            if df.empty:
                return pd.Series(dtype=float)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            s = df.dropna().drop_duplicates("time").sort_values("time").set_index("time")["close"]
            s.index = s.index.tz_convert(None).normalize()
            return s
        except Exception as exc:
            warnings.append(f"BTC daily series unavailable: {exc}")
            return pd.Series(dtype=float)

    def _fred_series(self, series_id: str, api_key: str) -> pd.Series:
        data = self._safe_get_json(
            f"{FRED_BASE}/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "asc",
                "limit": "500",
            },
        )
        rows = []
        for obs in data.get("observations", []):
            value = obs.get("value")
            if value in {None, "."}:
                continue
            rows.append((pd.Timestamp(obs["date"]), float(value)))
        if not rows:
            return pd.Series(dtype=float)
        return pd.Series([v for _, v in rows], index=[t for t, _ in rows], dtype=float)

    def _etf_flow_factor(self, warnings: List[str]) -> float:
        url = os.getenv("BTC_ETF_FLOW_API_URL", "").strip()
        if not url:
            return 0.0

        key = os.getenv("BTC_ETF_FLOW_API_KEY", "").strip()
        header_name = os.getenv("BTC_ETF_FLOW_API_KEY_HEADER", "").strip()
        param_name = os.getenv("BTC_ETF_FLOW_API_KEY_PARAM", "").strip()
        headers: Dict[str, str] = {}
        params: Dict[str, Any] = {}
        if key and header_name:
            headers[header_name] = key
        if key and param_name:
            params[param_name] = key

        try:
            payload = self._safe_get_json(url, params=params or None, headers=headers or None)
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(rows, list) or not rows:
                return 0.0
            df = pd.DataFrame(rows)
            col = None
            for candidate in ["netFlow", "net_flow", "total_net_flow", "flow"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                return 0.0
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(ser) < 5:
                return 0.0
            window = ser.tail(min(30, len(ser)))
            z = (window.iloc[-1] - window.mean()) / (window.std(ddof=0) + EPS)
            return float(np.clip(z / 4.0, -1.0, 1.0))
        except Exception as exc:
            code = self._http_status(exc)
            if code in {401, 403, 404}:
                return 0.0
            warnings.append(f"ETF flow unavailable: {exc}")
            return 0.0

    def _fomc_event_risk(self) -> float:
        dates_raw = os.getenv("FOMC_DATES", "").strip()
        if not dates_raw:
            return 0.0
        today = datetime.now(timezone.utc).date()
        next_deltas: List[int] = []
        for token in dates_raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                d = datetime.strptime(token, "%Y-%m-%d").date()
                delta = (d - today).days
                if delta >= 0:
                    next_deltas.append(delta)
            except Exception:
                continue
        if not next_deltas:
            return 0.0
        return 1.0 if min(next_deltas) <= 7 else 0.0

    def _behavioral_factors(self, warnings: List[str], long_short_ratio: float) -> Dict[str, float]:
        out = {
            "fear_greed": 0.0,
            "social_velocity": 0.0,
            "long_short_ratio_signal": float(np.clip(1.2 - long_short_ratio, -1.0, 1.0)),
        }

        try:
            fg = self._safe_get_json(FEAR_GREED_URL)
            data = fg.get("data", [])
            if data:
                value = float(data[0]["value"])
                out["fear_greed"] = (value - 50.0) / 50.0
        except Exception as exc:
            warnings.append(f"Fear & Greed unavailable: {exc}")

        url = os.getenv("SOCIAL_VELOCITY_API_URL", "").strip()
        if not url:
            return out

        key = os.getenv("SOCIAL_VELOCITY_API_KEY", "").strip()
        header_name = os.getenv("SOCIAL_VELOCITY_API_KEY_HEADER", "").strip()
        param_name = os.getenv("SOCIAL_VELOCITY_API_KEY_PARAM", "").strip()
        headers: Dict[str, str] = {}
        params: Dict[str, Any] = {}
        if key and header_name:
            headers[header_name] = key
        if key and param_name:
            params[param_name] = key

        try:
            raw = self._safe_get_json(url, params=params or None, headers=headers or None)
            out["social_velocity"] = self._extract_social_velocity(raw)
        except Exception as exc:
            code = self._http_status(exc)
            if code == 429:
                return out
            warnings.append(f"Social velocity unavailable: {exc}")

        return out

    @staticmethod
    def _extract_social_velocity(raw: Any) -> float:
        if isinstance(raw, dict):
            timeline = raw.get("timeline", [])
            if isinstance(timeline, list) and len(timeline) >= 5:
                vals: List[float] = []
                for row in timeline:
                    for key in ["value", "count", "norm"]:
                        if key in row:
                            try:
                                vals.append(float(row[key]))
                                break
                            except Exception:
                                pass
                if len(vals) >= 5:
                    arr = np.array(vals, dtype=float)
                    z = (arr[-1] - arr.mean()) / (arr.std(ddof=0) + EPS)
                    return float(np.clip(z / 4.0, -1.0, 1.0))

            for key in ["data", "results", "values"]:
                seq = raw.get(key)
                if isinstance(seq, list) and len(seq) >= 5:
                    nums = pd.to_numeric(pd.Series(seq), errors="coerce").dropna()
                    if len(nums) >= 5:
                        z = (nums.iloc[-1] - nums.mean()) / (nums.std(ddof=0) + EPS)
                        return float(np.clip(z / 4.0, -1.0, 1.0))

        if isinstance(raw, list) and len(raw) >= 5:
            nums = pd.to_numeric(pd.Series(raw), errors="coerce").dropna()
            if len(nums) >= 5:
                z = (nums.iloc[-1] - nums.mean()) / (nums.std(ddof=0) + EPS)
                return float(np.clip(z / 4.0, -1.0, 1.0))

        return 0.0

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _market_probability(
        self,
        spot: float,
        target: float,
        horizon_minutes: int,
        annual_vol: float,
        direction: str,
    ) -> float:
        t = max(horizon_minutes, 1) / (365.0 * 24.0 * 60.0)
        sigma = max(annual_vol, 0.08) * math.sqrt(t)
        if sigma <= 0:
            return 0.5
        log_ratio = math.log(target / spot)
        z = log_ratio / sigma
        p_above = 1.0 - self._normal_cdf(z)
        if direction == "above":
            return float(np.clip(p_above, 0.01, 0.99))
        return float(np.clip(1.0 - p_above, 0.01, 0.99))

    def _model_probability(
        self,
        spot: float,
        target: float,
        horizon_minutes: int,
        annual_vol: float,
        direction: str,
        factors: Dict[str, float],
        return_details: bool = False,
    ) -> Any:
        weights_main = {
            "orderbook_imbalance": 0.55,
            "trade_flow_imbalance": 0.60,
            "funding_rate": -0.30,
            "open_interest_change": 0.35,
            "liquidation_imbalance": 0.40,
            "perp_premium": 0.30,
            "futures_funding_rate": -0.35,
            "futures_open_interest_change_1m": 0.40,
            "futures_open_interest_change_5m": 0.30,
            "futures_liquidation_imbalance": 0.45,
            "futures_liquidation_pressure": -0.10,
            "futures_orderbook_imbalance": 0.50,
            "momentum_1h": 0.70,
            "short_term_return": 0.45,
            "return_1m": 0.60,
            "return_3m": 0.65,
            "return_5m": 0.55,
            "return_10m": 0.40,
            "momentum_acceleration": 0.45,
            "realized_vol_expansion": -0.20,
            "vol_5m": -0.08,
            "vol_15m": -0.10,
            "vwap_deviation": 0.35,
            "distance_to_high_15m": -0.20,
            "distance_to_low_15m": 0.20,
            "volume_spike": 0.25,
            "local_range_break": 0.35,
            "price_minus_ema_3": 0.55,
            "price_minus_ema_10": 0.40,
            "ema_3_minus_ema_10": 0.45,
            "distance_vol_normalized_signed": 0.90,
            "distance_vol_normalized_abs": -0.20,
            "time_adjusted_distance": -0.18,
            "spx_btc_corr": 0.20,
            "dxy_return": -0.25,
            "bond_yield_change": -0.15,
            "etf_flows_z": 0.35,
            "fear_greed": 0.20,
            "social_velocity": 0.30,
            "long_short_ratio_signal": 0.25,
            "fomc_event_risk": -0.15,
            "cpi_yoy": -0.10,
        }

        weights_near = {
            "distance_vol_normalized_signed": 1.50,
            "distance_vol_normalized_abs": -0.15,
            "return_1m": 0.95,
            "return_3m": 0.85,
            "return_5m": 0.65,
            "momentum_acceleration": 0.80,
            "vol_1m": -0.10,
            "vol_5m": -0.10,
            "vol_15m": -0.12,
            "minutes_remaining_scaled": -0.12,
            "time_adjusted_distance": -0.22,
            "price_minus_ema_3": 0.75,
            "price_minus_ema_10": 0.55,
            "ema_3_minus_ema_10": 0.65,
            "distance_to_high_15m": -0.30,
            "distance_to_low_15m": 0.30,
            "vwap_deviation": 0.30,
            "orderbook_imbalance": 0.45,
            "trade_flow_imbalance": 0.45,
            "liquidation_imbalance": 0.25,
            "futures_orderbook_imbalance": 0.40,
            "futures_liquidation_imbalance": 0.30,
        }

        def normalize_feature(name: str, value: float) -> float:
            if name == "volume_spike":
                value = value - 1.0
            if name in {"realized_vol_annual", "vol_1m", "vol_5m", "vol_15m"}:
                value = math.log(max(value, 1e-6))
            if name == "minutes_remaining_scaled":
                value = (value - 0.5) * 2.0
            return float(np.clip(value, -3.0, 3.0))

        def score_with(weights: Dict[str, float], intercept: float) -> float:
            score = intercept
            for name, weight in weights.items():
                score += weight * normalize_feature(name, float(factors.get(name, 0.0)))
            return float(score)

        strike_distance_pct = float(abs(spot - target) / (target + EPS))
        near_strike = strike_distance_pct < 0.0025
        score = score_with(weights_near, 0.0) if near_strike else score_with(weights_main, -0.05)

        regime = 1.0 + 0.3 * np.clip(float(factors.get("realized_vol_expansion", 0.0)), -1.0, 1.5)
        vol_adj = max(annual_vol * regime, 0.08)
        t = max(horizon_minutes, 1) / (365.0 * 24.0 * 60.0)
        sigma = vol_adj * math.sqrt(t)

        expected_move = max(spot * sigma, EPS)
        trend_mu = (
            0.55 * float(factors.get("return_1m", 0.0))
            + 0.45 * float(factors.get("return_3m", 0.0))
            + 0.30 * float(factors.get("return_5m", 0.0))
            + 0.20 * float(factors.get("momentum_1h", 0.0))
            + 0.15 * float(factors.get("price_minus_ema_3", 0.0))
            + 0.10 * float(factors.get("ema_3_minus_ema_10", 0.0))
        )
        expected_final_price = float(max(spot * (1.0 + trend_mu), EPS))
        expected_variance = float(max(expected_move**2, EPS))
        dist_z = (target - expected_final_price) / math.sqrt(expected_variance)
        p_above_dist = 1.0 - self._normal_cdf(dist_z)

        drift = 0.35 * sigma * score
        z = (math.log(target / spot) - drift) / (sigma + EPS)
        p_above_from_drift = 1.0 - self._normal_cdf(z)
        p_above_lr = self._sigmoid(score)
        if near_strike:
            p_above = 0.25 * p_above_from_drift + 0.25 * p_above_lr + 0.50 * p_above_dist
        else:
            p_above = 0.45 * p_above_from_drift + 0.25 * p_above_lr + 0.30 * p_above_dist
        p_above = float(np.clip(p_above, 0.01, 0.99))

        out_prob = p_above if direction == "above" else float(np.clip(1.0 - p_above, 0.01, 0.99))
        if not return_details:
            return out_prob
        return out_prob, {
            "near_strike_model_used": bool(near_strike),
            "model_score": float(score),
            "expected_final_price": float(expected_final_price),
            "expected_variance": float(expected_variance),
            "p_above_from_distribution": float(np.clip(p_above_dist, 0.01, 0.99)),
            "p_above_from_drift": float(np.clip(p_above_from_drift, 0.01, 0.99)),
            "p_above_from_score": float(np.clip(p_above_lr, 0.01, 0.99)),
            "expected_move": float(expected_move),
        }

    def _decision_table(
        self,
        model_yes: float,
        model_no: float,
        market_yes: float,
        market_no: float,
        bankroll: Optional[float],
        stake: Optional[float],
        kelly_fraction: float,
        warnings: List[str],
    ) -> Dict[str, Any]:
        model_yes = float(np.clip(model_yes, 1e-6, 1 - 1e-6))
        model_no = float(np.clip(model_no, 1e-6, 1 - 1e-6))
        market_yes = float(np.clip(market_yes, 1e-6, 1 - 1e-6))
        market_no = float(np.clip(market_no, 1e-6, 1 - 1e-6))

        market_sum = market_yes + market_no
        if abs(market_sum - 1.0) > 0.06:
            warnings.append(f"Market probability sum is off ({market_sum:.4f}); market may be unreliable.")
        if market_yes < 0.03 or market_yes > 0.97 or market_no < 0.03 or market_no > 0.97:
            warnings.append("Market appears potentially illiquid/extreme (probability near 0 or 1).")

        payout_yes = 1.0 / market_yes
        payout_no = 1.0 / market_no

        row_yes = self._ev_row(
            side="YES",
            model_prob=model_yes,
            market_prob=market_yes,
            payout=payout_yes,
            bankroll=bankroll,
            stake=stake,
            kelly_fraction=kelly_fraction,
        )
        row_no = self._ev_row(
            side="NO",
            model_prob=model_no,
            market_prob=market_no,
            payout=payout_no,
            bankroll=bankroll,
            stake=stake,
            kelly_fraction=kelly_fraction,
        )

        positive = [r for r in [row_yes, row_no] if r["ev"] > 0]
        reasoning_steps: List[str] = []
        reasoning_steps.append(
            f"Computed payout from market probs: YES={payout_yes:.4f}, NO={payout_no:.4f} (1/prob)."
        )
        reasoning_steps.append(
            f"Computed EV per $1: YES={row_yes['ev']:.4f}, NO={row_no['ev']:.4f}; selected only positive-EV candidates."
        )

        if not positive:
            reasoning_steps.append("No side has positive EV. Recommended action is NO BET.")
            return {
                "recommended_side": "NO BET",
                "recommended_action": "NO BET",
                "reasoning_steps": reasoning_steps,
                "rows": {"YES": row_yes, "NO": row_no},
            }

        # Small-bankroll rule: choose highest chance among positive-EV sides.
        positive_sorted = sorted(positive, key=lambda r: (r["model_prob"], r["ev"]), reverse=True)
        best = positive_sorted[0]
        reasoning_steps.append(
            f"Small-bankroll policy: among positive-EV sides, pick higher model probability -> {best['side']}."
        )
        reasoning_steps.append(
            f"Recommended stake uses fractional Kelly ({kelly_fraction:.2f}x) with bankroll cap."
        )
        return {
            "recommended_side": best["side"],
            "recommended_action": f"BET_{best['side']}",
            "reasoning_steps": reasoning_steps,
            "rows": {"YES": row_yes, "NO": row_no},
        }

    @staticmethod
    def _ev_row(
        side: str,
        model_prob: float,
        market_prob: float,
        payout: float,
        bankroll: Optional[float],
        stake: Optional[float],
        kelly_fraction: float,
    ) -> Dict[str, Any]:
        p = float(np.clip(model_prob, 1e-6, 1 - 1e-6))
        q = 1.0 - p

        # User-specified EV form: EV = p * payout - q * 1
        ev = float(p * payout - q)
        x_win = float(payout)
        x_lose = -1.0
        variance = float(p * (x_win - ev) ** 2 + q * (x_lose - ev) ** 2)
        max_loss_per_1 = 1.0

        b = max(payout - 1.0, 1e-9)
        kelly_full = float(np.clip(((b * p) - q) / b, 0.0, 1.0))
        kelly_used = float(np.clip(kelly_full * kelly_fraction, 0.0, 0.25))
        recommended_stake: Optional[float] = None
        recommended_stake_pct: Optional[float] = None
        if bankroll is not None:
            recommended_stake = round(bankroll * kelly_used, 2)
            recommended_stake_pct = float(kelly_used)
        elif stake is not None:
            recommended_stake = round(stake, 2)
            recommended_stake_pct = None

        return {
            "side": side,
            "model_prob": p,
            "market_prob": float(np.clip(market_prob, 1e-6, 1 - 1e-6)),
            "payout": float(payout),
            "ev": ev,
            "variance": variance,
            "max_loss_per_1": max_loss_per_1,
            "staking": {
                "kelly_fraction_full": kelly_full,
                "kelly_fraction_used": kelly_used,
                "recommended_stake": recommended_stake,
                "recommended_stake_pct_bankroll": recommended_stake_pct,
                "expected_value_per_1": ev,
                "implied_market_odds": float(payout - 1.0),
            },
        }

    @staticmethod
    def _staking_suggestion(
        model_prob: float,
        market_prob: float,
        bankroll: Optional[float],
        stake: Optional[float],
    ) -> Dict[str, Optional[float]]:
        mp = float(np.clip(market_prob, 0.01, 0.99))
        p = float(np.clip(model_prob, 0.01, 0.99))
        b = (1.0 / mp) - 1.0
        q = 1.0 - p

        kelly = ((b * p) - q) / b if b > 0 else 0.0
        kelly = float(np.clip(kelly, 0.0, 0.25))
        ev_per_1 = float(p * b - q)

        suggested_stake: Optional[float] = None
        if bankroll is not None:
            suggested_stake = round(bankroll * kelly, 2)
        elif stake is not None:
            suggested_stake = round(stake, 2)

        return {
            "kelly_fraction": kelly,
            "suggested_stake": suggested_stake,
            "expected_value_per_1": ev_per_1,
            "implied_market_odds": b,
        }

    def _plot_distribution(self, result: Dict[str, Any]) -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not installed; skipping chart")
            return None

        spot = float(result["spot_price"])
        target = float(result["target_price"])
        horizon = int(result["timeframe_minutes"])
        annual_vol = float(result["factors"].get("realized_vol_annual", 0.55))

        t = max(horizon, 1) / (365.0 * 24.0 * 60.0)
        sigma = max(annual_vol, 0.08) * math.sqrt(t)
        mu = math.log(spot)

        x = np.linspace(spot * 0.85, spot * 1.15, 350)
        lx = np.log(np.maximum(x, EPS))
        density = (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((lx - mu) / sigma) ** 2)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, density, lw=2)
        ax.axvline(target, color="tab:red", linestyle="--", label=f"Target {target:,.0f}")
        ax.set_title(f"BTC Probability Distribution ({horizon}m)")
        ax.set_xlabel("BTC Price")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        fig.tight_layout()

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = self.charts_dir / f"btc_distribution_{ts}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _print_summary(self, result: Dict[str, Any]) -> None:
        ea = result["edge_above"]
        eb = result["edge_below"]
        sa = "+" if ea >= 0 else ""
        sb = "+" if eb >= 0 else ""

        print("\nBTC Target Alert")
        print("----------------")
        print(f"Timestamp (UTC): {result['timestamp_utc']}")
        print(f"Spot Price: ${result['spot_price']:,.2f}")
        print(f"Target Price: ${result['target_price']:,.2f}")
        print(f"Timeframe: {result['timeframe_minutes']} minutes")
        if result.get("market_ticker"):
            print(f"Market Ticker: {result['market_ticker']}")
        print(f"Calibration: {result['calibration_method']} ({result['calibration_source']})")
        print(f"Market Prob Source: {result['market_probability_source']}")

        print("\nAbove Target")
        print(f"Raw Model Probability: {result['model_probability_above_raw'] * 100:.2f}%")
        print(f"Model Probability: {result['model_probability_above'] * 100:.2f}%")
        print(f"Market Probability: {result['market_probability_above'] * 100:.2f}%")
        print(f"Edge: {sa}{ea * 100:.2f}%")
        print(f"Suggested Action: {result['suggested_action_above']}")

        print("\nBelow Target")
        print(f"Raw Model Probability: {result['model_probability_below_raw'] * 100:.2f}%")
        print(f"Model Probability: {result['model_probability_below'] * 100:.2f}%")
        print(f"Market Probability: {result['market_probability_below'] * 100:.2f}%")
        print(f"Edge: {sb}{eb * 100:.2f}%")
        print(f"Suggested Action: {result['suggested_action_below']}")

        print("\nEV / Risk / Stake Table")
        header = (
            f"{'Side':<6} {'ModelProb':>10} {'MktProb':>9} {'Payout':>8} "
            f"{'EV($1)':>9} {'Variance':>10} {'MaxLoss':>8} {'Stake$':>10} {'Stake%':>8}"
        )
        print(header)
        print("-" * len(header))
        for side in ["YES", "NO"]:
            row = result["ev_table"][side]
            st = row["staking"]
            stake_dollar = "-" if st["recommended_stake"] is None else f"{st['recommended_stake']:.2f}"
            stake_pct = "-" if st["recommended_stake_pct_bankroll"] is None else f"{st['recommended_stake_pct_bankroll'] * 100:.2f}%"
            print(
                f"{side:<6} {row['model_prob'] * 100:>9.2f}% {row['market_prob'] * 100:>8.2f}% {row['payout']:>8.3f} "
                f"{row['ev']:>9.4f} {row['variance']:>10.4f} {row['max_loss_per_1']:>8.2f} {stake_dollar:>10} {stake_pct:>8}"
            )

        print("\nRecommended Decision")
        print(f"Action: {result['recommended_action']}")
        print(f"Side: {result['recommended_side']}")

        print("\nValidation Metrics")
        print(f"Model YES Prob: {result['model_prob_yes']:.4f}")
        print(f"Market YES Prob: {result['market_prob_yes']:.4f}")
        print(f"Edge: {result['edge']:+.4f}")
        print(f"Strike Distance: {result['strike_distance_pct'] * 100:.2f}%")
        print(f"Vol-Normalized Distance: {result['distance_vol_normalized']:.4f}")
        print(f"Moneyness: {result['moneyness_category']}")
        print(f"Confidence: {result['confidence_tier']} ({result['confidence_score']:.3f})")
        print(f"Expected Value: {result['expected_value_per_1']:+.4f}")
        print(f"Recommended Action (Validation): {result['bet_signal']}")
        print(f"Near-Strike Model Used: {result['near_strike_model_used']}")
        print(f"Expected Final Price: ${result['expected_final_price']:,.2f}")
        print(f"Expected Variance: {result['expected_variance']:.4f}")
        if result["bet_signal"] != "NO_BET":
            print(f"Kelly Fraction: {result['kelly_fraction_signal']:.4f}")
            if result.get("suggested_stake_signal") is not None:
                print(f"Suggested Stake: ${float(result['suggested_stake_signal']):.2f}")

        print("\nMarket Factors Used")
        mkt = result["market_factor_summary"]
        print(f"- Source: {mkt['source']}")
        print(f"- Market Yes/No: {mkt['market_yes'] * 100:.2f}% / {mkt['market_no'] * 100:.2f}%")
        print(f"- Payout Yes/No: {mkt['payout_yes']:.4f} / {mkt['payout_no']:.4f}")
        print(f"- Market Sum: {mkt['market_sum']:.4f}")
        print(f"- Calibration Method: {mkt['calibration_method']}")

        if result["warnings"]:
            print("\nWarnings")
            for w in result["warnings"]:
                print(f"- {w}")

    def _print_futures_summary(self, result: Dict[str, Any]) -> None:
        print("\nBTC Futures Signal")
        print("------------------")
        print(f"Timestamp (UTC): {result['timestamp_utc']}")
        print(f"Spot Price: ${result['spot_price']:,.2f}")
        print(f"Timeframe: {result['timeframe_minutes']} minutes")
        print(f"Expected Return: {result['expected_return'] * 100:.4f}%")
        print(f"Expected Price: ${result['expected_price']:,.2f}")
        print(f"Prob Up/Down: {result['prob_up'] * 100:.2f}% / {result['prob_down'] * 100:.2f}%")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Signal Strength: {result['signal_strength']:.3f}")
        print(f"Signal: {result['signal']}")
        print(f"Leverage: {result['leverage']:.1f}x")
        if result.get("take_profit") is not None:
            print(f"Entry: ${result['entry_price']:,.2f}")
            print(f"Take Profit: ${result['take_profit']:,.2f}")
            print(f"Stop Loss: ${result['stop_loss']:,.2f}")
        if result.get("contract_check_enabled"):
            print("\nContract Check")
            print(f"- Maintenance Margin Rate: {result['maintenance_margin_rate'] * 100:.3f}%")
            print(f"- Roundtrip Fees Assumed: {result['fee_roundtrip_bps']:.2f} bps")
            if result.get("estimated_liquidation_price") is not None:
                print(f"- Est. Liquidation Price: ${result['estimated_liquidation_price']:,.2f}")
            if result.get("distance_to_liquidation_pct") is not None:
                print(f"- Distance to Liquidation: {result['distance_to_liquidation_pct'] * 100:.3f}%")
            if result.get("take_profit_unlevered_pct") is not None:
                print(f"- TP Move (Unlevered): {result['take_profit_unlevered_pct'] * 100:.3f}%")
            if result.get("stop_loss_unlevered_pct") is not None:
                print(f"- SL Move (Unlevered): {result['stop_loss_unlevered_pct'] * 100:.3f}%")
            if result.get("take_profit_pnl_pct_levered_net") is not None:
                print(f"- TP PnL (Levered Net Fees): {result['take_profit_pnl_pct_levered_net'] * 100:.3f}%")
            if result.get("stop_loss_pnl_pct_levered_net") is not None:
                print(f"- SL PnL (Levered Net Fees): {result['stop_loss_pnl_pct_levered_net'] * 100:.3f}%")
            if result.get("stop_within_liquidation_buffer") is not None:
                print(f"- Stop Within Liq Buffer: {result['stop_within_liquidation_buffer']}")
        if result["warnings"]:
            print("\nWarnings")
            for w in result["warnings"]:
                print(f"- {w}")

    def _send_discord_alert(self, webhook: str, result: Dict[str, Any]) -> None:
        max_ev = max(result["ev_table"]["YES"]["ev"], result["ev_table"]["NO"]["ev"])
        color = 0x2ECC71 if (max_ev > 0 and result["recommended_side"] != "NO BET") else 0xE74C3C

        lines = [
            f"**BTC Target:** ${result['target_price']:,.2f}",
            f"**Spot:** ${result['spot_price']:,.2f}",
            f"**Timeframe:** {result['timeframe_minutes']} minutes",
            f"**Market Ticker:** {result.get('market_ticker', 'n/a')}",
            f"**Calibration:** {result['calibration_method']}",
            f"**Market Source:** {result['market_probability_source']}",
            f"**Above Raw/Calibrated/Market:** "
            f"{result['model_probability_above_raw'] * 100:.2f}% / "
            f"{result['model_probability_above'] * 100:.2f}% / "
            f"{result['market_probability_above'] * 100:.2f}%",
            f"**Above Edge:** {result['edge_above'] * 100:+.2f}%",
            f"**Above Action:** {result['suggested_action_above']}",
            f"**Below Raw/Calibrated/Market:** "
            f"{result['model_probability_below_raw'] * 100:.2f}% / "
            f"{result['model_probability_below'] * 100:.2f}% / "
            f"{result['market_probability_below'] * 100:.2f}%",
            f"**Below Edge:** {result['edge_below'] * 100:+.2f}%",
            f"**Below Action:** {result['suggested_action_below']}",
            f"**EV YES/NO per $1:** {result['ev_table']['YES']['ev']:.4f} / {result['ev_table']['NO']['ev']:.4f}",
            f"**Variance YES/NO:** {result['ev_table']['YES']['variance']:.4f} / {result['ev_table']['NO']['variance']:.4f}",
            f"**Recommended:** {result['recommended_action']} ({result['recommended_side']})",
            f"**Model YES Prob:** {result['model_prob_yes']:.4f}",
            f"**Market YES Prob:** {result['market_prob_yes']:.4f}",
            f"**Edge:** {result['edge']:+.4f}",
            f"**Strike Distance:** {result['strike_distance_pct'] * 100:.2f}%",
            f"**Vol-Normalized Distance:** {result['distance_vol_normalized']:.4f}",
            f"**Moneyness:** {result['moneyness_category']}",
            f"**Confidence:** {result['confidence_tier']} ({result['confidence_score']:.3f})",
            f"**Expected Value:** {result['expected_value_per_1']:+.4f}",
            f"**Validation Action:** {result['bet_signal']}",
            f"**Near-Strike Model Used:** {result['near_strike_model_used']}",
            f"**Timestamp (UTC):** {result['timestamp_utc']}",
        ]
        yes_stake = result["ev_table"]["YES"]["staking"].get("recommended_stake")
        no_stake = result["ev_table"]["NO"]["staking"].get("recommended_stake")
        if yes_stake is not None:
            lines.append(f"**YES Suggested Stake:** ${yes_stake:,.2f}")
        if no_stake is not None:
            lines.append(f"**NO Suggested Stake:** ${no_stake:,.2f}")
        for w in result.get("warnings", []):
            lines.append(w)

        payload = {
            "embeds": [
                {
                    "title": "BTC Model Alert",
                    "description": "\n".join(lines),
                    "color": color,
                }
            ]
        }

        try:
            response = requests.post(webhook, json=payload, timeout=10)
            response.raise_for_status()
            print("Discord alert sent successfully.")
        except Exception as exc:
            print(f"Discord alert failed: {exc}")

    def _send_discord_alert_futures(self, webhook: str, result: Dict[str, Any]) -> None:
        direction = result.get("signal", "NO_TRADE")
        color = 0x2ECC71 if direction == "LONG" else 0xE74C3C if direction == "SHORT" else 0x95A5A6

        lines = [
            f"**Spot:** ${result['spot_price']:,.2f}",
            f"**Timeframe:** {result['timeframe_minutes']} minutes",
            f"**Expected Return:** {result['expected_return'] * 100:.4f}%",
            f"**Expected Price:** ${result['expected_price']:,.2f}",
            f"**Prob Up/Down:** {result['prob_up'] * 100:.2f}% / {result['prob_down'] * 100:.2f}%",
            f"**Confidence:** {result['confidence_score']:.3f}",
            f"**Signal Strength:** {result['signal_strength']:.3f}",
            f"**Signal:** {direction}",
            f"**Leverage:** {result['leverage']:.1f}x",
        ]

        if result.get("take_profit") is not None:
            lines.extend(
                [
                    f"**Entry:** ${result['entry_price']:,.2f}",
                    f"**Take Profit:** ${result['take_profit']:,.2f}",
                    f"**Stop Loss:** ${result['stop_loss']:,.2f}",
                ]
            )
        if result.get("contract_check_enabled"):
            lines.extend(
                [
                    f"**Maint Margin Rate:** {result['maintenance_margin_rate'] * 100:.3f}%",
                    f"**Roundtrip Fees:** {result['fee_roundtrip_bps']:.2f} bps",
                ]
            )
            if result.get("estimated_liquidation_price") is not None:
                lines.append(f"**Est. Liquidation:** ${result['estimated_liquidation_price']:,.2f}")
            if result.get("distance_to_liquidation_pct") is not None:
                lines.append(f"**Distance to Liq:** {result['distance_to_liquidation_pct'] * 100:.3f}%")
            if result.get("stop_within_liquidation_buffer") is not None:
                lines.append(f"**Stop Within Liq Buffer:** {result['stop_within_liquidation_buffer']}")
        for w in result.get("warnings", []):
            lines.append(w)

        payload = {
            "embeds": [
                {
                    "title": "BTC Futures Signal",
                    "description": "\n".join(lines),
                    "color": color,
                }
            ]
        }

        try:
            response = requests.post(webhook, json=payload, timeout=10)
            response.raise_for_status()
            print("Discord alert sent successfully.")
        except Exception as exc:
            print(f"Discord alert failed: {exc}")

    def _append_query_log(self, result: Dict[str, Any]) -> None:
        contract_id = self._contract_id(
            target_price=float(result["target_price"]),
            resolution_due_utc=str(result["resolution_due_utc"]),
        )
        row = {
            "contract_id": contract_id,
            "timestamp_utc": result["timestamp_utc"],
            "target_price": result["target_price"],
            "spot_price": result["spot_price"],
            "timeframe_minutes": result["timeframe_minutes"],
            "market_ticker": result.get("market_ticker"),
            "model_probability_above": result["model_probability_above"],
            "model_probability_below": result["model_probability_below"],
            "model_probability_above_raw": result["model_probability_above_raw"],
            "model_probability_below_raw": result["model_probability_below_raw"],
            "market_probability_above": result["market_probability_above"],
            "market_probability_below": result["market_probability_below"],
            "market_probability_source": result["market_probability_source"],
            "edge_above": result["edge_above"],
            "edge_below": result["edge_below"],
            "suggested_action_above": result["suggested_action_above"],
            "suggested_action_below": result["suggested_action_below"],
            "recommended_side": result["recommended_side"],
            "recommended_action": result["recommended_action"],
            "model_prob_yes": result["model_prob_yes"],
            "market_prob_yes": result["market_prob_yes"],
            "edge": result["edge"],
            "strike_distance_pct": result["strike_distance_pct"],
            "distance_vol_normalized": result["distance_vol_normalized"],
            "moneyness_category": result["moneyness_category"],
            "confidence_tier": result["confidence_tier"],
            "confidence_score": result["confidence_score"],
            "bet_signal": result["bet_signal"],
            "expected_value_per_1": result["expected_value_per_1"],
            "kelly_fraction_signal": result["kelly_fraction_signal"],
            "suggested_stake_signal": result["suggested_stake_signal"],
            "expected_final_price": result["expected_final_price"],
            "expected_variance": result["expected_variance"],
            "near_strike_model_used": result["near_strike_model_used"],
            "ev_yes": result["ev_table"]["YES"]["ev"],
            "ev_no": result["ev_table"]["NO"]["ev"],
            "variance_yes": result["ev_table"]["YES"]["variance"],
            "variance_no": result["ev_table"]["NO"]["variance"],
            "payout_yes": result["ev_table"]["YES"]["payout"],
            "payout_no": result["ev_table"]["NO"]["payout"],
            "kelly_used_yes": result["ev_table"]["YES"]["staking"]["kelly_fraction_used"],
            "kelly_used_no": result["ev_table"]["NO"]["staking"]["kelly_fraction_used"],
            "stake_yes": result["ev_table"]["YES"]["staking"]["recommended_stake"],
            "stake_no": result["ev_table"]["NO"]["staking"]["recommended_stake"],
            "calibration_method": result["calibration_method"],
            "calibration_source": result["calibration_source"],
            "resolution_due_utc": result["resolution_due_utc"],
            "resolved": result["resolved"],
            "resolved_price": result.get("resolved_price"),
            "actual_hit_above": result["actual_hit_above"],
            "actual_hit_below": result["actual_hit_below"],
        }

        self._ensure_log_schema(list(row.keys()))
        if not self.query_log_path.exists():
            with self.query_log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        df = self._safe_read_query_log()
        if df.empty:
            df = pd.DataFrame(columns=list(row.keys()))
        for c in row.keys():
            if c not in df.columns:
                df[c] = None
        df = df[list(row.keys())]
        if "contract_id" in df.columns:
            missing_id = df["contract_id"].isna() | (df["contract_id"].astype(str).str.strip() == "")
            if bool(missing_id.any()):
                df.loc[missing_id, "contract_id"] = df.loc[missing_id].apply(
                    lambda r: self._contract_id(
                        target_price=float(pd.to_numeric(r.get("target_price"), errors="coerce")),
                        resolution_due_utc=str(r.get("resolution_due_utc")),
                    ),
                    axis=1,
                )

        if "contract_id" in df.columns:
            match = df["contract_id"].astype(str) == str(contract_id)
            if bool(match.any()):
                resolved_mask = df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])
                unresolved_match = match & ~resolved_mask
                idx = int(df[unresolved_match].index[-1]) if bool(unresolved_match.any()) else int(df[match].index[-1])
                # Keep exactly one prediction row per contract_id by updating in-place.
                update_fields = [
                    "timestamp_utc",
                    "model_probability_above",
                    "model_probability_below",
                    "market_probability_above",
                    "market_probability_below",
                    "edge_above",
                    "edge_below",
                    "ev_yes",
                    "ev_no",
                    "variance_yes",
                    "variance_no",
                    "spot_price",
                    "market_ticker",
                    "suggested_action_above",
                    "suggested_action_below",
                    "recommended_side",
                    "recommended_action",
                    "model_prob_yes",
                    "market_prob_yes",
                    "edge",
                    "strike_distance_pct",
                    "distance_vol_normalized",
                    "moneyness_category",
                    "confidence_tier",
                    "confidence_score",
                    "bet_signal",
                    "expected_value_per_1",
                    "kelly_fraction_signal",
                    "suggested_stake_signal",
                    "expected_final_price",
                    "expected_variance",
                    "near_strike_model_used",
                    "stake_yes",
                    "stake_no",
                    "market_probability_source",
                ]
                for col in update_fields:
                    if col in row:
                        df.loc[idx, col] = row[col]
                # Keep resolution/outcome fields untouched on updates.
                df = self._dedupe_contract_rows(df)
                df.to_csv(self.query_log_path, index=False)
                return

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = self._dedupe_contract_rows(df)
        df.to_csv(self.query_log_path, index=False)

    def _append_futures_log(self, result: Dict[str, Any]) -> None:
        row = {
            "timestamp_utc": result["timestamp_utc"],
            "timeframe_minutes": result["timeframe_minutes"],
            "spot_price": result["spot_price"],
            "expected_return": result["expected_return"],
            "expected_price": result["expected_price"],
            "prob_up": result["prob_up"],
            "prob_down": result["prob_down"],
            "confidence_score": result["confidence_score"],
            "signal_strength": result["signal_strength"],
            "signal": result["signal"],
            "entry_price": result["entry_price"],
            "take_profit": result.get("take_profit"),
            "stop_loss": result.get("stop_loss"),
            "leverage": result["leverage"],
            "expected_return_threshold": result["expected_return_threshold"],
            "min_confidence": result["min_confidence"],
            "min_signal_strength": result["min_signal_strength"],
            "take_profit_mult": result["take_profit_mult"],
            "stop_loss_mult": result["stop_loss_mult"],
            "contract_check_enabled": result.get("contract_check_enabled"),
            "maintenance_margin_rate": result.get("maintenance_margin_rate"),
            "fee_roundtrip_bps": result.get("fee_roundtrip_bps"),
            "estimated_liquidation_price": result.get("estimated_liquidation_price"),
            "distance_to_liquidation_pct": result.get("distance_to_liquidation_pct"),
            "take_profit_unlevered_pct": result.get("take_profit_unlevered_pct"),
            "stop_loss_unlevered_pct": result.get("stop_loss_unlevered_pct"),
            "take_profit_pnl_pct_levered_net": result.get("take_profit_pnl_pct_levered_net"),
            "stop_loss_pnl_pct_levered_net": result.get("stop_loss_pnl_pct_levered_net"),
            "stop_within_liquidation_buffer": result.get("stop_within_liquidation_buffer"),
            "realized_volatility": result["realized_volatility"],
            "expected_move": result["expected_move"],
            "resolution_due_utc": result["resolution_due_utc"],
        }

        header = list(row.keys())
        if not self.futures_log_path.exists():
            with self.futures_log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerow(row)
            return

        try:
            with self.futures_log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(row)
        except Exception as exc:
            print(f"Warning: failed writing futures log: {exc}")

    def _safe_read_futures_log(self) -> pd.DataFrame:
        if not self.futures_log_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.futures_log_path)
        except Exception:
            try:
                df = pd.read_csv(self.futures_log_path, on_bad_lines="skip", engine="python")
                print("Warning: futures_query_log had malformed rows; skipped bad lines while reading.")
                return df
            except Exception as exc:
                print(f"Warning: failed reading futures_query_log: {exc}")
                return pd.DataFrame()

    def _resolve_past_futures_queries(self) -> None:
        if not self.futures_log_path.exists():
            return

        df = self._safe_read_futures_log()
        if df.empty:
            return

        needed = [
            "resolved",
            "resolved_price",
            "settled_at_utc",
            "realized_return_unlevered",
            "realized_pnl_pct_levered_net",
            "realized_direction_up",
            "trade_won",
        ]
        for c in needed:
            if c not in df.columns:
                df[c] = None

        changed = False
        now = datetime.now(timezone.utc)
        for idx, row in df.iterrows():
            resolved = str(row.get("resolved", "")).strip().lower() in {"1", "true", "yes"}
            due_raw = row.get("resolution_due_utc")
            if pd.isna(due_raw):
                continue
            try:
                due = datetime.fromisoformat(str(due_raw))
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if due > now:
                continue

            resolved_price_exists = pd.notna(pd.to_numeric(pd.Series([row.get("resolved_price")]), errors="coerce").iloc[0])
            if resolved and resolved_price_exists:
                continue

            signal = str(row.get("signal", "NO_TRADE")).strip().upper()
            entry = float(pd.to_numeric(pd.Series([row.get("entry_price")]), errors="coerce").fillna(row.get("spot_price")).iloc[0])
            lev = float(pd.to_numeric(pd.Series([row.get("leverage")]), errors="coerce").fillna(1.0).iloc[0])
            fee_bps = float(pd.to_numeric(pd.Series([row.get("fee_roundtrip_bps")]), errors="coerce").fillna(0.0).iloc[0])

            if (not np.isfinite(entry)) or entry <= 0:
                continue

            try:
                realized_px = float(self._price_at_or_after(due))
            except Exception:
                continue

            is_up = 1 if realized_px >= entry else 0
            if signal == "LONG":
                unlev = float((realized_px - entry) / max(entry, EPS))
            elif signal == "SHORT":
                unlev = float((entry - realized_px) / max(entry, EPS))
            else:
                unlev = 0.0

            lev_net = float(unlev * max(lev, 0.0) - (fee_bps / 10000.0 if signal in {"LONG", "SHORT"} else 0.0))
            won = None
            if signal in {"LONG", "SHORT"}:
                won = int(lev_net > 0.0)

            df.loc[idx, "resolved"] = True
            df.loc[idx, "resolved_price"] = float(realized_px)
            df.loc[idx, "settled_at_utc"] = now.isoformat()
            df.loc[idx, "realized_return_unlevered"] = float(unlev)
            df.loc[idx, "realized_pnl_pct_levered_net"] = float(lev_net)
            df.loc[idx, "realized_direction_up"] = int(is_up)
            df.loc[idx, "trade_won"] = won
            changed = True

        if changed:
            df.to_csv(self.futures_log_path, index=False)

    def run_futures_backtest(
        self,
        min_confidence: float = 0.0,
        min_signal_strength: float = 0.0,
        include_no_trade: bool = False,
    ) -> Dict[str, Any]:
        self._resolve_past_futures_queries()
        df = self._safe_read_futures_log()
        if df.empty:
            out = {
                "status": "empty",
                "message": "No futures log rows found. Run --mode futures first.",
            }
            print(json.dumps(out, indent=2))
            return out

        for c in [
            "confidence_score",
            "signal_strength",
            "realized_pnl_pct_levered_net",
            "realized_direction_up",
            "prob_up",
            "resolved",
        ]:
            if c not in df.columns:
                df[c] = np.nan

        df["signal"] = df.get("signal", "").astype(str).str.upper()
        df["resolved_bool"] = df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
        df["signal_strength"] = pd.to_numeric(df["signal_strength"], errors="coerce")
        df["realized_pnl_pct_levered_net"] = pd.to_numeric(df["realized_pnl_pct_levered_net"], errors="coerce")
        df["realized_direction_up"] = pd.to_numeric(df["realized_direction_up"], errors="coerce")
        df["prob_up"] = pd.to_numeric(df["prob_up"], errors="coerce")

        mask = df["resolved_bool"]
        if not include_no_trade:
            mask &= df["signal"].isin(["LONG", "SHORT"])
        mask &= df["confidence_score"].fillna(0.0) >= float(min_confidence)
        mask &= df["signal_strength"].abs().fillna(0.0) >= float(min_signal_strength)

        scored = df[mask].copy()
        scored = scored.dropna(subset=["realized_pnl_pct_levered_net"])
        if scored.empty:
            out = {
                "status": "no_scored_rows",
                "message": "No resolved rows met filters yet.",
                "resolved_rows_total": int(df["resolved_bool"].sum()),
            }
            print(json.dumps(out, indent=2))
            return out

        pnl = scored["realized_pnl_pct_levered_net"].to_numpy(dtype=float)
        trades = scored[scored["signal"].isin(["LONG", "SHORT"])].copy()
        wins = int((trades["realized_pnl_pct_levered_net"] > 0).sum()) if not trades.empty else 0
        losses = int((trades["realized_pnl_pct_levered_net"] < 0).sum()) if not trades.empty else 0
        trade_count = int(len(trades))
        win_rate = float(wins / trade_count) if trade_count > 0 else None

        gross_profit = float(trades.loc[trades["realized_pnl_pct_levered_net"] > 0, "realized_pnl_pct_levered_net"].sum()) if not trades.empty else 0.0
        gross_loss = float(-trades.loc[trades["realized_pnl_pct_levered_net"] < 0, "realized_pnl_pct_levered_net"].sum()) if not trades.empty else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        # Directional accuracy conditioned on executed signal.
        directional_acc = None
        if not trades.empty and "realized_direction_up" in trades.columns:
            pred_up = (trades["signal"] == "LONG").astype(int)
            actual_up = pd.to_numeric(trades["realized_direction_up"], errors="coerce")
            m = actual_up.notna()
            if bool(m.any()):
                directional_acc = float(np.mean(pred_up[m].to_numpy(dtype=int) == actual_up[m].to_numpy(dtype=int)))

        summary = {
            "status": "ok",
            "rows_total": int(len(df)),
            "rows_resolved": int(df["resolved_bool"].sum()),
            "rows_scored": int(len(scored)),
            "trade_count": trade_count,
            "filters": {
                "min_confidence": float(min_confidence),
                "min_signal_strength": float(min_signal_strength),
                "include_no_trade": bool(include_no_trade),
            },
            "win_rate": win_rate,
            "directional_accuracy": directional_acc,
            "avg_pnl_pct_levered_net": float(np.mean(pnl)),
            "median_pnl_pct_levered_net": float(np.median(pnl)),
            "total_pnl_pct_levered_net": float(np.sum(pnl)),
            "gross_profit_pct": gross_profit,
            "gross_loss_pct": gross_loss,
            "profit_factor": profit_factor,
            "plus_ev_by_realized_mean": bool(float(np.mean(pnl)) > 0.0),
        }

        out_path = self.latest_dir / "futures_backtest_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nFutures Backtest Summary")
        print("------------------------")
        print(f"Rows Total / Resolved / Scored: {summary['rows_total']} / {summary['rows_resolved']} / {summary['rows_scored']}")
        print(f"Trade Count: {summary['trade_count']}")
        print(f"Win Rate: {('%.2f%%' % (100.0 * summary['win_rate'])) if summary['win_rate'] is not None else 'n/a'}")
        print(
            f"Directional Accuracy: "
            f"{('%.2f%%' % (100.0 * summary['directional_accuracy'])) if summary['directional_accuracy'] is not None else 'n/a'}"
        )
        print(f"Avg PnL (levered, net fees): {summary['avg_pnl_pct_levered_net'] * 100:.3f}%")
        print(f"Median PnL (levered, net fees): {summary['median_pnl_pct_levered_net'] * 100:.3f}%")
        print(f"Total PnL (levered, net fees): {summary['total_pnl_pct_levered_net'] * 100:.3f}%")
        print(f"Profit Factor: {summary['profit_factor'] if summary['profit_factor'] is not None else 'n/a'}")
        print(f"Plus EV (realized mean > 0): {summary['plus_ev_by_realized_mean']}")
        print(f"Saved report: {out_path}")

        return summary

    @staticmethod
    def _contract_id(target_price: float, resolution_due_utc: str) -> str:
        due = pd.to_datetime(resolution_due_utc, utc=True, errors="coerce")
        if pd.isna(due):
            due_part = str(resolution_due_utc)
        else:
            due_part = due.floor("min").strftime("%Y-%m-%dT%H:%M")
        if not np.isfinite(float(target_price)):
            t = "NA"
        elif abs(float(target_price) - round(float(target_price))) < 1e-9:
            t = str(int(round(float(target_price))))
        else:
            t = f"{float(target_price):.2f}".rstrip("0").rstrip(".")
        return f"BTC_{t}_{due_part}"

    def _backfill_contract_ids_in_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "contract_id" not in df.columns:
            return df
        df["contract_id"] = df["contract_id"].astype("object")
        missing_id = df["contract_id"].isna() | (df["contract_id"].astype(str).str.strip() == "")
        if not bool(missing_id.any()):
            return df
        df.loc[missing_id, "contract_id"] = df.loc[missing_id].apply(
            lambda r: self._contract_id(
                target_price=float(pd.to_numeric(r.get("target_price"), errors="coerce")),
                resolution_due_utc=str(r.get("resolution_due_utc")),
            ),
            axis=1,
        )
        return df

    @staticmethod
    def _dedupe_contract_rows(df: pd.DataFrame) -> pd.DataFrame:
        if "contract_id" not in df.columns or df.empty:
            return df
        work = df.copy()
        work["_ts"] = pd.to_datetime(work.get("timestamp_utc"), utc=True, errors="coerce")
        work["_resolved_rank"] = work.get("resolved", "").astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
        work = work.sort_values(["contract_id", "_resolved_rank", "_ts"]).drop_duplicates(subset=["contract_id"], keep="last")
        work = work.sort_values("_ts", na_position="last").drop(columns=["_ts", "_resolved_rank"])
        return work.reset_index(drop=True)

    def _ensure_log_schema(self, expected_header: List[str]) -> None:
        if not self.query_log_path.exists():
            return
        try:
            with self.query_log_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header == expected_header:
                df = self._safe_read_query_log()
                if not df.empty:
                    df2 = self._backfill_contract_ids_in_df(df.copy())
                    df2 = self._dedupe_contract_rows(df2)
                    if not df2.equals(df):
                        df2.to_csv(self.query_log_path, index=False)
                return
            df = self._safe_read_query_log()
            if df.empty:
                return
            for c in expected_header:
                if c not in df.columns:
                    df[c] = None
            df = df[expected_header]
            df = self._backfill_contract_ids_in_df(df)
            df = self._dedupe_contract_rows(df)
            df.to_csv(self.query_log_path, index=False)
            print("Detected query_log schema change. Migrated existing log to new schema in-place.")
        except Exception as exc:
            print(f"Warning: could not validate/rotate query log schema: {exc}")

    def _safe_read_query_log(self) -> pd.DataFrame:
        if not self.query_log_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.query_log_path)
        except Exception:
            try:
                df = pd.read_csv(self.query_log_path, on_bad_lines="skip", engine="python")
                print("Warning: query_log had malformed rows; skipped bad lines while reading.")
                return df
            except Exception as exc:
                print(f"Warning: failed reading query_log: {exc}")
                return pd.DataFrame()

    def _resolve_past_queries(self) -> None:
        if not self.query_log_path.exists():
            return

        df = self._safe_read_query_log()
        if df.empty or "resolved" not in df.columns:
            return
        df = self._backfill_contract_ids_in_df(df)

        changed = False
        now = datetime.now(timezone.utc)
        for idx, row in df.iterrows():
            resolved = str(row.get("resolved", "")).strip().lower() in {"1", "true", "yes"}

            due_raw = row.get("resolution_due_utc")
            if pd.isna(due_raw):
                continue

            try:
                due = datetime.fromisoformat(str(due_raw))
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            if due > now:
                continue

            resolved_price_exists = pd.notna(pd.to_numeric(pd.Series([row.get("resolved_price")]), errors="coerce").iloc[0])
            if resolved and resolved_price_exists:
                continue

            try:
                realized = self._price_at_or_after(due)
                target = float(row["target_price"])
                df.loc[idx, "resolved_price"] = float(realized)
                if (not resolved) or pd.isna(row.get("actual_hit_above")) or pd.isna(row.get("actual_hit_below")):
                    df.loc[idx, "actual_hit_above"] = 1 if realized >= target else 0
                    df.loc[idx, "actual_hit_below"] = 1 if realized <= target else 0
                    df.loc[idx, "resolved"] = True
                changed = True
            except Exception:
                continue

        if changed:
            df = self._dedupe_contract_rows(df)
            df.to_csv(self.query_log_path, index=False)

    def _price_at_or_after(self, ts_utc: datetime) -> float:
        product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
        headers = self._coinbase_headers() or None
        due_ts = pd.Timestamp(ts_utc, tz="UTC") if pd.Timestamp(ts_utc).tzinfo is None else pd.Timestamp(ts_utc).tz_convert("UTC")
        due_ts = due_ts.floor("min")
        # 1m candle open-time that closes exactly at due timestamp.
        target_candle_open = due_ts - pd.Timedelta(minutes=1)
        start = (target_candle_open - pd.Timedelta(minutes=5)).to_pydatetime()
        end = (due_ts + pd.Timedelta(minutes=5)).to_pydatetime()
        params = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "granularity": "60",
        }
        rows = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/candles", params=params, headers=headers)
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        if df.empty:
            raise ValueError("No resolution candles")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().sort_values("time")
        exact = df[df["time"] == target_candle_open]
        if not exact.empty:
            return float(exact.iloc[-1]["close"])
        # Fallback: nearest candle at or before intended close boundary.
        prior = df[df["time"] <= target_candle_open]
        if not prior.empty:
            return float(prior.iloc[-1]["close"])
        after = df[df["time"] > target_candle_open]
        if not after.empty:
            return float(after.iloc[0]["close"])
        raise ValueError("No usable 1m resolution candle")

    def _print_accuracy_summary(self) -> None:
        if not self.query_log_path.exists():
            return

        df = self._safe_read_query_log()
        if df.empty:
            return

        resolved = df[df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])].copy()
        if resolved.empty:
            print("Historical accuracy: no resolved predictions yet.")
            return

        for col in [
            "actual_hit_above",
            "actual_hit_below",
            "model_probability_above",
            "model_probability_below",
        ]:
            resolved[col] = pd.to_numeric(resolved[col], errors="coerce")

        scored = resolved.dropna(
            subset=["actual_hit_above", "actual_hit_below", "model_probability_above", "model_probability_below"]
        )
        if scored.empty:
            print("Historical accuracy: resolved rows exist but missing scoring fields.")
            return

        acc_above = float(
            np.mean((scored["model_probability_above"] >= 0.5).astype(int) == scored["actual_hit_above"].astype(int))
        )
        acc_below = float(
            np.mean((scored["model_probability_below"] >= 0.5).astype(int) == scored["actual_hit_below"].astype(int))
        )
        brier_above = float(np.mean((scored["model_probability_above"] - scored["actual_hit_above"]) ** 2))
        brier_below = float(np.mean((scored["model_probability_below"] - scored["actual_hit_below"]) ** 2))
        brier_mean = float((brier_above + brier_below) / 2.0)

        print(
            "Historical accuracy "
            f"({len(scored)} resolved) | "
            f"Above Acc: {acc_above * 100:.2f}% | "
            f"Below Acc: {acc_below * 100:.2f}% | "
            f"Brier Above: {brier_above:.4f} | "
            f"Brier Below: {brier_below:.4f} | "
            f"Brier Mean: {brier_mean:.4f}"
        )

    def _write_latest(self, result: Dict[str, Any]) -> None:
        out = self.latest_dir / "latest_alert.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


def run_btc_target_alert(
    target_price: float,
    timeframe_minutes: int,
    bankroll: Optional[float] = None,
    stake: Optional[float] = None,
    market_yes: Optional[float] = None,
    market_no: Optional[float] = None,
    edge_threshold: float = 0.03,
    kelly_fraction: float = 0.10,
    plot: bool = False,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    app = BTCProbabilityAlertApp(base_dir or Path(__file__).resolve().parent)
    return app.run(
        target_price=target_price,
        timeframe_minutes=timeframe_minutes,
        bankroll=bankroll,
        stake=stake,
        market_yes=market_yes,
        market_no=market_no,
        edge_threshold=edge_threshold,
        kelly_fraction=kelly_fraction,
        plot=plot,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC target probability + edge + Discord alert")
    parser.add_argument(
        "--mode",
        choices=["manual", "auto", "futures", "futures_backtest"],
        default="manual",
        help="manual = single run, auto = continuous strike scan, futures = directional signal, futures_backtest = score resolved futures logs",
    )
    parser.add_argument("--interactive", action="store_true", help="Prompt for numeric inputs")
    parser.add_argument("--target", required=False, type=float, help="Target BTC price in USD")
    parser.add_argument("--timeframe", required=False, type=int, help="Timeframe in integer minutes (e.g., 20, 25, 50)")
    parser.add_argument("--bankroll", default=None, type=float, help="Optional bankroll")
    parser.add_argument("--stake", default=None, type=float, help="Optional stake")
    parser.add_argument("--market-yes", default=None, type=float, help="Kalshi market YES probability (0-1 or 0-100)")
    parser.add_argument("--market-no", default=None, type=float, help="Kalshi market NO probability (0-1 or 0-100)")
    parser.add_argument("--edge-threshold", default=0.04, type=float, help="YES threshold (probability points)")
    parser.add_argument("--kelly-fraction", default=0.10, type=float, help="Fractional Kelly multiplier in (0,1]")
    parser.add_argument("--plot", action="store_true", help="Save probability distribution PNG")
    parser.add_argument("--poll-seconds", default=60, type=int, help="Auto mode poll interval seconds")
    parser.add_argument("--strike-step", default=100.0, type=float, help="Auto mode strike step around spot")
    parser.add_argument("--strikes-per-side", default=5, type=int, help="Auto mode number of strikes on each side of spot")
    parser.add_argument("--market-prob-min", default=0.03, type=float, help="Auto mode: reject extreme markets below this prob")
    parser.add_argument("--market-prob-max", default=0.97, type=float, help="Auto mode: reject extreme markets above this prob")
    parser.add_argument("--max-spread", default=0.12, type=float, help="Auto mode: reject markets with spread above this (fraction)")
    parser.add_argument("--min-expiry-minutes", default=2.0, type=float, help="Auto mode: minimum minutes to expiry")
    parser.add_argument("--max-expiry-minutes", default=180.0, type=float, help="Auto mode: maximum minutes to expiry")
    parser.add_argument("--max-cycles", default=None, type=int, help="Auto mode max cycles before stopping")
    parser.add_argument("--leverage", default=10.0, type=float, help="Futures mode leverage (x)")
    parser.add_argument("--return-threshold", default=0.0005, type=float, help="Futures mode expected return threshold")
    parser.add_argument("--min-confidence", default=0.52, type=float, help="Futures mode minimum confidence score")
    parser.add_argument("--min-signal-strength", default=0.10, type=float, help="Futures mode minimum signal strength")
    parser.add_argument("--prob-threshold", default=0.55, type=float, help="Futures mode probability threshold for long/short")
    parser.add_argument("--take-profit-mult", default=1.20, type=float, help="Futures mode take-profit multiplier")
    parser.add_argument("--stop-loss-mult", default=0.70, type=float, help="Futures mode stop-loss multiplier")
    parser.add_argument("--contract-check", action="store_true", help="Futures mode: include liquidation-buffer and fee-aware contract risk checks")
    parser.add_argument(
        "--maintenance-margin-rate",
        default=0.005,
        type=float,
        help="Futures mode contract check: maintenance margin rate as fraction (default 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--taker-fee-bps",
        default=5.0,
        type=float,
        help="Futures mode contract check: taker fee per side in bps (roundtrip is 2x)",
    )
    parser.add_argument("--backtest-min-confidence", default=0.0, type=float, help="Futures backtest: minimum confidence score filter")
    parser.add_argument(
        "--backtest-min-signal-strength",
        default=0.0,
        type=float,
        help="Futures backtest: minimum absolute signal strength filter",
    )
    parser.add_argument("--backtest-include-no-trade", action="store_true", help="Futures backtest: include NO_TRADE rows in scored set")
    return parser


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    if load_dotenv is not None:
        # Prefer BTCNEW/.env, then repo root .env
        load_dotenv(base_dir.parent / ".env")
        load_dotenv(base_dir.parent.parent / ".env")

    args = build_parser().parse_args()
    app = BTCProbabilityAlertApp(base_dir)

    target = args.target
    timeframe = args.timeframe
    bankroll = args.bankroll
    stake = args.stake
    market_yes = args.market_yes
    market_no = args.market_no

    if args.interactive and args.mode == "manual":
        target = float(input("Target BTC price (USD): ").strip())
        timeframe = int(input("Timeframe (minutes): ").strip())
        bankroll_raw = input("Bankroll (optional, press Enter to skip): ").strip()
        stake_raw = input("Stake (optional, press Enter to skip): ").strip()
        market_yes_raw = input("Kalshi market YES probability (optional, 0-1 or 0-100): ").strip()
        bankroll = float(bankroll_raw) if bankroll_raw else None
        stake = float(stake_raw) if stake_raw else None
        market_yes = float(market_yes_raw) if market_yes_raw else None
        if market_yes is None:
            market_no = None
        elif market_yes > 1.0:
            market_no = 100.0 - market_yes
        else:
            market_no = 1.0 - market_yes

    if args.mode == "auto":
        if timeframe is None:
            raise SystemExit("Error: auto mode requires --timeframe")
        app.run_auto(
            timeframe_minutes=timeframe,
            poll_seconds=args.poll_seconds,
            strike_step=args.strike_step,
            strikes_per_side=args.strikes_per_side,
            market_prob_min=args.market_prob_min,
            market_prob_max=args.market_prob_max,
            max_spread=args.max_spread,
            min_expiry_minutes=args.min_expiry_minutes,
            max_expiry_minutes=args.max_expiry_minutes,
            bankroll=bankroll,
            stake=stake,
            edge_threshold=args.edge_threshold,
            kelly_fraction=args.kelly_fraction,
            plot=args.plot,
            max_cycles=args.max_cycles,
        )
        return

    if args.mode == "futures":
        if timeframe is None:
            raise SystemExit("Error: futures mode requires --timeframe")
        app.run_futures(
            timeframe_minutes=timeframe,
            leverage=args.leverage,
            expected_return_threshold=args.return_threshold,
            min_confidence=args.min_confidence,
            min_signal_strength=args.min_signal_strength,
            prob_threshold=args.prob_threshold,
            take_profit_mult=args.take_profit_mult,
            stop_loss_mult=args.stop_loss_mult,
            contract_check=args.contract_check,
            maintenance_margin_rate=args.maintenance_margin_rate,
            taker_fee_bps=args.taker_fee_bps,
            plot=args.plot,
        )
        return

    if args.mode == "futures_backtest":
        app.run_futures_backtest(
            min_confidence=float(args.backtest_min_confidence),
            min_signal_strength=float(args.backtest_min_signal_strength),
            include_no_trade=bool(args.backtest_include_no_trade),
        )
        return

    if target is None or timeframe is None:
        raise SystemExit("Error: provide --target and --timeframe, or use --interactive")

    app.run(
        target_price=target,
        timeframe_minutes=timeframe,
        bankroll=bankroll,
        stake=stake,
        market_yes=market_yes,
        market_no=market_no,
        edge_threshold=args.edge_threshold,
        kelly_fraction=args.kelly_fraction,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
