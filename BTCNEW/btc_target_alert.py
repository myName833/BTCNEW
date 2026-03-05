#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
        annual_vol = max(float(factors.get("realized_vol_annual", 0.55)), 0.08)

        market_above_model = self._market_probability(spot, target_price, timeframe_minutes, annual_vol, "above")
        market_below_model = self._market_probability(spot, target_price, timeframe_minutes, annual_vol, "below")
        market_above, market_below, market_prob_source = self._resolve_market_probabilities(
            market_yes=market_yes,
            market_no=market_no,
            fallback_yes=market_above_model,
            fallback_no=market_below_model,
            warnings=warnings,
        )
        model_above_raw = self._model_probability(spot, target_price, timeframe_minutes, annual_vol, "above", factors)
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
        staking_above = decision["rows"]["YES"]["staking"]
        staking_below = decision["rows"]["NO"]["staking"]

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
            "actual_hit_above": None,
            "actual_hit_below": None,
        }
        return result

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
            warnings.append("joblib unavailable; calibration disabled (passthrough)")
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
            fr_df = pd.DataFrame(fr)
            if "funding_rate" in fr_df.columns:
                ser = pd.to_numeric(fr_df["funding_rate"], errors="coerce").dropna()
                if not ser.empty:
                    out["funding_rate"] = float(ser.iloc[-1])
        except Exception as exc:
            warnings.append(f"Coinalyze funding unavailable: {exc}")

        try:
            oi = self._safe_get_json(
                f"{COINALYZE_BASE}/open-interest-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            oi_df = pd.DataFrame(oi)
            if "open_interest" in oi_df.columns:
                ser = pd.to_numeric(oi_df["open_interest"], errors="coerce").dropna()
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
            liq_df = pd.DataFrame(liq)
            longs = pd.to_numeric(liq_df.get("long_liquidation_usd", 0.0), errors="coerce").fillna(0.0)
            shorts = pd.to_numeric(liq_df.get("short_liquidation_usd", 0.0), errors="coerce").fillna(0.0)
            lsum = float(longs.sum())
            ssum = float(shorts.sum())
            out["liquidation_imbalance"] = (ssum - lsum) / (ssum + lsum + EPS)
        except Exception as exc:
            warnings.append(f"Coinalyze liquidations unavailable: {exc}")

        try:
            premium = self._safe_get_json(
                f"{COINALYZE_BASE}/premium-index-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            pm_df = pd.DataFrame(premium)
            if "value" in pm_df.columns:
                ser = pd.to_numeric(pm_df["value"], errors="coerce").dropna()
                if not ser.empty:
                    out["perp_premium"] = float(ser.iloc[-1])
        except Exception as exc:
            warnings.append(f"Coinalyze perp premium unavailable: {exc}")

        try:
            ls = self._safe_get_json(
                f"{COINALYZE_BASE}/long-short-ratio-history",
                params={"symbols": symbol, "from": start, "to": now, "interval": "1hour"},
                headers=headers,
            )
            ls_df = pd.DataFrame(ls)
            if "r" in ls_df.columns:
                ser = pd.to_numeric(ls_df["r"], errors="coerce").dropna()
                if not ser.empty:
                    out["long_short_ratio"] = float(ser.iloc[-1])
        except Exception as exc:
            warnings.append(f"Coinalyze long/short ratio unavailable: {exc}")

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
        interval_minutes = self._select_candle_interval_minutes(horizon_minutes)

        try:
            if interval_minutes == 1:
                df_1m = self._fetch_coinbase_candles(product, 60, start, end, headers)
                if df_1m.empty:
                    warnings.append("No 1m candle data returned; using neutral price-action factors")
                    return pd.DataFrame(), interval_minutes
                return df_1m, interval_minutes

            if interval_minutes == 3:
                # Coinbase does not support 3m granularity directly; build from 1m candles.
                df_1m = self._fetch_coinbase_candles(product, 60, start, end, headers)
                if df_1m.empty:
                    warnings.append("No 1m candle data for 3m resample; using neutral price-action factors")
                    return pd.DataFrame(), interval_minutes
                rs = df_1m.resample("3min").agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                rs = rs.dropna()
                return rs, interval_minutes

            # 5m default
            df_5m = self._fetch_coinbase_candles(product, 300, start, end, headers)
            if df_5m.empty:
                warnings.append("No candle data returned; using neutral price-action factors")
                return pd.DataFrame(), interval_minutes
            return df_5m, interval_minutes
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
            "realized_vol_annual": 0.55,
            "realized_vol_expansion": 0.0,
            "vwap_deviation": 0.0,
            "volume_spike": 1.0,
            "local_range_break": 0.0,
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

        scale = math.sqrt((365.0 * 24.0 * 60.0) / float(candle_minutes))
        rv = float(rets.tail(min(len(rets), 288)).std(ddof=0) * scale)
        rv_short = float(rets.tail(min(len(rets), 24)).std(ddof=0) * scale)

        out["realized_vol_annual"] = max(rv, 0.08)
        out["realized_vol_expansion"] = (rv_short / (rv + EPS)) - 1.0

        bars_4h = max(2, 240 // candle_minutes)
        bars_2h = max(2, 120 // candle_minutes)
        vwap_num = (df["close"] * df["volume"]).tail(bars_4h).sum()
        vwap_den = df["volume"].tail(bars_4h).sum() + EPS
        vwap = vwap_num / vwap_den
        out["vwap_deviation"] = float((close.iloc[-1] - vwap) / (vwap + EPS))

        vol = df["volume"].tail(bars_4h)
        if len(vol) >= 10:
            out["volume_spike"] = float(vol.iloc[-1] / (vol.median() + EPS))

        recent_high = float(df["high"].tail(bars_2h).max())
        recent_low = float(df["low"].tail(bars_2h).min())
        last_close = float(close.iloc[-1])
        if last_close > recent_high:
            out["local_range_break"] = 1.0
        elif last_close < recent_low:
            out["local_range_break"] = -1.0

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
    ) -> float:
        weights = {
            "orderbook_imbalance": 0.55,
            "trade_flow_imbalance": 0.60,
            "funding_rate": -0.30,
            "open_interest_change": 0.35,
            "liquidation_imbalance": 0.40,
            "perp_premium": 0.30,
            "momentum_1h": 0.70,
            "short_term_return": 0.45,
            "realized_vol_expansion": -0.20,
            "vwap_deviation": 0.35,
            "volume_spike": 0.25,
            "local_range_break": 0.35,
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

        def normalize_feature(name: str, value: float) -> float:
            if name == "volume_spike":
                value = value - 1.0
            return float(np.clip(value, -3.0, 3.0))

        score = -0.05
        for name, weight in weights.items():
            score += weight * normalize_feature(name, float(factors.get(name, 0.0)))

        regime = 1.0 + 0.3 * np.clip(float(factors.get("realized_vol_expansion", 0.0)), -1.0, 1.5)
        vol_adj = max(annual_vol * regime, 0.08)
        t = max(horizon_minutes, 1) / (365.0 * 24.0 * 60.0)
        sigma = vol_adj * math.sqrt(t)

        drift = 0.40 * sigma * score
        z = (math.log(target / spot) - drift) / (sigma + EPS)
        p_above_from_drift = 1.0 - self._normal_cdf(z)
        p_above_lr = self._sigmoid(score)
        p_above = 0.7 * p_above_from_drift + 0.3 * p_above_lr
        p_above = float(np.clip(p_above, 0.01, 0.99))

        if direction == "above":
            return p_above
        return float(np.clip(1.0 - p_above, 0.01, 0.99))

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
        print("Reasoning:")
        for i, step in enumerate(result["decision_reasoning_steps"], start=1):
            print(f"{i}. {step}")

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

    def _send_discord_alert(self, webhook: str, result: Dict[str, Any]) -> None:
        max_ev = max(result["ev_table"]["YES"]["ev"], result["ev_table"]["NO"]["ev"])
        color = 0x2ECC71 if (max_ev > 0 and result["recommended_side"] != "NO BET") else 0xE74C3C

        lines = [
            f"**BTC Target:** ${result['target_price']:,.2f}",
            f"**Spot:** ${result['spot_price']:,.2f}",
            f"**Timeframe:** {result['timeframe_minutes']} minutes",
            f"**Calibration:** {result['calibration_method']}",
            f"**Market Source:** {result['market_probability_source']}",
            f"**Above Model/Market:** {result['model_probability_above'] * 100:.2f}% / {result['market_probability_above'] * 100:.2f}%",
            f"**Above Edge:** {result['edge_above'] * 100:+.2f}%",
            f"**Above Action:** {result['suggested_action_above']}",
            f"**Below Model/Market:** {result['model_probability_below'] * 100:.2f}% / {result['market_probability_below'] * 100:.2f}%",
            f"**Below Edge:** {result['edge_below'] * 100:+.2f}%",
            f"**Below Action:** {result['suggested_action_below']}",
            f"**EV YES/NO per $1:** {result['ev_table']['YES']['ev']:.4f} / {result['ev_table']['NO']['ev']:.4f}",
            f"**Variance YES/NO:** {result['ev_table']['YES']['variance']:.4f} / {result['ev_table']['NO']['variance']:.4f}",
            f"**Recommended:** {result['recommended_action']} ({result['recommended_side']})",
            f"**Timestamp (UTC):** {result['timestamp_utc']}",
        ]
        for i, step in enumerate(result.get("decision_reasoning_steps", []), start=1):
            lines.append(f"**Why {i}:** {step}")

        yes_stake = result["ev_table"]["YES"]["staking"].get("recommended_stake")
        no_stake = result["ev_table"]["NO"]["staking"].get("recommended_stake")
        if yes_stake is not None:
            lines.append(f"**YES Suggested Stake:** ${yes_stake:,.2f}")
        if no_stake is not None:
            lines.append(f"**NO Suggested Stake:** ${no_stake:,.2f}")

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

    def _append_query_log(self, result: Dict[str, Any]) -> None:
        row = {
            "timestamp_utc": result["timestamp_utc"],
            "target_price": result["target_price"],
            "spot_price": result["spot_price"],
            "timeframe_minutes": result["timeframe_minutes"],
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
            "actual_hit_above": result["actual_hit_above"],
            "actual_hit_below": result["actual_hit_below"],
        }

        self._ensure_log_schema(list(row.keys()))
        write_header = not self.query_log_path.exists()
        with self.query_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _ensure_log_schema(self, expected_header: List[str]) -> None:
        if not self.query_log_path.exists():
            return
        try:
            with self.query_log_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header == expected_header:
                return
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            legacy = self.logs_dir / f"query_log_legacy_{ts}.csv"
            shutil.move(str(self.query_log_path), str(legacy))
            print(f"Detected query_log schema change. Rotated old log to: {legacy}")
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

        changed = False
        now = datetime.now(timezone.utc)
        for idx, row in df.iterrows():
            resolved = str(row.get("resolved", "")).strip().lower() in {"1", "true", "yes"}
            if resolved:
                continue

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

            try:
                realized = self._price_at_or_after(due)
                target = float(row["target_price"])
                df.loc[idx, "actual_hit_above"] = 1 if realized >= target else 0
                df.loc[idx, "actual_hit_below"] = 1 if realized <= target else 0
                df.loc[idx, "resolved"] = True
                changed = True
            except Exception:
                continue

        if changed:
            df.to_csv(self.query_log_path, index=False)

    def _price_at_or_after(self, ts_utc: datetime) -> float:
        product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
        headers = self._coinbase_headers() or None
        start = ts_utc - timedelta(minutes=5)
        end = ts_utc + timedelta(minutes=30)
        params = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "granularity": "300",
        }
        rows = self._safe_get_json(f"{COINBASE_BASE}/products/{product}/candles", params=params, headers=headers)
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        if df.empty:
            raise ValueError("No resolution candles")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().sort_values("time")
        after = df[df["time"] >= pd.Timestamp(ts_utc)]
        if after.empty:
            return float(df.iloc[-1]["close"])
        return float(after.iloc[0]["close"])

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
    parser.add_argument("--interactive", action="store_true", help="Prompt for numeric inputs")
    parser.add_argument("--target", required=False, type=float, help="Target BTC price in USD")
    parser.add_argument("--timeframe", required=False, type=int, help="Timeframe in integer minutes (e.g., 20, 25, 50)")
    parser.add_argument("--bankroll", default=None, type=float, help="Optional bankroll")
    parser.add_argument("--stake", default=None, type=float, help="Optional stake")
    parser.add_argument("--market-yes", default=None, type=float, help="Kalshi market YES probability (0-1 or 0-100)")
    parser.add_argument("--market-no", default=None, type=float, help="Kalshi market NO probability (0-1 or 0-100)")
    parser.add_argument("--edge-threshold", default=0.03, type=float, help="YES threshold (probability points)")
    parser.add_argument("--kelly-fraction", default=0.10, type=float, help="Fractional Kelly multiplier in (0,1]")
    parser.add_argument("--plot", action="store_true", help="Save probability distribution PNG")
    return parser


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    if load_dotenv is not None:
        load_dotenv(base_dir / ".env")
        load_dotenv(base_dir.parent / "BTC" / ".env")

    args = build_parser().parse_args()

    target = args.target
    timeframe = args.timeframe
    bankroll = args.bankroll
    stake = args.stake
    market_yes = args.market_yes
    market_no = args.market_no

    if args.interactive:
        target = float(input("Target BTC price (USD): ").strip())
        timeframe = int(input("Timeframe (minutes): ").strip())
        bankroll_raw = input("Bankroll (optional, press Enter to skip): ").strip()
        stake_raw = input("Stake (optional, press Enter to skip): ").strip()
        market_yes_raw = input("Kalshi market YES probability (optional, 0-1 or 0-100): ").strip()
        market_no_raw = input("Kalshi market NO probability (optional, 0-1 or 0-100): ").strip()
        bankroll = float(bankroll_raw) if bankroll_raw else None
        stake = float(stake_raw) if stake_raw else None
        market_yes = float(market_yes_raw) if market_yes_raw else None
        market_no = float(market_no_raw) if market_no_raw else None

    if target is None or timeframe is None:
        raise SystemExit("Error: provide --target and --timeframe, or use --interactive")

    run_btc_target_alert(
        target_price=target,
        timeframe_minutes=timeframe,
        bankroll=bankroll,
        stake=stake,
        market_yes=market_yes,
        market_no=market_no,
        edge_threshold=args.edge_threshold,
        kelly_fraction=args.kelly_fraction,
        plot=args.plot,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    main()
