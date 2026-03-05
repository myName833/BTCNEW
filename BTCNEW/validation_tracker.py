#!/usr/bin/env python3
"""BTCNEW full probability validation tracker.

This mode is pure probability validation (no spread/depth/edge/exposure filters):
- Logs ALL BTC hourly strike predictions in the active expiry bucket.
- Updates official settlement outcomes once resolved.
- Computes threshold/bucket calibration diagnostics and optional flat-bet PnL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from btc_target_alert import BTCAlert

FULL_COLUMNS = [
    "timestamp_prediction",
    "market_id",
    "strike",
    "expiry_time",
    "expiration_window_utc",
    "model_yes_probability",
    "market_yes_probability",
    "market_no_probability",
    "time_to_expiry",
    "spot_price_at_prediction",
    "resolved_outcome",
    "resolved",
    "settled_at_utc",
    "flat_bet_yes",
    "flat_entry_price_yes",
    "flat_stake",
    "flat_pnl",
    "flat_cumulative_pnl",
]

DIR_COLUMNS = [
    "simulation_id",
    "timestamp_prediction",
    "horizon_minutes",
    "expiry_time",
    "strike",
    "spot_price_at_prediction",
    "model_yes_probability",
    "market_yes_probability",
    "predicted_yes",
    "predicted_direction",
    "resolved_outcome_yes",
    "actual_direction",
    "correct_direction",
    "simulated_stake",
    "simulated_entry_price",
    "simulated_pnl",
    "simulated_cumulative_pnl",
    "resolved",
    "settled_at_utc",
]


@dataclass
class Config:
    flat_bet_threshold: float = 0.65
    flat_stake: float = 15.0
    min_minutes_left: int = 10
    max_minutes_left: int = 90
    directional_yes_threshold: float = 0.65
    directional_fake_stake: float = 15.0
    directional_horizon_minutes: int = 60
    directional_spot_band_pct: float = 0.03
    strike_step: float = 500.0
    strikes_per_cycle: int = 21
    pure_model: bool = False


class KalshiClient:
    def __init__(self) -> None:
        self.base_candidates = self._base_candidates()
        self.headers = self._headers()
        self.last_fetch_error: Optional[str] = None

    @staticmethod
    def _base_candidates() -> List[str]:
        env_base = os.getenv("KALSHI_API_BASE", "").strip()
        cands = [
            env_base,
            "https://api.kalshi.com/trade-api/v2",
            "https://api.elections.kalshi.com/trade-api/v2",
            "https://trading-api.kalshi.com/trade-api/v2",
        ]
        out: List[str] = []
        seen = set()
        for c in cands:
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        return out

    @staticmethod
    def _headers() -> Dict[str, str]:
        key = os.getenv("KALSHI_API_KEY", "").strip()
        token = os.getenv("KALSHI_API_TOKEN", "").strip()
        h = {"accept": "application/json"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        if key:
            h["Authorization"] = f"Bearer {key}"
            h["KALSHI-ACCESS-KEY"] = key
        return h

    @staticmethod
    def _safe_get(url: str, headers: Dict[str, str], params: Optional[Dict[str, str]] = None) -> Any:
        timeout = float(os.getenv("KALSHI_HTTP_TIMEOUT_SECONDS", "8"))
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _norm_prob(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            x = float(v)
            if x > 100.0:
                x /= 10000.0
            elif x > 1.0:
                x /= 100.0
            return float(np.clip(x, 0.0, 1.0))
        except Exception:
            return None

    @staticmethod
    def _extract_market_fields(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ticker = row.get("ticker") or row.get("market_ticker") or row.get("marketTicker")
        if not ticker:
            return None

        yes_ask = KalshiClient._norm_prob(row.get("yes_ask") or row.get("yesAsk") or row.get("yes_ask_price") or row.get("yesAskPrice"))
        yes_bid = KalshiClient._norm_prob(row.get("yes_bid") or row.get("yesBid") or row.get("yes_bid_price") or row.get("yesBidPrice"))
        no_ask = KalshiClient._norm_prob(row.get("no_ask") or row.get("noAsk") or row.get("no_ask_price") or row.get("noAskPrice"))
        no_bid = KalshiClient._norm_prob(row.get("no_bid") or row.get("noBid") or row.get("no_bid_price") or row.get("noBidPrice"))
        yes_last = KalshiClient._norm_prob(row.get("yes_price") or row.get("yesPrice"))
        no_last = KalshiClient._norm_prob(row.get("no_price") or row.get("noPrice"))

        y = yes_ask
        n = no_ask
        if y is None and yes_bid is not None and yes_ask is not None:
            y = (yes_bid + yes_ask) / 2.0
        if n is None and no_bid is not None and no_ask is not None:
            n = (no_bid + no_ask) / 2.0
        if y is None and yes_last is not None:
            y = yes_last
        if n is None and no_last is not None:
            n = no_last
        if y is None and n is not None:
            y = 1.0 - n
        if n is None and y is not None:
            n = 1.0 - y
        if y is None or n is None:
            return None

        expiry = (
            row.get("close_time")
            or row.get("closeTime")
            or row.get("expiration_time")
            or row.get("expirationTime")
            or row.get("expiration_date")
            or row.get("expirationDate")
            or row.get("settlement_time")
            or row.get("settlementTime")
        )
        title = row.get("title") or row.get("subtitle") or ""
        strike = (
            row.get("strike")
            or row.get("strike_price")
            or row.get("strikePrice")
            or row.get("floor_strike")
            or row.get("floorStrike")
            or row.get("cap_strike")
            or row.get("capStrike")
            or row.get("subtitle")
        )
        if strike is None:
            mm = re.findall(r"(?<!\d)(\d{4,7})(?!\d)", f"{ticker} {title}")
            if mm:
                strike = float(mm[-1])
        try:
            strike = float(strike) if strike is not None else None
            if strike is not None and strike > 300000:
                strike /= 100.0
        except Exception:
            strike = None

        return {
            "market_id": str(ticker),
            "market_title": str(title),
            "strike": strike,
            "expiry_raw": expiry,
            "yes_ask": float(np.clip(y, 0.0, 1.0)),
            "no_ask": float(np.clip(n, 0.0, 1.0)),
            "raw": row,
        }

    @staticmethod
    def _series_candidates() -> List[str]:
        raw = os.getenv("KALSHI_BTC_SERIES", "KXBTCD,KXBTC,KXBT")
        out: List[str] = []
        seen = set()
        for s in re.split(r"[,\s;]+", raw):
            t = s.strip().upper()
            if t and t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def _fetch_markets_with_params(self, params: Dict[str, str], pages: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for base in self.base_candidates:
            url = f"{base}/markets"
            for h in (self.headers, {"accept": "application/json"}):
                cursor = None
                for _ in range(max(1, pages)):
                    q = dict(params)
                    if cursor:
                        q["cursor"] = str(cursor)
                    try:
                        payload = self._safe_get(url, headers=h, params=q)
                    except Exception as exc:
                        self.last_fetch_error = f"{url} params={q} err={exc}"
                        break
                    rows = []
                    if isinstance(payload, dict):
                        rows = payload.get("markets") or payload.get("data") or payload.get("results") or []
                        cursor = payload.get("cursor") or payload.get("next_cursor") or payload.get("nextCursor")
                    elif isinstance(payload, list):
                        rows = payload
                        cursor = None
                    if not isinstance(rows, list):
                        break
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        rr = r.get("market") if isinstance(r.get("market"), dict) else r
                        p = self._extract_market_fields(rr)
                        if p:
                            out.append(p)
                    if not cursor:
                        break
        uniq: Dict[str, Dict[str, Any]] = {}
        for m in out:
            mid = str(m.get("market_id", ""))
            if mid:
                uniq[mid] = m
        return list(uniq.values())

    def fetch_btc_markets(self) -> List[Dict[str, Any]]:
        queries: List[Dict[str, str]] = [
            {"status": "open", "limit": "1000", "search": "bitcoin"},
            {"status": "open", "limit": "1000", "search": "btc"},
            {"status": "active", "limit": "1000", "search": "bitcoin"},
            {"status": "active", "limit": "1000", "search": "btc"},
        ]
        for s in self._series_candidates():
            queries.extend(
                [
                    {"status": "open", "limit": "1000", "series_ticker": s},
                    {"status": "active", "limit": "1000", "series_ticker": s},
                ]
            )

        out: List[Dict[str, Any]] = []
        for q in queries:
            out.extend(self._fetch_markets_with_params(q, pages=max(1, int(os.getenv("KALSHI_TARGETED_MAX_PAGES", "2")))))

        if not out:
            out.extend(self._fetch_markets_with_params({"status": "open", "limit": "1000"}, pages=max(1, int(os.getenv("KALSHI_MAX_PAGES", "3")))))

        keys = ["BTC", "BITCOIN", "KXBT", "KXBTC", "KXBTCD", "XBT"]
        filt: Dict[str, Dict[str, Any]] = {}
        for m in out:
            raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
            txt = " ".join([
                str(m.get("market_id", "")).upper(),
                str(m.get("market_title", "")).upper(),
                str(raw.get("title", "")).upper(),
                str(raw.get("subtitle", "")).upper(),
                str(raw.get("series_ticker", raw.get("seriesTicker", ""))).upper(),
                str(raw.get("event_ticker", raw.get("eventTicker", ""))).upper(),
            ])
            if any(k in txt for k in keys):
                filt[str(m["market_id"]).upper()] = m
        return list(filt.values())

    def fetch_market(self, market_id: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for base in self.base_candidates:
            for h in (self.headers, {"accept": "application/json"}):
                try:
                    return self._safe_get(f"{base}/markets/{market_id}", headers=h)
                except Exception as exc:
                    last_err = exc
        raise RuntimeError(f"Kalshi fetch failed for {market_id}: {last_err}")


class ValidationTracker:
    def __init__(self, out_dir: Path, cfg: Config) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.full_csv = self.out_dir / "full_validation_predictions.csv"
        self.report_json = self.out_dir / "validation_full_report.json"
        self.directional_csv = self.out_dir / "directional_sim_predictions.csv"
        self.directional_report_json = self.out_dir / "directional_validation_report.json"
        self.cfg = cfg
        self.kalshi = KalshiClient()
        self.model = BTCAlert(Path(__file__).resolve().parent)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _load_df(path: Path) -> pd.DataFrame:
        if not path.exists():
            df = pd.DataFrame(columns=FULL_COLUMNS)
            for c in FULL_COLUMNS:
                df[c] = df[c].astype("object")
            return df
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        for c in FULL_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[FULL_COLUMNS]
        # Keep write-path columns object-typed to avoid pandas future dtype warnings on row appends/assignments.
        for c in FULL_COLUMNS:
            df[c] = df[c].astype("object")
        return df

    @staticmethod
    def _save_df(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df[FULL_COLUMNS].to_csv(path, index=False)

    @staticmethod
    def _load_dir_df(path: Path) -> pd.DataFrame:
        if not path.exists():
            df = pd.DataFrame(columns=DIR_COLUMNS)
            for c in DIR_COLUMNS:
                df[c] = df[c].astype("object")
            return df
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        for c in DIR_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[DIR_COLUMNS]
        for c in DIR_COLUMNS:
            df[c] = df[c].astype("object")
        return df

    @staticmethod
    def _save_dir_df(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df[DIR_COLUMNS].to_csv(path, index=False)

    @staticmethod
    def _yes_represents_above(title: str, market_id: str) -> bool:
        t = f"{title} {market_id}".upper()
        if any(k in t for k in ["OR BELOW", "BELOW", "UNDER", "LESS THAN"]):
            return False
        return True

    @staticmethod
    def _active_expiry_bucket(markets: List[Dict[str, Any]], now: pd.Timestamp) -> List[Dict[str, Any]]:
        rows: List[Tuple[float, Dict[str, Any]]] = []
        for m in markets:
            exp = pd.to_datetime(m.get("expiry_raw"), utc=True, errors="coerce")
            if pd.isna(exp):
                continue
            mins = float((exp - now).total_seconds() / 60.0)
            if 0.0 < mins <= 120.0:
                rows.append((mins, m))
        if not rows:
            return []
        active = min(x[0] for x in rows)
        return [m for mins, m in rows if abs(mins - active) <= 3.0]

    @staticmethod
    def _extract_settlement(payload: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        nested = payload.get("market") if isinstance(payload.get("market"), dict) else {}
        probes = [payload] + ([nested] if nested else [])

        status = ""
        for p in probes:
            status = str(
                p.get("status")
                or p.get("market_status")
                or p.get("marketStatus")
                or p.get("settlement_status")
                or p.get("settlementStatus")
                or ""
            ).strip().lower()
            if status:
                break
        settled = status in {"settled", "finalized", "resolved", "closed", "determined"}

        cands: List[Any] = []
        for p in probes:
            cands.extend(
                [
                    p.get("result"),
                    p.get("outcome"),
                    p.get("settlement_outcome"),
                    p.get("settlementOutcome"),
                    p.get("yes_outcome"),
                    p.get("yesOutcome"),
                    p.get("settlement_value"),
                    p.get("settlementValue"),
                    p.get("settlement_price"),
                    p.get("settlementPrice"),
                    p.get("final_value"),
                    p.get("finalValue"),
                    p.get("winning_outcome"),
                    p.get("winningOutcome"),
                    p.get("yes_settlement"),
                    p.get("yesSettlement"),
                    p.get("yes_result"),
                    p.get("yesResult"),
                ]
            )

        out: Optional[int] = None
        for c in cands:
            if c is None:
                continue
            if isinstance(c, (int, float)):
                out = 1 if float(c) >= 0.5 else 0
                break
            if isinstance(c, dict):
                if "yes" in c and c.get("yes") is not None:
                    try:
                        out = 1 if float(c.get("yes")) >= 0.5 else 0
                        break
                    except Exception:
                        pass
                if "value" in c:
                    sv = str(c.get("value")).strip().lower()
                    if sv in {"yes", "y", "true", "1"}:
                        out = 1
                        break
                    if sv in {"no", "n", "false", "0"}:
                        out = 0
                        break
            s = str(c).strip().lower()
            if s in {"yes", "y", "true", "1"}:
                out = 1
                break
            if s in {"no", "n", "false", "0"}:
                out = 0
                break

        if out is not None:
            settled = True
        return settled, out

    @staticmethod
    def _logloss(y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    @staticmethod
    def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        out = 0.0
        n = len(y)
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
            if not np.any(m):
                continue
            out += (np.sum(m) / n) * abs(float(np.mean(p[m])) - float(np.mean(y[m])))
        return float(out)

    @staticmethod
    def _bucket_table(y: np.ndarray, p: np.ndarray) -> List[Dict[str, Any]]:
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        rows: List[Dict[str, Any]] = []
        bins = np.arange(0.50, 0.951, 0.05)
        for lo in bins[:-1]:
            hi = lo + 0.05
            m = (p >= lo) & (p < hi)
            n = int(np.sum(m))
            if n == 0:
                continue
            wr = float(np.mean(y[m]))
            se = math.sqrt(max(wr * (1.0 - wr), 0.0) / max(n, 1))
            rows.append(
                {
                    "bucket": f"{lo:.2f}-{hi:.2f}",
                    "avg_pred": float(np.mean(p[m])),
                    "actual_yes_rate": wr,
                    "count": n,
                    "ci95_low": float(max(0.0, wr - 1.96 * se)),
                    "ci95_high": float(min(1.0, wr + 1.96 * se)),
                    "low_count": bool(n < 20),
                }
            )
        return rows

    def _next_expiry(self, now: pd.Timestamp) -> pd.Timestamp:
        exp = now + pd.Timedelta(minutes=max(1, int(self.cfg.directional_horizon_minutes)))
        return exp.floor("min")

    def _candidate_strikes(self, spot: float) -> List[float]:
        step = max(1.0, float(self.cfg.strike_step))
        # Always generate around spot using a configurable percentage band.
        band = float(np.clip(float(self.cfg.directional_spot_band_pct), 0.005, 0.20))
        lo = float(spot) * (1.0 - band)
        hi = float(spot) * (1.0 + band)
        center = round(float(spot) / step) * step

        # Candidate count: minimum 5, odd for center alignment, allow larger for faster sample collection.
        n_target = int(np.clip(int(self.cfg.strikes_per_cycle), 5, 101))
        if n_target % 2 == 0:
            n_target += 1
            if n_target > 101:
                n_target = 101
        k = n_target // 2

        arr = [center + (i * step) for i in range(-k, k + 1)]
        arr = [x for x in arr if lo <= x <= hi]

        # If band+step produced too few strikes, widen symmetrically around center to keep >=5.
        while len(arr) < 5:
            k += 1
            arr = [center + (i * step) for i in range(-k, k + 1)]
            arr = [x for x in arr if 5000 <= x <= 300000]
            if len(arr) >= 5:
                break

        return [float(x) for x in sorted(set(arr)) if 5000 <= float(x) <= 300000]

    def scan_and_log_directional_once(self) -> Dict[str, int]:
        now = pd.Timestamp.now(tz="UTC")
        warnings: List[str] = []
        try:
            spot = float(self.model._fetch_spot(warnings))
        except Exception:
            return {"generated_strikes": 0, "new_logged": 0, "evaluated": 0}

        expiry = self._next_expiry(now)
        horizon = max(1, int(round((expiry - now).total_seconds() / 60.0)))
        strikes = self._candidate_strikes(spot)
        if not strikes:
            return {"generated_strikes": 0, "new_logged": 0, "evaluated": 0}

        df = self._load_dir_df(self.directional_csv)
        existing = set()
        if not df.empty:
            key_df = df[["expiry_time", "strike"]].copy()
            for _, r in key_df.iterrows():
                existing.add((str(r["expiry_time"]), float(pd.to_numeric(pd.Series([r["strike"]]), errors="coerce").iloc[0])))

        factors = self.model._factors(horizon, warnings=[])
        annual_vol = max(float(factors.get("realized_vol_annual", 0.55)), 0.08)
        rows: List[Dict[str, Any]] = []

        for strike in strikes:
            key = (expiry.isoformat(), float(strike))
            if key in existing:
                continue
            raw_yes = self.model._model_probability(float(spot), float(strike), int(horizon), annual_vol, "above", factors)
            market_yes = self.model._market_probability(float(spot), float(strike), int(horizon), annual_vol, "above")
            if self.cfg.pure_model:
                model_yes = float(np.clip(raw_yes, 1e-6, 1 - 1e-6))
            else:
                model_yes = float(self.model._post_process_probability(raw_yes, market_prob=market_yes))
            predicted_yes = int(model_yes >= float(self.cfg.directional_yes_threshold))
            pred_dir = "YES" if predicted_yes == 1 else "NO"
            entry = float(market_yes if predicted_yes == 1 else (1.0 - market_yes))
            rows.append(
                {
                    "simulation_id": f"{expiry.strftime('%Y%m%d%H%M')}_{int(round(strike))}",
                    "timestamp_prediction": now.isoformat(),
                    "horizon_minutes": int(horizon),
                    "expiry_time": expiry.isoformat(),
                    "strike": float(strike),
                    "spot_price_at_prediction": float(spot),
                    "model_yes_probability": float(model_yes),
                    "market_yes_probability": float(market_yes),
                    "predicted_yes": int(predicted_yes),
                    "predicted_direction": pred_dir,
                    "resolved_outcome_yes": None,
                    "actual_direction": None,
                    "correct_direction": None,
                    "simulated_stake": float(self.cfg.directional_fake_stake),
                    "simulated_entry_price": float(np.clip(entry, 1e-6, 1 - 1e-6)),
                    "simulated_pnl": None,
                    "simulated_cumulative_pnl": None,
                    "resolved": False,
                    "settled_at_utc": None,
                }
            )

        if rows:
            new_df = pd.DataFrame([{c: row.get(c, None) for c in DIR_COLUMNS} for row in rows], columns=DIR_COLUMNS)
            if df.empty:
                df = new_df
            else:
                df = pd.concat([df, new_df], ignore_index=True)
            self._save_dir_df(self.directional_csv, df)

        return {"generated_strikes": int(len(strikes)), "new_logged": int(len(rows)), "evaluated": int(len(strikes))}

    def update_directional_outcomes_once(self) -> Dict[str, int]:
        df = self._load_dir_df(self.directional_csv)
        if df.empty:
            return {"checked": 0, "updated": 0, "resolved_total": 0}

        unresolved = df[~df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])].index.tolist()
        now = pd.Timestamp.now(tz="UTC")
        updated = 0

        for idx in unresolved:
            exp = pd.to_datetime(df.loc[idx, "expiry_time"], utc=True, errors="coerce")
            if pd.isna(exp) or now < exp:
                continue
            strike = pd.to_numeric(pd.Series([df.loc[idx, "strike"]]), errors="coerce").iloc[0]
            if pd.isna(strike):
                continue
            try:
                settle_px = float(self.model._price_at_or_after(pd.Timestamp(exp).to_pydatetime()))
            except Exception:
                continue
            outcome_yes = int(float(settle_px) >= float(strike))
            pred_yes = int(pd.to_numeric(pd.Series([df.loc[idx, "predicted_yes"]]), errors="coerce").fillna(0).iloc[0])
            correct = int(pred_yes == outcome_yes)
            entry = float(pd.to_numeric(pd.Series([df.loc[idx, "simulated_entry_price"]]), errors="coerce").fillna(0.5).iloc[0])
            stake = float(pd.to_numeric(pd.Series([df.loc[idx, "simulated_stake"]]), errors="coerce").fillna(self.cfg.directional_fake_stake).iloc[0])
            if pred_yes == 1:
                pnl = stake * (1.0 - entry) if outcome_yes == 1 else -stake * entry
            else:
                pnl = stake * (1.0 - entry) if outcome_yes == 0 else -stake * entry

            for c in ["resolved_outcome_yes", "resolved", "settled_at_utc", "actual_direction", "correct_direction", "simulated_pnl"]:
                df[c] = df[c].astype("object")
            df.loc[idx, "resolved_outcome_yes"] = int(outcome_yes)
            df.loc[idx, "actual_direction"] = "YES" if outcome_yes == 1 else "NO"
            df.loc[idx, "correct_direction"] = int(correct)
            df.loc[idx, "simulated_pnl"] = float(pnl)
            df.loc[idx, "resolved"] = True
            df.loc[idx, "settled_at_utc"] = self._now_iso()
            updated += 1

        # Running cumulative PnL in prediction order.
        df["timestamp_prediction"] = pd.to_datetime(df["timestamp_prediction"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_prediction").reset_index(drop=True)
        run = 0.0
        cum: List[Optional[float]] = []
        for _, r in df.iterrows():
            p = pd.to_numeric(pd.Series([r.get("simulated_pnl")]), errors="coerce").iloc[0]
            if pd.isna(p):
                cum.append(None)
            else:
                run += float(p)
                cum.append(float(run))
        df["simulated_cumulative_pnl"] = cum
        self._save_dir_df(self.directional_csv, df)

        resolved_total = int(df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"]).sum())
        return {"checked": int(len(unresolved)), "updated": int(updated), "resolved_total": int(resolved_total)}

    def compute_directional_report(self) -> Dict[str, Any]:
        df = self._load_dir_df(self.directional_csv)
        rep: Dict[str, Any] = {
            "generated_at_utc": self._now_iso(),
            "total_predictions": int(len(df)),
            "resolved_predictions": 0,
            "directional_accuracy": None,
            "headline_wr_source": "expiry_grouped",
            "yes_threshold": float(self.cfg.directional_yes_threshold),
            "pure_model": bool(self.cfg.pure_model),
            "thresholded_accuracy": {},
            "avg_model_yes_probability": None,
            "actual_yes_rate": None,
            "simulated_pnl": {"count": 0, "pnl_total": 0.0, "stake_total": 0.0, "roi": 0.0},
            "strike_level_accuracy": {},
            "expiry_level_accuracy": {"count": 0, "rows": []},
            "moneyness_bins": [],
        }
        if df.empty:
            self.directional_report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            return rep

        resolved = df[df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])].copy()
        rep["resolved_predictions"] = int(len(resolved))
        if resolved.empty:
            self.directional_report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            return rep

        resolved["model_yes_probability"] = pd.to_numeric(resolved["model_yes_probability"], errors="coerce")
        resolved["resolved_outcome_yes"] = pd.to_numeric(resolved["resolved_outcome_yes"], errors="coerce")
        resolved["correct_direction"] = pd.to_numeric(resolved["correct_direction"], errors="coerce")
        resolved["simulated_pnl"] = pd.to_numeric(resolved["simulated_pnl"], errors="coerce")
        resolved["simulated_stake"] = pd.to_numeric(resolved["simulated_stake"], errors="coerce")
        resolved = resolved.dropna(subset=["model_yes_probability", "resolved_outcome_yes", "correct_direction"])
        if resolved.empty:
            self.directional_report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            return rep

        p = np.clip(resolved["model_yes_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        y = resolved["resolved_outcome_yes"].to_numpy(dtype=float)
        c = resolved["correct_direction"].to_numpy(dtype=float)
        rep["avg_model_yes_probability"] = float(np.mean(p))
        rep["actual_yes_rate"] = float(np.mean(y))

        # Strike-level accuracy remains available for stress testing.
        rep["strike_level_accuracy"] = {
            "count": int(len(resolved)),
            "directional_accuracy": float(np.mean(c)),
            "avg_model_yes_probability": float(np.mean(p)),
            "actual_yes_rate": float(np.mean(y)),
        }

        # Expiry-grouped headline W/R to reduce correlated strike inflation.
        eg = resolved.copy()
        eg["expiry_time"] = pd.to_datetime(eg["expiry_time"], utc=True, errors="coerce")
        eg["model_yes_probability"] = pd.to_numeric(eg["model_yes_probability"], errors="coerce")
        eg["resolved_outcome_yes"] = pd.to_numeric(eg["resolved_outcome_yes"], errors="coerce")
        eg["correct_direction"] = pd.to_numeric(eg["correct_direction"], errors="coerce")
        eg = eg.dropna(subset=["expiry_time", "model_yes_probability", "resolved_outcome_yes", "correct_direction"])
        if not eg.empty:
            grp = (
                eg.groupby("expiry_time", dropna=True)
                .agg(
                    n_strikes=("strike", "count"),
                    strike_directional_accuracy=("correct_direction", "mean"),
                    avg_model_yes_probability=("model_yes_probability", "mean"),
                    actual_yes_rate=("resolved_outcome_yes", "mean"),
                )
                .reset_index()
            )
            rep["expiry_level_accuracy"] = {
                "count": int(len(grp)),
                "directional_accuracy_mean": float(grp["strike_directional_accuracy"].mean()),
                "rows": [
                    {
                        "expiry_time": pd.Timestamp(r["expiry_time"]).isoformat(),
                        "n_strikes": int(r["n_strikes"]),
                        "strike_directional_accuracy": float(r["strike_directional_accuracy"]),
                        "avg_model_yes_probability": float(r["avg_model_yes_probability"]),
                        "actual_yes_rate": float(r["actual_yes_rate"]),
                    }
                    for _, r in grp.iterrows()
                ],
            }
            rep["directional_accuracy"] = float(grp["strike_directional_accuracy"].mean())
        else:
            rep["directional_accuracy"] = float(np.mean(c))

        thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]
        t_out: Dict[str, Any] = {}
        for t in thresholds:
            # YES-threshold buckets are based on model_yes_probability only.
            m = p >= float(t)
            n = int(np.sum(m))
            t_out[f">={t:.2f}"] = {
                "count": n,
                "yes_win_rate": float(np.mean(y[m])) if n > 0 else None,
                "avg_model_yes_probability": float(np.mean(p[m])) if n > 0 else None,
                "directional_accuracy": float(np.mean(c[m])) if n > 0 else None,
            }
        rep["thresholded_accuracy"] = t_out

        # Moneyness bins around spot at prediction time.
        mb = resolved.copy()
        mb["strike"] = pd.to_numeric(mb["strike"], errors="coerce")
        mb["spot_price_at_prediction"] = pd.to_numeric(mb["spot_price_at_prediction"], errors="coerce")
        mb = mb.dropna(subset=["strike", "spot_price_at_prediction"])
        if not mb.empty:
            rel = np.abs((mb["strike"].to_numpy(dtype=float) / np.clip(mb["spot_price_at_prediction"].to_numpy(dtype=float), 1e-9, None)) - 1.0)
            labels = np.full(len(rel), "2%+", dtype=object)
            labels[rel <= 0.005] = "ATM_0.5%"
            labels[(rel > 0.005) & (rel <= 0.01)] = "0.5-1%"
            labels[(rel > 0.01) & (rel <= 0.02)] = "1-2%"
            mb["moneyness_bin"] = labels
            rows: List[Dict[str, Any]] = []
            for b in ["ATM_0.5%", "0.5-1%", "1-2%", "2%+"]:
                g = mb[mb["moneyness_bin"] == b]
                if g.empty:
                    rows.append({"bin": b, "count": 0, "directional_accuracy": None, "avg_model_yes_probability": None, "actual_yes_rate": None})
                    continue
                rows.append(
                    {
                        "bin": b,
                        "count": int(len(g)),
                        "directional_accuracy": float(pd.to_numeric(g["correct_direction"], errors="coerce").mean()),
                        "avg_model_yes_probability": float(pd.to_numeric(g["model_yes_probability"], errors="coerce").mean()),
                        "actual_yes_rate": float(pd.to_numeric(g["resolved_outcome_yes"], errors="coerce").mean()),
                    }
                )
            rep["moneyness_bins"] = rows

        pnl = resolved["simulated_pnl"].dropna()
        stake_total = float(resolved["simulated_stake"].fillna(self.cfg.directional_fake_stake).sum())
        pnl_total = float(pnl.sum()) if len(pnl) else 0.0
        rep["simulated_pnl"] = {
            "count": int(len(resolved)),
            "pnl_total": pnl_total,
            "stake_total": stake_total,
            "roi": float(pnl_total / stake_total) if stake_total > 0 else 0.0,
            "cumulative_last": float(pd.to_numeric(resolved["simulated_cumulative_pnl"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(resolved["simulated_cumulative_pnl"], errors="coerce").notna().any() else 0.0,
        }

        self.directional_report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return rep

    def scan_and_log_once(self) -> Dict[str, int]:
        markets = self.kalshi.fetch_btc_markets()
        if not markets:
            return {"seen_markets": 0, "active_group_markets": 0, "evaluated": 0, "new_logged": 0}

        now = pd.Timestamp.now(tz="UTC")
        warnings: List[str] = []
        try:
            spot = float(self.model._fetch_spot(warnings))
        except Exception:
            return {"seen_markets": len(markets), "active_group_markets": 0, "evaluated": 0, "new_logged": 0}

        active = self._active_expiry_bucket(markets, now)
        df = self._load_df(self.full_csv)
        logged = 0
        evaluated = 0
        factors_cache: Dict[int, Dict[str, float]] = {}
        append_rows: List[Dict[str, Any]] = []

        existing = set(df["market_id"].astype(str).tolist()) if not df.empty else set()

        for m in active:
            mid = str(m.get("market_id", ""))
            if not mid or mid in existing:
                continue
            strike = m.get("strike")
            try:
                strike = float(strike)
            except Exception:
                continue
            if not (5000 <= strike <= 300000):
                continue

            exp = pd.to_datetime(m.get("expiry_raw"), utc=True, errors="coerce")
            if pd.isna(exp):
                continue
            mins = float((exp - now).total_seconds() / 60.0)
            if not (self.cfg.min_minutes_left <= mins <= self.cfg.max_minutes_left):
                continue

            horizon = max(1, int(round(mins)))
            if horizon not in factors_cache:
                factors_cache[horizon] = self.model._factors(horizon, warnings=[])
            factors = factors_cache[horizon]

            annual_vol = max(float(factors.get("realized_vol_annual", 0.55)), 0.08)
            raw_above = self.model._model_probability(spot, strike, horizon, annual_vol, "above", factors)
            model_above = float(self.model._post_process_probability(raw_above, market_prob=None))
            yes_is_above = self._yes_represents_above(str(m.get("market_title", "")), mid)
            model_yes = model_above if yes_is_above else float(np.clip(1.0 - model_above, 1e-6, 1 - 1e-6))

            market_yes = float(m.get("yes_ask", m.get("yes_prob", 0.5)))
            market_no = float(m.get("no_ask", m.get("no_prob", 0.5)))
            market_yes = float(np.clip(market_yes, 1e-6, 1 - 1e-6))
            market_no = float(np.clip(market_no, 1e-6, 1 - 1e-6))

            ts_pred = pd.Timestamp.now(tz="UTC")
            if ts_pred >= pd.Timestamp(exp):
                continue

            flat_bet = int(model_yes >= self.cfg.flat_bet_threshold)
            flat_entry = market_yes if flat_bet else None
            row = {
                "timestamp_prediction": ts_pred.isoformat(),
                "market_id": mid,
                "strike": float(strike),
                "expiry_time": pd.Timestamp(exp).isoformat(),
                "expiration_window_utc": pd.Timestamp(exp).floor("h").isoformat(),
                "model_yes_probability": float(model_yes),
                "market_yes_probability": float(market_yes),
                "market_no_probability": float(market_no),
                "time_to_expiry": float(mins),
                "spot_price_at_prediction": float(spot),
                "resolved_outcome": None,
                "resolved": False,
                "settled_at_utc": None,
                "flat_bet_yes": int(flat_bet),
                "flat_entry_price_yes": float(flat_entry) if flat_entry is not None else None,
                "flat_stake": float(self.cfg.flat_stake) if flat_bet else None,
                "flat_pnl": None,
                "flat_cumulative_pnl": None,
            }
            append_rows.append({c: row.get(c, None) for c in FULL_COLUMNS})
            existing.add(mid)
            logged += 1
            evaluated += 1

        if append_rows:
            new_df = pd.DataFrame(append_rows, columns=FULL_COLUMNS)
            if df.empty:
                df = new_df
            else:
                df = pd.concat([df, new_df], ignore_index=True)

        self._save_df(self.full_csv, df)
        return {
            "seen_markets": len(markets),
            "active_group_markets": len(active),
            "evaluated": evaluated,
            "new_logged": logged,
        }

    def update_outcomes_once(self) -> Dict[str, int]:
        df = self._load_df(self.full_csv)
        if df.empty:
            return {"checked": 0, "updated": 0, "resolved_total": 0, "settled_no_outcome": 0, "fetch_fail": 0}

        unresolved = df[~df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])].index.tolist()
        updated = 0
        settled_no_outcome = 0
        fetch_fail = 0

        for idx in unresolved:
            mid = str(df.loc[idx, "market_id"])
            try:
                payload = self.kalshi.fetch_market(mid)
                settled, out = self._extract_settlement(payload)
                if not settled:
                    continue
                if out is None:
                    settled_no_outcome += 1
                    continue
                # Ensure object dtype for mixed writes.
                df["resolved_outcome"] = df["resolved_outcome"].astype("object")
                df["resolved"] = df["resolved"].astype("object")
                df["settled_at_utc"] = df["settled_at_utc"].astype("object")
                df["flat_pnl"] = df["flat_pnl"].astype("object")
                df.loc[idx, "resolved_outcome"] = int(out)
                df.loc[idx, "resolved"] = True
                df.loc[idx, "settled_at_utc"] = self._now_iso()

                flat = int(pd.to_numeric(pd.Series([df.loc[idx, "flat_bet_yes"]]), errors="coerce").fillna(0).iloc[0])
                if flat == 1:
                    entry = float(pd.to_numeric(pd.Series([df.loc[idx, "flat_entry_price_yes"]]), errors="coerce").fillna(0).iloc[0])
                    stake = float(pd.to_numeric(pd.Series([df.loc[idx, "flat_stake"]]), errors="coerce").fillna(self.cfg.flat_stake).iloc[0])
                    pnl = stake * (1.0 - entry) if int(out) == 1 else -stake * entry
                    df.loc[idx, "flat_pnl"] = float(pnl)
                updated += 1
            except Exception:
                fetch_fail += 1

        # cumulative flat pnl in prediction time order
        df["timestamp_prediction"] = pd.to_datetime(df["timestamp_prediction"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_prediction").reset_index(drop=True)
        run = 0.0
        cum = []
        for _, r in df.iterrows():
            p = pd.to_numeric(pd.Series([r.get("flat_pnl")]), errors="coerce").iloc[0]
            if pd.isna(p):
                cum.append(None)
            else:
                run += float(p)
                cum.append(float(run))
        df["flat_cumulative_pnl"] = cum

        self._save_df(self.full_csv, df)
        resolved_total = int(df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"]).sum())
        return {
            "checked": int(len(unresolved)),
            "updated": int(updated),
            "resolved_total": int(resolved_total),
            "settled_no_outcome": int(settled_no_outcome),
            "fetch_fail": int(fetch_fail),
        }

    def compute_report(self) -> Dict[str, Any]:
        df = self._load_df(self.full_csv)
        if df.empty:
            rep = {
                "generated_at_utc": self._now_iso(),
                "total_predictions": 0,
                "resolved_predictions": 0,
            }
            self.report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            return rep

        resolved = df[df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])].copy()
        for c in ["model_yes_probability", "market_yes_probability", "resolved_outcome", "flat_pnl", "flat_bet_yes"]:
            resolved[c] = pd.to_numeric(resolved[c], errors="coerce")
        resolved = resolved.dropna(subset=["model_yes_probability", "market_yes_probability", "resolved_outcome"])

        rep: Dict[str, Any] = {
            "generated_at_utc": self._now_iso(),
            "total_predictions": int(len(df)),
            "resolved_predictions": int(len(resolved)),
            "avg_predicted_probability": None,
            "overall_win_rate": None,
            "brier_score": None,
            "market_brier_score": None,
            "naive50_brier_score": None,
            "log_loss": None,
            "ece": None,
            "calibration_buckets": [],
            "reliability_table": [],
            "threshold_metrics": {},
            "simulated_flat_pnl": {},
            "strike_level_accuracy": {},
            "expiration_level_accuracy": {},
            "questions_notes": {
                "probabilities_independent_per_strike": False,
                "probability_model": "shared distribution per horizon then strike-level CDF readout",
                "missing_feature_checks": ["spot", "realized_vol_annual", "orderbook/trade factors", "macro/behavioral factors"],
                "bias_checks": ["lookahead_guard_enabled", "first_prediction_per_contract_only", "grouped_expiration_metrics_included"],
            },
        }

        if resolved.empty:
            self.report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            return rep

        y = resolved["resolved_outcome"].to_numpy(dtype=float)
        p = np.clip(resolved["model_yes_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        m = np.clip(resolved["market_yes_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

        rep["avg_predicted_probability"] = float(np.mean(p))
        rep["overall_win_rate"] = float(np.mean(y))
        rep["brier_score"] = float(np.mean((p - y) ** 2))
        rep["market_brier_score"] = float(np.mean((m - y) ** 2))
        rep["naive50_brier_score"] = float(np.mean((0.5 - y) ** 2))
        rep["log_loss"] = self._logloss(y, p)
        rep["ece"] = self._ece(y, p, bins=10)

        buckets = self._bucket_table(y, p)
        rep["calibration_buckets"] = buckets
        rep["reliability_table"] = buckets

        thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]
        tm: Dict[str, Any] = {}
        for t in thresholds:
            mask = p >= t
            n = int(np.sum(mask))
            if n == 0:
                tm[f">={t:.2f}"] = {"count": 0, "actual_yes_rate": None, "avg_pred": None, "brier": None, "log_loss": None}
            else:
                tm[f">={t:.2f}"] = {
                    "count": n,
                    "actual_yes_rate": float(np.mean(y[mask])),
                    "avg_pred": float(np.mean(p[mask])),
                    "brier": float(np.mean((p[mask] - y[mask]) ** 2)),
                    "log_loss": self._logloss(y[mask], p[mask]),
                }
        rep["threshold_metrics"] = tm

        # Flat betting simulation: model_yes >= threshold, buy YES @ ask, $15 stake.
        flat = resolved[resolved["flat_bet_yes"].fillna(0).astype(int) == 1].copy()
        if flat.empty:
            rep["simulated_flat_pnl"] = {"count": 0, "stake_total": 0.0, "pnl_total": 0.0, "roi": 0.0, "cumulative_last": 0.0}
        else:
            flat_pnl = pd.to_numeric(flat["flat_pnl"], errors="coerce").dropna()
            stake_total = float(pd.to_numeric(flat["flat_stake"], errors="coerce").fillna(self.cfg.flat_stake).sum())
            pnl_total = float(flat_pnl.sum()) if len(flat_pnl) else 0.0
            rep["simulated_flat_pnl"] = {
                "count": int(len(flat)),
                "stake_total": stake_total,
                "pnl_total": pnl_total,
                "roi": float(pnl_total / stake_total) if stake_total > 0 else 0.0,
                "cumulative_last": float(pd.to_numeric(flat["flat_cumulative_pnl"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(flat["flat_cumulative_pnl"], errors="coerce").notna().any() else 0.0,
            }

        # Strike-level accuracy (prediction-level)
        rep["strike_level_accuracy"] = {
            "count": int(len(resolved)),
            "avg_pred": float(np.mean(p)),
            "actual_yes_rate": float(np.mean(y)),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": self._logloss(y, p),
        }

        # Expiration-level accuracy to reduce correlated-strike inflation.
        g = resolved.copy()
        g["expiry_hour"] = pd.to_datetime(g["expiry_time"], utc=True, errors="coerce").dt.floor("h")
        grp = g.groupby("expiry_hour", dropna=True).agg(pred=("model_yes_probability", "mean"), actual=("resolved_outcome", "mean"), n=("market_id", "count")).reset_index()
        if grp.empty:
            rep["expiration_level_accuracy"] = {"count": 0}
        else:
            yp = grp["actual"].to_numpy(dtype=float)
            pp = np.clip(grp["pred"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            rep["expiration_level_accuracy"] = {
                "count": int(len(grp)),
                "avg_pred": float(np.mean(pp)),
                "actual_yes_rate": float(np.mean(yp)),
                "brier": float(np.mean((pp - yp) ** 2)),
                "log_loss": self._logloss((yp >= 0.5).astype(float), pp),
                "avg_strikes_per_expiration": float(np.mean(grp["n"].to_numpy(dtype=float))),
            }

        self.report_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return rep

    def run_auto_loop(self, poll_seconds: int = 300, stop_at_resolved: Optional[int] = None) -> None:
        print("[validation-tracker] auto loop started | full probability validation")
        print(f"[validation-tracker] poll={poll_seconds}s stop_at_resolved={stop_at_resolved}")
        while True:
            s = self.scan_and_log_once()
            u = self.update_outcomes_once()
            r = self.compute_report()
            print(
                "[validation-tracker] auto-cycle "
                f"seen_markets={s.get('seen_markets', 0)} "
                f"active_group_markets={s.get('active_group_markets', 0)} "
                f"evaluated={s.get('evaluated', 0)} "
                f"new_logged={s.get('new_logged', 0)} "
                f"checked={u.get('checked', 0)} "
                f"updated={u.get('updated', 0)} "
                f"settled_no_outcome={u.get('settled_no_outcome', 0)} "
                f"fetch_fail={u.get('fetch_fail', 0)} "
                f"resolved_total={u.get('resolved_total', 0)} "
                f"report_resolved={r.get('resolved_predictions', 0)}"
            )
            if self.kalshi.last_fetch_error:
                print(f"[validation-tracker] fetch-warning: {self.kalshi.last_fetch_error}")
            if stop_at_resolved is not None and int(u.get("resolved_total", 0)) >= int(stop_at_resolved):
                print(f"[validation-tracker] reached resolved target: {stop_at_resolved}. stopping.")
                return
            time.sleep(max(30, int(poll_seconds)))

    def run_directional_auto_loop(self, poll_seconds: int = 300, stop_at_resolved: Optional[int] = None) -> None:
        # Start fresh for each directional simulation run.
        if self.directional_csv.exists():
            self.directional_csv.unlink()
        if self.directional_report_json.exists():
            self.directional_report_json.unlink()
        print("[validation-tracker] directional auto loop started | synthetic strike simulation")
        print(
            f"[validation-tracker] poll={poll_seconds}s stop_at_resolved={stop_at_resolved} "
            f"yes_threshold={self.cfg.directional_yes_threshold:.2f} horizon={self.cfg.directional_horizon_minutes}m "
            f"spot_band=±{self.cfg.directional_spot_band_pct*100:.2f}% step={self.cfg.strike_step} "
            f"strikes_per_cycle={self.cfg.strikes_per_cycle} pure_model={self.cfg.pure_model}"
        )
        while True:
            s = self.scan_and_log_directional_once()
            u = self.update_directional_outcomes_once()
            r = self.compute_directional_report()
            print(
                "[validation-tracker] directional-cycle "
                f"generated_strikes={s.get('generated_strikes', 0)} "
                f"new_logged={s.get('new_logged', 0)} "
                f"checked={u.get('checked', 0)} "
                f"updated={u.get('updated', 0)} "
                f"resolved_total={u.get('resolved_total', 0)} "
                f"report_resolved={r.get('resolved_predictions', 0)} "
                f"w/r={('%.2f%%' % (100.0 * float(r.get('directional_accuracy')))) if r.get('directional_accuracy') is not None else 'n/a'}"
            )
            if stop_at_resolved is not None and int(u.get("resolved_total", 0)) >= int(stop_at_resolved):
                print(f"[validation-tracker] reached directional resolved target: {stop_at_resolved}. stopping.")
                print(json.dumps(r, indent=2))
                return
            time.sleep(max(30, int(poll_seconds)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BTCNEW full probability validation tracker")
    p.add_argument("--out-dir", type=str, default="artifacts/validation")
    p.add_argument("--flat-bet-threshold", type=float, default=0.65)
    p.add_argument("--flat-stake", type=float, default=15.0)
    p.add_argument("--min-minutes-left", type=int, default=10)
    p.add_argument("--max-minutes-left", type=int, default=90)
    p.add_argument("--yes-threshold", type=float, default=0.65)
    p.add_argument("--fake-stake", type=float, default=15.0)
    p.add_argument("--sim-horizon-minutes", type=int, default=60)
    p.add_argument("--spot-band-pct", type=float, default=0.03)
    p.add_argument("--strike-step", type=float, default=500.0)
    p.add_argument("--strikes-per-cycle", type=int, default=21)
    p.add_argument("--pure-model", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)
    au = sub.add_parser("auto")
    au.add_argument("--poll-seconds", type=int, default=300)
    au.add_argument("--stop-at-resolved", type=int, default=0)
    sim = sub.add_parser("sim-auto")
    sim.add_argument("--poll-seconds", type=int, default=300)
    sim.add_argument("--stop-at-resolved", type=int, default=0)
    sub.add_parser("update")
    sub.add_parser("summary")
    sub.add_parser("sim-update")
    sub.add_parser("sim-summary")
    return p


def _resolve_path(raw: str, base_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=base_dir / ".env")
    args = build_parser().parse_args()
    cfg = Config(
        flat_bet_threshold=float(args.flat_bet_threshold),
        flat_stake=float(args.flat_stake),
        min_minutes_left=int(args.min_minutes_left),
        max_minutes_left=int(args.max_minutes_left),
        directional_yes_threshold=float(args.yes_threshold),
        directional_fake_stake=float(args.fake_stake),
        directional_horizon_minutes=int(args.sim_horizon_minutes),
        directional_spot_band_pct=float(args.spot_band_pct),
        strike_step=float(args.strike_step),
        strikes_per_cycle=int(args.strikes_per_cycle),
        pure_model=bool(args.pure_model),
    )
    tracker = ValidationTracker(_resolve_path(args.out_dir, base_dir), cfg)

    if args.cmd == "auto":
        stop = None if int(args.stop_at_resolved) <= 0 else int(args.stop_at_resolved)
        tracker.run_auto_loop(poll_seconds=int(args.poll_seconds), stop_at_resolved=stop)
        return
    if args.cmd == "update":
        print(json.dumps(tracker.update_outcomes_once(), indent=2))
        return
    if args.cmd == "summary":
        print(json.dumps(tracker.compute_report(), indent=2))
        return
    if args.cmd == "sim-update":
        print(json.dumps(tracker.update_directional_outcomes_once(), indent=2))
        return
    if args.cmd == "sim-summary":
        print(json.dumps(tracker.compute_directional_report(), indent=2))
        return
    if args.cmd == "sim-auto":
        stop = None if int(args.stop_at_resolved) <= 0 else int(args.stop_at_resolved)
        tracker.run_directional_auto_loop(poll_seconds=int(args.poll_seconds), stop_at_resolved=stop)
        return


if __name__ == "__main__":
    main()
