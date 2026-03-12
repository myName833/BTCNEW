#!/usr/bin/env python3
"""BTC structural mispricing detection engine.

This script is intentionally separate from btc_prob_model.py.
It implements:
- Structural probability model: P(BTC > target in Y minutes)
- Walk-forward validation with Brier + simulated Kalshi EV
- Platt calibration (default)
- Missing-data-safe feature pipeline
- Regime classification + probability inertia
- Continuous runner with Kalshi odds polling + edge rules + alerts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

EPS = 1e-9
COINBASE_EXCHANGE_BASE = "https://api.exchange.coinbase.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
FRED_BASE = "https://api.stlouisfed.org/fred"
BLS_CALENDAR_ICS = "https://www.bls.gov/schedule/news_release/bls.ics"
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"
STATE_DIR = ARTIFACTS_DIR / "state"
LATEST_DIR = ARTIFACTS_DIR / "latest"
KALSHI_LAST_ERRORS: List[str] = []


def _kalshi_note_error(msg: str) -> None:
    KALSHI_LAST_ERRORS.append(str(msg))
    if len(KALSHI_LAST_ERRORS) > 200:
        del KALSHI_LAST_ERRORS[:-200]


@dataclass
class WalkForwardConfig:
    train_bars: int
    test_bars: int
    step_bars: int


@dataclass
class RunnerState:
    contract_prev: Dict[str, Dict[str, object]]
    active_snapshot: Optional[Dict[str, object]]
    signal_persistence: Dict[str, int]
    alert_dedup: Dict[str, pd.Timestamp]
    stoploss_dedup: Dict[str, pd.Timestamp]
    open_positions: Dict[str, Dict[str, object]]
    focus_market_ticker: Optional[str]
    last_position_update_at: Dict[str, pd.Timestamp]
    strike_last_eval: Dict[str, pd.Timestamp]
    spot_samples: List[Tuple[pd.Timestamp, float]]
    market_cache: List[Dict[str, object]]
    market_cache_updated_at: Optional[pd.Timestamp]


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=BASE_DIR / ".env")


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _fmt_ts(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
    et = t.tz_convert("America/New_York")
    return f"{et.strftime('%Y-%m-%d %I:%M:%S %p ET')} ({t.strftime('%Y-%m-%d %H:%M:%S UTC')})"


def _parse_float_input(v: str) -> float:
    s = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
    return float(s)


def _parse_prob_input(v: str) -> float:
    x = _parse_float_input(v)
    if x > 1.0:
        x /= 100.0
    return float(np.clip(x, 0.0, 1.0))


def _safe_get_json(url: str, params: Dict[str, str] | None = None, headers: Dict[str, str] | None = None) -> object:
    timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
    if "kalshi.com" in url:
        timeout = float(os.getenv("KALSHI_HTTP_TIMEOUT_SECONDS", "6"))
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _safe_post_json(url: str, payload: Dict[str, object], headers: Dict[str, str] | None = None) -> None:
    requests.post(url, json=payload, headers=headers, timeout=15)


def _interval_minutes(interval: str) -> int:
    s = interval.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    raise ValueError("interval must look like 5m/15m/1h")


def _coinbase_granularity(interval: str) -> int:
    m = _interval_minutes(interval)
    sec = m * 60
    allowed = {60, 300, 900, 3600, 21600, 86400}
    if sec not in allowed:
        raise ValueError(f"Unsupported Coinbase granularity for {interval}")
    return sec


def _period_days(period: str) -> int:
    s = period.strip().lower()
    if s.endswith("d"):
        return int(float(s[:-1]))
    if s.endswith("mo"):
        return int(float(s[:-2]) * 30)
    if s.endswith("y"):
        return int(float(s[:-1]) * 365)
    raise ValueError("Unsupported period (use Nd/Nmo/Ny)")


def fetch_coinbase_candles(interval: str, period: str, strict_fresh_minutes: Optional[int] = None) -> pd.DataFrame:
    product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    gran = _coinbase_granularity(interval)
    end = _now_utc().floor("min")
    start = end - pd.Timedelta(days=_period_days(period))

    rows: List[list] = []
    max_points = 300
    step = pd.Timedelta(seconds=gran * max_points)
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        params = {
            "start": cur.isoformat().replace("+00:00", "Z"),
            "end": nxt.isoformat().replace("+00:00", "Z"),
            "granularity": str(gran),
        }
        data = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product}/candles", params=params)
        if isinstance(data, list) and data:
            rows.extend(data)
        cur = nxt

    if not rows:
        raise ValueError("No Coinbase candle data returned")

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates(subset=["time"]).set_index("time").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Always append latest 1-minute live candle snapshot for freshness.
    now_utc = _now_utc()
    params = {
        "start": (now_utc - pd.Timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
        "end": now_utc.isoformat().replace("+00:00", "Z"),
        "granularity": "60",
    }
    live = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product}/candles", params=params)
    if isinstance(live, list) and live:
        dfl = pd.DataFrame(live, columns=["time", "low", "high", "open", "close", "volume"])
        dfl["time"] = pd.to_datetime(dfl["time"], unit="s", utc=True)
        latest = dfl.sort_values("time").iloc[-1]
        ts = pd.Timestamp(latest["time"])
        df.loc[ts, ["open", "high", "low", "close", "volume"]] = [
            float(latest["open"]),
            float(latest["high"]),
            float(latest["low"]),
            float(latest["close"]),
            float(latest["volume"]),
        ]
        df = df.sort_index()

    if strict_fresh_minutes is not None:
        age_min = float((_now_utc() - pd.Timestamp(df.index.max())).total_seconds() / 60.0)
        if age_min > strict_fresh_minutes:
            raise ValueError(f"Stale candles: {age_min:.2f} min > allowed {strict_fresh_minutes}")

    return df


def fetch_coinbase_spot() -> float:
    product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    data = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product}/ticker")
    if not isinstance(data, dict) or "price" not in data:
        raise ValueError("Coinbase ticker missing price")
    return float(data["price"])


def fetch_order_book_depth_ratio() -> float:
    product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    data = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product}/book", params={"level": "2"})
    if not isinstance(data, dict):
        raise ValueError("Coinbase book payload invalid")
    bids = data.get("bids", [])
    asks = data.get("asks", [])
    bid_notional = sum(float(px) * float(sz) for px, sz, *_ in bids[:200])
    ask_notional = sum(float(px) * float(sz) for px, sz, *_ in asks[:200])
    if ask_notional <= 0:
        return 1.0
    return float(bid_notional / (ask_notional + EPS))


def fetch_trade_flow(interval_minutes: int = 60) -> Dict[str, float]:
    product = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    rows = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product}/trades", params={"limit": "1000"})
    if not isinstance(rows, list) or not rows:
        return {"trade_flow_imbalance": 0.0, "trade_flow_buy_share": 0.5, "trade_flow_notional_usd": 0.0}

    df = pd.DataFrame(rows)
    if not {"time", "price", "size"}.issubset(df.columns):
        return {"trade_flow_imbalance": 0.0, "trade_flow_buy_share": 0.5, "trade_flow_notional_usd": 0.0}

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["time", "price", "size"]).sort_values("time")
    cutoff = _now_utc() - pd.Timedelta(minutes=interval_minutes)
    df = df[df["time"] >= cutoff]
    if df.empty:
        return {"trade_flow_imbalance": 0.0, "trade_flow_buy_share": 0.5, "trade_flow_notional_usd": 0.0}

    df["notional"] = df["price"] * df["size"]
    side = df["side"].astype(str).str.lower() if "side" in df.columns else pd.Series(["" for _ in range(len(df))])
    buy = df[side == "buy"]["notional"].sum()
    sell = df[side == "sell"]["notional"].sum()
    total = float(buy + sell)
    if total <= 0:
        return {"trade_flow_imbalance": 0.0, "trade_flow_buy_share": 0.5, "trade_flow_notional_usd": 0.0}
    return {
        "trade_flow_imbalance": float((buy - sell) / (total + EPS)),
        "trade_flow_buy_share": float(buy / (total + EPS)),
        "trade_flow_notional_usd": total,
    }


def _coinalyze_interval(interval: str) -> str:
    m = _interval_minutes(interval)
    mapping = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "1hour", 240: "4hour", 1440: "daily"}
    return mapping.get(m, "1hour")


def _coinalyze_history(payload: object, candidates: Sequence[str]) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    obj = payload[0]
    if not isinstance(obj, dict) or "history" not in obj:
        return pd.DataFrame()
    hist = obj.get("history")
    if not isinstance(hist, list) or not hist:
        return pd.DataFrame()

    df = pd.DataFrame(hist)
    if "t" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    out = pd.DataFrame(index=df.index)
    for c in candidates:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def _align_to_index(data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align external time series to target index with backward asof merge."""
    if data.empty:
        return pd.DataFrame(index=target_index)
    left = pd.DataFrame({"timestamp": target_index})
    right = data.reset_index().rename(columns={data.index.name or "index": "timestamp"}).sort_values("timestamp")
    left = left.sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    return merged.set_index("timestamp")


def fetch_coinalyze(index: pd.DatetimeIndex, interval: str) -> pd.DataFrame:
    key = os.getenv("COINALYZE_API_KEY", "").strip()
    symbol = os.getenv("COINALYZE_SYMBOL", "BTCUSDT_PERP.A")
    if not key:
        raise ValueError("COINALYZE_API_KEY is required")

    frm = int(index.min().timestamp())
    to = int(index.max().timestamp())
    iv = _coinalyze_interval(interval)

    out = pd.DataFrame(index=index)
    out[["funding_rate", "liquidations_long_usd", "liquidations_short_usd", "open_interest", "oi_change"]] = np.nan

    common = {"symbols": symbol, "interval": iv, "from": str(frm), "to": str(to), "api_key": key}

    try:
        p = _safe_get_json(f"{COINALYZE_BASE}/funding-rate-history", params=common)
        x = _coinalyze_history(p, ["c", "value", "funding_rate", "rate", "r", "fr", "f"])
        if not x.empty:
            src = "c" if "c" in x.columns else list(x.columns)[0]
            xa = _align_to_index(x[[src]].rename(columns={src: "funding_rate"}), out.index)
            out["funding_rate"] = xa["funding_rate"]
    except Exception:
        pass

    try:
        lp = dict(common)
        lp["convert_to_usd"] = "true"
        p = _safe_get_json(f"{COINALYZE_BASE}/liquidation-history", params=lp)
        x = _coinalyze_history(p, ["l", "s", "longs", "shorts"])
        lx = pd.DataFrame(index=x.index)
        if "l" in x.columns:
            lx["liquidations_long_usd"] = x["l"]
        elif "longs" in x.columns:
            lx["liquidations_long_usd"] = x["longs"]
        if "s" in x.columns:
            lx["liquidations_short_usd"] = x["s"]
        elif "shorts" in x.columns:
            lx["liquidations_short_usd"] = x["shorts"]
        if not lx.empty:
            la = _align_to_index(lx, out.index)
            for c in ["liquidations_long_usd", "liquidations_short_usd"]:
                if c in la.columns:
                    out[c] = la[c]
    except Exception:
        pass

    try:
        p = _safe_get_json(f"{COINALYZE_BASE}/open-interest-history", params=common)
        x = _coinalyze_history(p, ["c", "value", "open_interest"])
        if not x.empty:
            src = "c" if "c" in x.columns else list(x.columns)[0]
            oa = _align_to_index(x[[src]].rename(columns={src: "open_interest"}), out.index)
            out["open_interest"] = oa["open_interest"]
            out["oi_change"] = out["open_interest"].diff()
    except Exception:
        pass

    return out.sort_index()


def fetch_macro_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    dates = set()
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key:
        try:
            payload = _safe_get_json(
                f"{FRED_BASE}/releases/dates",
                params={
                    "api_key": fred_key,
                    "file_type": "json",
                    "realtime_start": index.min().strftime("%Y-%m-%d"),
                    "realtime_end": index.max().strftime("%Y-%m-%d"),
                    "limit": "1000",
                },
            )
            rel = payload.get("release_dates", []) if isinstance(payload, dict) else []
            for r in rel:
                d = pd.to_datetime(r.get("date"), utc=True, errors="coerce")
                if pd.notna(d):
                    dates.add(d.date())
        except Exception:
            pass

    try:
        txt = requests.get(BLS_CALENDAR_ICS, timeout=15).text
        for line in txt.splitlines():
            if line.startswith("DTSTART:"):
                tok = line.split(":", 1)[1].strip()
                d = pd.to_datetime(tok, utc=True, errors="coerce")
                if pd.notna(d):
                    dates.add(d.date())
    except Exception:
        pass

    out = pd.DataFrame(index=index)
    out["macro_event_flag"] = [1.0 if ts.date() in dates else 0.0 for ts in index]
    return out


def merge_live_factors(
    index: pd.DatetimeIndex,
    interval: str,
    strict: bool = True,
    critical_factor_max_age_minutes: int = 90,
) -> pd.DataFrame:
    factors = pd.DataFrame(index=index)
    factors = factors.join(fetch_coinalyze(index, interval), how="left")

    # Snapshot microstructure factors, applied to current index row and carried forward only for short horizon.
    depth_ratio = fetch_order_book_depth_ratio()
    tf = fetch_trade_flow(interval_minutes=60)
    factors["depth_ratio"] = depth_ratio
    factors["order_book_imbalance"] = (depth_ratio - 1.0) / (depth_ratio + 1.0 + EPS)
    factors["trade_flow_imbalance"] = tf["trade_flow_imbalance"]
    factors["trade_flow_buy_share"] = tf["trade_flow_buy_share"]
    factors["trade_flow_notional_usd"] = tf["trade_flow_notional_usd"]

    factors = factors.join(fetch_macro_flags(index), how="left")

    # Missing handling: forward-fill + missing flags, but preserve raw coverage checks before filling.
    critical = ["funding_rate", "liquidations_long_usd", "liquidations_short_usd", "order_book_imbalance"]
    c_meta = ["open_interest", "oi_change", *critical]
    for c in c_meta:
        if c not in factors.columns:
            factors[c] = np.nan

    raw = factors[c_meta].copy()
    for c in c_meta:
        factors[f"{c}_missing_flag"] = factors[c].isna().astype(float)

    factors = factors.sort_index()
    factors[c_meta] = factors[c_meta].ffill()

    # Keep NaNs if source never provided a value; do not force-mutate missing into zero.
    # Modeling path later imputes for math, while *_missing_flag preserves missingness signal.
    last_valid_age_minutes: Dict[str, Optional[float]] = {}
    recent_coverage_ratio: Dict[str, float] = {}
    recency_minutes = max(30, _interval_minutes(interval) * 6)
    recent_cutoff = factors.index.max() - pd.Timedelta(minutes=recency_minutes)
    recent_raw = raw.loc[raw.index >= recent_cutoff]
    for c in c_meta:
        ser = raw[c]
        valid_idx = ser.dropna().index
        if len(valid_idx) == 0:
            last_valid_age_minutes[c] = None
        else:
            age_min = float((factors.index.max() - valid_idx.max()).total_seconds() / 60.0)
            last_valid_age_minutes[c] = max(0.0, age_min)
        recent_coverage_ratio[c] = float(recent_raw[c].notna().mean()) if len(recent_raw) else 0.0

    if strict:
        bad: List[str] = []
        for c in critical:
            age = last_valid_age_minutes.get(c)
            cov = recent_coverage_ratio.get(c, 0.0)
            # Require some recent real observations from source for funding/liquidation features.
            if c in {"funding_rate", "liquidations_long_usd", "liquidations_short_usd"}:
                if age is None or age > float(critical_factor_max_age_minutes) or cov <= 0.0:
                    bad.append(c)
            else:
                # snapshot microstructure must exist now
                if age is None:
                    bad.append(c)
        if bad:
            raise ValueError(
                f"Critical live factors stale/missing: {bad}. "
                f"max_age={critical_factor_max_age_minutes}m"
            )

    factors.attrs["live_quality"] = {
        "recent_window_minutes": recency_minutes,
        "critical_factor_max_age_minutes": int(critical_factor_max_age_minutes),
        "last_valid_age_minutes": last_valid_age_minutes,
        "recent_coverage_ratio": recent_coverage_ratio,
    }

    return factors


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2) + EPS

    def _slope(y: np.ndarray) -> float:
        y_mean = np.mean(y)
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def build_structural_features(
    market: pd.DataFrame,
    factors: pd.DataFrame,
    horizon_minutes: int,
    interval: str,
) -> pd.DataFrame:
    df = market.join(factors, how="left").copy()
    interval_min = _interval_minutes(interval)
    horizon_bars = max(1, int(math.ceil(horizon_minutes / interval_min)))

    df["ret_1"] = np.log(df["close"]).diff()
    # replace reactive raw returns with smoothed structural return states
    df["ret_mean_5"] = df["ret_1"].rolling(5, min_periods=3).mean()
    df["ret_mean_3"] = df["ret_1"].rolling(3, min_periods=2).mean()
    df["trend_slope_12"] = _rolling_slope(df["close"].apply(np.log), 12)

    df["vol_24"] = df["ret_1"].rolling(24, min_periods=12).std()
    df["ret_z"] = df["ret_1"] / (df["vol_24"] + EPS)

    # distance anchor construction
    expected_move = (df["vol_24"].clip(lower=EPS) * math.sqrt(max(horizon_bars, 1)))
    df["expected_move"] = expected_move

    # structural expansion
    liq_total = df["liquidations_long_usd"].clip(lower=0) + df["liquidations_short_usd"].clip(lower=0)
    liq_mean = liq_total.rolling(96, min_periods=24).mean()
    liq_std = liq_total.rolling(96, min_periods=24).std()
    df["liq_z"] = (liq_total - liq_mean) / (liq_std + EPS)

    fund_mean = df["funding_rate"].rolling(96, min_periods=24).mean()
    df["funding_slope"] = fund_mean.diff(3)

    tf_smooth_w = int(float(os.getenv("TRADE_FLOW_SMOOTH_W", "4")))
    tf_smooth_w = max(1, tf_smooth_w)
    df["trade_flow_imbalance_smooth"] = df["trade_flow_imbalance"].rolling(tf_smooth_w, min_periods=1).mean()

    vol_rank_w = int(float(os.getenv("VOL_RANK_WINDOW", "288")))
    vol_rank_w = max(24, vol_rank_w)
    df["vol_rank"] = df["vol_24"].rolling(vol_rank_w, min_periods=24).rank(pct=True)

    df["oi_accel"] = df["oi_change"].diff(3)

    # target
    df["fwd_ret_h"] = np.log(df["close"].shift(-horizon_bars) / df["close"])

    return df


def classify_regime(df: pd.DataFrame) -> pd.Series:
    # 0 low vol, 1 medium vol, 2 high vol, 3 macro event
    r = pd.Series(np.ones(len(df), dtype=int), index=df.index)
    if "vol_rank" in df.columns:
        r[df["vol_rank"] <= 0.33] = 0
        r[df["vol_rank"] >= 0.67] = 2
    if "macro_event_flag" in df.columns:
        r[df["macro_event_flag"] > 0] = 3
    return r.rename("regime")


def make_training_panel(df: pd.DataFrame, z_grid: Sequence[float]) -> pd.DataFrame:
    base = df.dropna(subset=["fwd_ret_h", "expected_move", "vol_24"]).copy()
    rows = []
    for z in z_grid:
        t = base.copy()
        t["threshold_z"] = float(z)
        t["threshold_ret"] = t["expected_move"] * float(z)
        t["threshold_abs"] = t["threshold_ret"].abs()
        t["threshold_ret_sq"] = t["threshold_ret"] ** 2
        t["threshold_z_abs"] = t["threshold_z"].abs()
        t["y_event"] = (t["fwd_ret_h"] > t["threshold_ret"]).astype(int)
        rows.append(t)
    return pd.concat(rows, axis=0).sort_index()


def split_walk_forward(n: int, cfg: WalkForwardConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    s = 0
    while True:
        tr_e = s + cfg.train_bars
        te_e = tr_e + cfg.test_bars
        if te_e > n:
            break
        out.append((np.arange(s, tr_e), np.arange(tr_e, te_e)))
        s += cfg.step_bars
    if not out:
        raise ValueError("No walk-forward splits. Reduce bars or increase data")
    return out


def fit_calibrator(raw_pred: np.ndarray, y: np.ndarray, method: str) -> object:
    m = method.lower()
    yp = np.asarray(y).reshape(-1)
    rp = np.asarray(raw_pred, dtype=float).reshape(-1)
    # Calibrators need at least two classes; otherwise fallback to passthrough.
    if len(yp) == 0 or len(np.unique(yp)) < 2:
        return {"method": "none", "model": None}
    if m == "none":
        return {"method": "none", "model": None}
    if m == "platt":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(rp.reshape(-1, 1), yp)
        return {"method": "platt", "model": lr}
    if m == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(rp, yp)
        return {"method": "isotonic", "model": iso}
    raise ValueError("Unknown calibration method")


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


def eval_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "event_rate": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
    }


def simulated_kalshi_ev(
    probs: np.ndarray,
    y_true: np.ndarray,
    spread_bps: float = 150.0,
    min_liquidity: float = 2000.0,
) -> Dict[str, float]:
    """Simulate EV against synthetic Kalshi-like pricing with spread/liquidity gates."""
    rng = np.random.default_rng(42)
    base = probs + rng.normal(0.0, 0.05, size=len(probs))
    market_p = np.clip(base, 0.05, 0.95)
    spread = spread_bps / 10000.0
    yes_ask = np.clip(market_p + spread / 2.0, 0.01, 0.99)
    no_ask = np.clip((1.0 - market_p) + spread / 2.0, 0.01, 0.99)
    liq = rng.lognormal(mean=8.0, sigma=0.7, size=len(probs))

    pnl = []
    n_trades = 0
    for i in range(len(probs)):
        if liq[i] < min_liquidity:
            continue
        p = probs[i]
        y = y_true[i]
        # simple edge rule
        edge_yes = p - yes_ask[i]
        edge_no = (1.0 - p) - no_ask[i]
        if edge_yes >= 0.10:
            n_trades += 1
            pnl.append((1.0 if y == 1 else 0.0) - yes_ask[i])
        elif edge_no >= 0.10:
            n_trades += 1
            pnl.append((1.0 if y == 0 else 0.0) - no_ask[i])

    if not pnl:
        return {"sim_ev_per_trade": 0.0, "sim_trades": 0.0, "sim_total_ev": 0.0}
    arr = np.array(pnl, dtype=float)
    return {
        "sim_ev_per_trade": float(arr.mean()),
        "sim_trades": float(n_trades),
        "sim_total_ev": float(arr.sum()),
    }


def train(args: argparse.Namespace) -> None:
    market = fetch_coinbase_candles(args.interval, args.period)
    factors = merge_live_factors(
        market.index,
        args.interval,
        strict=args.require_live_factors,
        critical_factor_max_age_minutes=args.critical_factor_max_age_minutes,
    )
    feat = build_structural_features(market, factors, args.horizon_minutes, args.interval)
    feat["regime"] = classify_regime(feat)

    z_grid = [float(x.strip()) for x in args.threshold_z_grid.split(",") if x.strip()]
    panel = make_training_panel(feat, z_grid)

    excluded = {
        "fwd_ret_h",
        "y_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "macro_event_flag",  # keep binary regime but avoid duplicated direct label-like effect
    }
    feature_cols = [
        c for c in panel.columns if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])
    ]

    # stronger regularization + return-group dampening via scaling
    return_like = [c for c in feature_cols if c.startswith("ret_") or "trend" in c or "slope" in c]
    panel = panel.dropna(subset=feature_cols + ["y_event"]).copy()
    n = len(panel)

    # auto-adjust windows
    tr_b, te_b, st_b = args.train_bars, args.test_bars, args.step_bars
    if tr_b + te_b > n:
        tr_b = max(200, int(n * 0.6))
        te_b = max(60, int(n * 0.2))
        st_b = max(30, int(te_b * 0.5))

    cfg = WalkForwardConfig(train_bars=tr_b, test_bars=te_b, step_bars=st_b)
    splits = split_walk_forward(n, cfg)

    wf_rows = []
    fold_metrics = []

    for i, (tr_i, te_i) in enumerate(splits, start=1):
        tr = panel.iloc[tr_i]
        te = panel.iloc[te_i]

        xtr = tr[feature_cols].to_numpy(dtype=float)
        xte = te[feature_cols].to_numpy(dtype=float)
        ytr = tr["y_event"].to_numpy()
        yte = te["y_event"].to_numpy()

        scaler = StandardScaler()
        xtr_s = scaler.fit_transform(xtr)
        xte_s = scaler.transform(xte)

        # damp return feature group influence
        if return_like:
            idx = [feature_cols.index(c) for c in return_like]
            xtr_s[:, idx] *= float(args.return_group_scale)
            xte_s[:, idx] *= float(args.return_group_scale)

        model = LogisticRegression(C=float(args.C), max_iter=2000, solver="lbfgs")
        model.fit(xtr_s, ytr)

        # chrono calibrator fit on tail of train
        cut = max(int(0.8 * len(xtr_s)), 1)
        raw_cal = model.predict_proba(xtr_s[cut:])[:, 1] if len(xtr_s) > cut else model.predict_proba(xtr_s)[:, 1]
        y_cal = ytr[cut:] if len(ytr) > cut else ytr
        if len(np.unique(y_cal)) < 2 and len(np.unique(ytr)) >= 2:
            raw_cal = model.predict_proba(xtr_s)[:, 1]
            y_cal = ytr
        calibrator = fit_calibrator(raw_cal, y_cal, args.calibration_method)

        raw_te = model.predict_proba(xte_s)[:, 1]
        p_te = apply_calibrator(calibrator, raw_te)

        wf_rows.append(
            pd.DataFrame(
                {
                    "timestamp": te.index,
                    "fold": i,
                    "y_true": yte,
                    "p_model": p_te,
                    "p_raw": raw_te,
                }
            )
        )

        m = eval_metrics(yte, p_te)
        m["fold"] = i
        fold_metrics.append(m)

    oos = pd.concat(wf_rows, ignore_index=True).sort_values("timestamp")
    overall = eval_metrics(oos["y_true"].to_numpy(), oos["p_model"].to_numpy())
    ev = simulated_kalshi_ev(oos["p_model"].to_numpy(), oos["y_true"].to_numpy(), spread_bps=args.sim_spread_bps)

    # final model
    x = panel[feature_cols].to_numpy(dtype=float)
    y = panel["y_event"].to_numpy()
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    if return_like:
        idx = [feature_cols.index(c) for c in return_like]
        xs[:, idx] *= float(args.return_group_scale)

    model = LogisticRegression(C=float(args.C), max_iter=2000, solver="lbfgs")
    model.fit(xs, y)

    cut = max(int(0.8 * len(xs)), 1)
    raw_cal = model.predict_proba(xs[cut:])[:, 1] if len(xs) > cut else model.predict_proba(xs)[:, 1]
    y_cal = y[cut:] if len(y) > cut else y
    if len(np.unique(y_cal)) < 2 and len(np.unique(y)) >= 2:
        raw_cal = model.predict_proba(xs)[:, 1]
        y_cal = y
    calibrator = fit_calibrator(raw_cal, y_cal, args.calibration_method)

    bundle = {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "return_like": return_like,
        "return_group_scale": float(args.return_group_scale),
        "interval": args.interval,
        "horizon_minutes": int(args.horizon_minutes),
        "threshold_z_grid": z_grid,
        "trained_at": str(_now_utc()),
        "metrics": overall,
        "sim_ev": ev,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / args.model_name
    report_path = out_dir / f"{Path(args.model_name).stem}_train_report.json"
    oos_path = out_dir / f"{Path(args.model_name).stem}_walk_forward_predictions.csv"

    joblib.dump(bundle, model_path)
    oos.to_csv(oos_path, index=False)

    report = {
        "overall": overall,
        "simulated_kalshi_ev": ev,
        "folds": fold_metrics,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "calibration_method": args.calibration_method,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print("=== Training Summary ===")
    print(json.dumps({"metrics": overall, "sim_ev": ev}, indent=2))
    print("Saved model:", model_path)
    print("Saved report:", report_path)


def _distance_probability(norm_dist: float) -> float:
    # Dominant boundary behavior from normalized distance.
    return float(1.0 - norm.cdf(norm_dist))


def _regime_shift(prev_regime: int, curr_regime: int, prev_norm: float, curr_norm: float, prev_struct: np.ndarray, curr_struct: np.ndarray) -> bool:
    if prev_regime != curr_regime:
        return True
    if (abs(prev_norm) < 1.0 <= abs(curr_norm)) or (abs(curr_norm) < 1.0 <= abs(prev_norm)):
        return True
    if np.linalg.norm(curr_struct - prev_struct) > 2.5:
        return True
    return False


def _structural_alignment(latest: pd.Series, direction_yes: bool) -> float:
    score = 0.0
    checks = 0

    # order-book / depth
    if "depth_ratio" in latest.index:
        checks += 1
        if direction_yes and latest["depth_ratio"] > 1:
            score += 1
        if (not direction_yes) and latest["depth_ratio"] < 1:
            score += 1

    # liquidations skew
    if "liquidations_short_usd" in latest.index and "liquidations_long_usd" in latest.index:
        checks += 1
        if direction_yes and latest["liquidations_short_usd"] > latest["liquidations_long_usd"]:
            score += 1
        if (not direction_yes) and latest["liquidations_long_usd"] > latest["liquidations_short_usd"]:
            score += 1

    # funding trend
    if "funding_slope" in latest.index:
        checks += 1
        if direction_yes and latest["funding_slope"] > 0:
            score += 1
        if (not direction_yes) and latest["funding_slope"] < 0:
            score += 1

    # trade flow
    if "trade_flow_imbalance_smooth" in latest.index:
        checks += 1
        if direction_yes and latest["trade_flow_imbalance_smooth"] > 0:
            score += 1
        if (not direction_yes) and latest["trade_flow_imbalance_smooth"] < 0:
            score += 1

    if checks == 0:
        return 0.0
    return float(score / checks)


def _build_live_snapshot(
    interval: str,
    refresh_period: str,
    require_live_factors: bool,
    critical_factor_max_age_minutes: int,
    max_staleness_minutes: int,
) -> Dict[str, object]:
    market = fetch_coinbase_candles(
        interval=interval,
        period=refresh_period,
        strict_fresh_minutes=max_staleness_minutes,
    )
    factors = merge_live_factors(
        market.index,
        interval,
        strict=require_live_factors,
        critical_factor_max_age_minutes=critical_factor_max_age_minutes,
    )
    return {"market": market, "factors": factors, "interval": interval}


def _calc_contract_probability(
    bundle: Dict[str, object],
    snapshot: Dict[str, object],
    target_price: float,
    current_price: float,
    contract_key: str,
    prev_state: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    feature_cols = bundle["feature_cols"]
    return_like = bundle.get("return_like", [])
    return_scale = float(bundle.get("return_group_scale", 1.0))
    horizon_minutes = int(bundle["horizon_minutes"])
    interval = str(bundle["interval"])
    market = snapshot["market"]
    factors = snapshot["factors"]

    feat = build_structural_features(market, factors, horizon_minutes=horizon_minutes, interval=interval)
    feat["regime"] = classify_regime(feat)
    latest = feat.iloc[-1].copy()
    latest_candle_ts = pd.Timestamp(market.index.max())
    seconds_since_latest_candle = float((_now_utc() - latest_candle_ts).total_seconds())

    threshold_ret = float(np.log(float(target_price) / float(current_price)))
    expected_move = float(max(latest.get("expected_move", np.nan), EPS))
    norm_dist = float(threshold_ret / expected_move)

    latest["threshold_ret"] = threshold_ret
    latest["threshold_z"] = norm_dist
    latest["threshold_abs"] = abs(threshold_ret)
    latest["threshold_ret_sq"] = threshold_ret ** 2
    latest["threshold_z_abs"] = abs(norm_dist)

    critical_missing = []
    for c in ["funding_rate", "liquidations_long_usd", "liquidations_short_usd", "order_book_imbalance"]:
        flag = f"{c}_missing_flag"
        if flag in latest.index and float(latest.get(flag, 0.0)) > 0.5:
            critical_missing.append(c)
    if critical_missing:
        raise ValueError(f"Critical live data missing at inference: {critical_missing}")

    x = pd.DataFrame([latest]).reindex(columns=feature_cols)
    # Use only available live columns for median fill; some training-only cols
    # (e.g., threshold_* variants) are injected directly from `latest`.
    med_cols = [c for c in feature_cols if c in feat.columns]
    med = feat[med_cols].median(numeric_only=True) if med_cols else pd.Series(dtype=float)
    x = x.fillna(med).fillna(0.0)
    xs = bundle["scaler"].transform(x.to_numpy(dtype=float))
    if return_like:
        idx = [feature_cols.index(c) for c in return_like if c in feature_cols]
        xs[:, idx] *= return_scale

    raw = float(np.clip(bundle["model"].predict_proba(xs)[:, 1][0], 1e-6, 1 - 1e-6))
    calibrated = float(apply_calibrator(bundle["calibrator"], np.array([raw]))[0])

    coefs = bundle["model"].coef_[0]
    contrib = (xs[0] * coefs).astype(float)
    contrib_pairs = sorted(
        [{"feature": feature_cols[i], "contribution": float(contrib[i])} for i in range(len(feature_cols))],
        key=lambda z: abs(z["contribution"]),
        reverse=True,
    )

    p_dist = _distance_probability(norm_dist)
    weight_dist = 0.75 if abs(norm_dist) > 1.0 else 0.35
    p_struct = (1.0 - weight_dist) * calibrated + weight_dist * p_dist

    curr_regime = int(latest.get("regime", 1))
    struct_vec = np.array(
        [
            float(latest.get("order_book_imbalance", 0.0)),
            float(latest.get("trade_flow_imbalance_smooth", 0.0)),
            float(latest.get("liq_z", 0.0)),
            float(latest.get("funding_slope", 0.0)),
            float(latest.get("vol_rank", 0.5)),
        ],
        dtype=float,
    )

    prev_prob = None if prev_state is None else prev_state.get("prev_prob")
    prev_norm = None if prev_state is None else prev_state.get("prev_norm_dist")
    prev_regime = None if prev_state is None else prev_state.get("prev_regime")
    prev_struct = None
    if prev_state is not None and prev_state.get("prev_struct_vec") is not None:
        prev_struct = np.array(prev_state.get("prev_struct_vec"), dtype=float)

    regime_shift = False
    if prev_prob is not None and prev_struct is not None and prev_norm is not None and prev_regime is not None:
        regime_shift = _regime_shift(int(prev_regime), curr_regime, float(prev_norm), norm_dist, prev_struct, struct_vec)

    if prev_prob is None or regime_shift:
        final_prob = p_struct
    else:
        final_prob = 0.7 * float(prev_prob) + 0.3 * p_struct
    final_prob = float(np.clip(final_prob, 1e-6, 1 - 1e-6))
    prob_jump = None if prev_prob is None else float(abs(final_prob - float(prev_prob)))

    expiry = _now_utc() + pd.Timedelta(minutes=horizon_minutes)
    next_state = {
        "prev_prob": final_prob,
        "prev_norm_dist": norm_dist,
        "prev_regime": curr_regime,
        "prev_struct_vec": struct_vec.tolist(),
        "updated_at": str(_now_utc()),
    }

    result = {
        "contract_key": contract_key,
        "symbol": "BTC-USD",
        "interval": interval,
        "horizon_minutes": horizon_minutes,
        "prediction_timestamp": str(_now_utc()),
        "expiry_timestamp": str(expiry),
        "current_price": float(current_price),
        "target_price": float(target_price),
        "threshold_return": threshold_ret,
        "normalized_distance": norm_dist,
        "probability_above_target": final_prob,
        "probability_below_or_equal_target": float(1.0 - final_prob),
        "debug": {
            "raw_model_prob": raw,
            "calibrated_prob": calibrated,
            "distance_prob": p_dist,
            "distance_weight": weight_dist,
            "post_structure_prob": p_struct,
            "inertia_applied": bool(prev_prob is not None and not regime_shift),
            "regime_shift": regime_shift,
            "prob_jump": prob_jump,
            "regime": curr_regime,
            "critical_missing": critical_missing,
            "latest_candle_timestamp_utc": str(latest_candle_ts),
            "seconds_since_latest_candle": seconds_since_latest_candle,
            "live_factor_quality": factors.attrs.get("live_quality", {}),
            "top_feature_contributions": contrib_pairs[:10],
            "structural_snapshot": {
                "depth_ratio": float(latest.get("depth_ratio", np.nan)),
                "trade_flow_imbalance_smooth": float(latest.get("trade_flow_imbalance_smooth", np.nan)),
                "liq_z": float(latest.get("liq_z", np.nan)),
                "funding_slope": float(latest.get("funding_slope", np.nan)),
                "vol_rank": float(latest.get("vol_rank", np.nan)),
                "oi_missing_flag": float(latest.get("open_interest_missing_flag", latest.get("oi_missing_flag", 0.0))),
            },
        },
    }
    return result, next_state


def _discover_models(model_path: str) -> Dict[int, Dict[str, object]]:
    path = Path(model_path)
    if path.is_file():
        b = joblib.load(path)
        return {int(b["horizon_minutes"]): {"path": str(path), "bundle": b}}
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Model path not found: {model_path}")
    out: Dict[int, Dict[str, object]] = {}
    for fp in sorted(path.glob("*.joblib")):
        try:
            b = joblib.load(fp)
        except Exception:
            continue
        if not isinstance(b, dict) or "horizon_minutes" not in b:
            continue
        h = int(b["horizon_minutes"])
        if h not in out:
            out[h] = {"path": str(fp), "bundle": b}
    if not out:
        raise ValueError(f"No valid model bundles found in {model_path}")
    return out


def _discover_model_catalog(model_path: str) -> List[Dict[str, object]]:
    path = Path(model_path)
    if path.is_file():
        b = joblib.load(path)
        if not isinstance(b, dict) or "horizon_minutes" not in b or "interval" not in b:
            raise ValueError(f"Invalid model bundle: {model_path}")
        return [
            {
                "path": str(path),
                "bundle": b,
                "horizon_minutes": int(b["horizon_minutes"]),
                "interval": str(b["interval"]),
            }
        ]
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Model path not found: {model_path}")
    out: List[Dict[str, object]] = []
    for fp in sorted(path.glob("*.joblib")):
        try:
            b = joblib.load(fp)
        except Exception:
            continue
        if not isinstance(b, dict) or "horizon_minutes" not in b or "interval" not in b:
            continue
        out.append(
            {
                "path": str(fp),
                "bundle": b,
                "horizon_minutes": int(b["horizon_minutes"]),
                "interval": str(b["interval"]),
            }
        )
    if not out:
        raise ValueError(f"No valid model bundles found in {model_path}")
    return out


def _suggest_interval_from_minutes(minutes_left: int) -> str:
    m = int(minutes_left)
    if m <= 15:
        return "1m"
    if m <= 60:
        return "5m"
    if m <= 240:
        return "15m"
    return "1h"


def _pick_model_for_manual(catalog: List[Dict[str, object]], minutes_left: int, preferred_interval: Optional[str]) -> Tuple[Dict[str, object], str]:
    note = ""
    if preferred_interval:
        same_iv = [r for r in catalog if str(r["interval"]).lower() == str(preferred_interval).lower()]
    else:
        same_iv = []
    if same_iv:
        pick = min(same_iv, key=lambda r: abs(int(r["horizon_minutes"]) - int(minutes_left)))
        return pick, note

    # If preferred interval not available, fallback to nearest horizon globally.
    pick = min(catalog, key=lambda r: abs(int(r["horizon_minutes"]) - int(minutes_left)))
    if preferred_interval:
        note = (
            f"Preferred interval '{preferred_interval}' not found; "
            f"using interval '{pick['interval']}' horizon={pick['horizon_minutes']}m."
        )
    return pick, note


def _nearest_model_for_horizon(models: Dict[int, Dict[str, object]], horizon_minutes: float, max_gap_minutes: int) -> Optional[Dict[str, object]]:
    hs = sorted(models.keys())
    if not hs:
        return None
    best = min(hs, key=lambda h: abs(h - horizon_minutes))
    if abs(best - horizon_minutes) > float(max_gap_minutes):
        return None
    return models[best]


def predict_once(args: argparse.Namespace) -> Dict[str, object]:
    bundle = joblib.load(args.model_path)
    interval = str(bundle["interval"])
    snapshot = _build_live_snapshot(
        interval=interval,
        refresh_period=args.refresh_period or "30d",
        require_live_factors=args.require_live_factors,
        critical_factor_max_age_minutes=args.critical_factor_max_age_minutes,
        max_staleness_minutes=args.max_staleness_minutes,
    )
    current_price = float(args.current_price) if args.current_price is not None else fetch_coinbase_spot()
    target_price = float(args.target_price)

    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    prev_state = None
    if state_path.exists():
        prev_state = json.loads(state_path.read_text())
    result, next_state = _calc_contract_probability(
        bundle=bundle,
        snapshot=snapshot,
        target_price=target_price,
        current_price=current_price,
        contract_key="single_predict",
        prev_state=prev_state,
    )
    state_path.write_text(json.dumps(next_state, indent=2))
    return result


def _kalshi_base_candidates() -> List[str]:
    env_base = os.getenv("KALSHI_API_BASE", "").strip()
    cands = [
        env_base,
        "https://api.kalshi.com/trade-api/v2",
        "https://api.elections.kalshi.com/trade-api/v2",
        "https://trading-api.kalshi.com/trade-api/v2",
    ]
    out = []
    seen = set()
    for b in cands:
        if not b:
            continue
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def _kalshi_headers() -> Dict[str, str]:
    key = os.getenv("KALSHI_API_KEY", "").strip()
    token = os.getenv("KALSHI_API_TOKEN", "").strip()
    headers = {"accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if key:
        # Keep bearer for compatibility with existing setup, but also include
        # Kalshi key header for accounts that require key-based auth.
        headers["Authorization"] = f"Bearer {key}"
        headers["KALSHI-ACCESS-KEY"] = key
    return headers


def _normalize_price_prob(v: object) -> Optional[float]:
    if v is None:
        return None
    x = float(v)
    if x > 1.0:
        x /= 100.0
    return float(np.clip(x, 0.0, 1.0))


def _extract_kalshi_market_fields(data: Dict[str, object], fallback_ticker: Optional[str] = None) -> Dict[str, object]:
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
    status = data.get("status") or data.get("market_status") or data.get("marketStatus")

    yes_ask = _normalize_price_prob(yes_ask_raw)
    yes_bid = _normalize_price_prob(yes_bid_raw)
    no_ask = _normalize_price_prob(no_ask_raw)
    no_bid = _normalize_price_prob(no_bid_raw)
    yes_last = _normalize_price_prob(yes_last_raw)
    no_last = _normalize_price_prob(no_last_raw)

    # Prefer executable ask-side pricing; fallback to mid; then last; then bid.
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
        "status": status,
        "raw": data,
    }


def fetch_kalshi_market(market_ticker: str) -> Dict[str, object]:
    headers = _kalshi_headers()
    last_err = None
    for base in _kalshi_base_candidates():
        for h in (headers, {"accept": "application/json"}):
            try:
                data = _safe_get_json(f"{base}/markets/{market_ticker}", headers=h)
                if not isinstance(data, dict):
                    continue
                m = _extract_kalshi_market_fields(data, fallback_ticker=market_ticker)
                m["source_base"] = base
                return m
            except Exception as e:
                last_err = e
                _kalshi_note_error(f"fetch_kalshi_market {market_ticker} via {base}: {e}")
                continue
    raise ValueError(f"Kalshi market fetch failed for {market_ticker}: {last_err}")


def fetch_kalshi_all_open_markets() -> List[Dict[str, object]]:
    headers = _kalshi_headers()
    out_rows: List[Dict[str, object]] = []
    page_cap = max(1, int(os.getenv("KALSHI_MAX_PAGES", "3")))
    for base in _kalshi_base_candidates():
        endpoints = [
            (f"{base}/markets", {"status": "open", "limit": "1000"}),
            (f"{base}/markets", {"status": "active", "limit": "1000"}),
            (f"{base}/markets", {"limit": "1000"}),
        ]
        for url, params in endpoints:
            for h in (headers, {"accept": "application/json"}):
                cursor = None
                for _ in range(page_cap):
                    q = dict(params)
                    if cursor:
                        q["cursor"] = str(cursor)
                    try:
                        payload = _safe_get_json(url, params=q, headers=h)
                    except Exception:
                        _kalshi_note_error(f"fetch_kalshi_all_open_markets {url} params={q} failed")
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
                        try:
                            parsed = _extract_kalshi_market_fields(rr)
                            if parsed.get("yes_prob") is None or parsed.get("no_prob") is None:
                                tkr = parsed.get("market_ticker")
                                if tkr:
                                    parsed = fetch_kalshi_market(str(tkr))
                            parsed["source_base"] = base
                            out_rows.append(parsed)
                        except Exception:
                            tkr = rr.get("ticker") or rr.get("market_ticker")
                            if tkr:
                                try:
                                    mm = fetch_kalshi_market(str(tkr))
                                    mm["source_base"] = base
                                    out_rows.append(mm)
                                except Exception:
                                    pass
                if not cursor:
                    break
            if out_rows:
                break
        if out_rows:
            break

    uniq: Dict[str, Dict[str, object]] = {}
    for m in out_rows:
        t = m.get("market_ticker")
        if t:
            uniq[str(t)] = m
    return list(uniq.values())


def _fetch_kalshi_markets_with_params(params: Dict[str, str]) -> List[Dict[str, object]]:
    headers = _kalshi_headers()
    out_rows: List[Dict[str, object]] = []
    page_cap = max(1, int(os.getenv("KALSHI_TARGETED_MAX_PAGES", "2")))
    for base in _kalshi_base_candidates():
        url = f"{base}/markets"
        for h in (headers, {"accept": "application/json"}):
            cursor = None
            for _ in range(page_cap):
                q = dict(params)
                if cursor:
                    q["cursor"] = str(cursor)
                try:
                    payload = _safe_get_json(url, params=q, headers=h)
                except Exception:
                    _kalshi_note_error(f"_fetch_kalshi_markets_with_params {url} params={q} failed")
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
                    try:
                        p = _extract_kalshi_market_fields(rr)
                        p["source_base"] = base
                        out_rows.append(p)
                    except Exception:
                        tkr = rr.get("ticker") or rr.get("market_ticker")
                        if tkr:
                            try:
                                mm = fetch_kalshi_market(str(tkr))
                                mm["source_base"] = base
                                out_rows.append(mm)
                            except Exception:
                                pass
                if not cursor:
                    break
    uniq: Dict[str, Dict[str, object]] = {}
    for m in out_rows:
        t = m.get("market_ticker")
        if t:
            uniq[str(t)] = m
    return list(uniq.values())


def _manual_kalshi_btc_tickers() -> List[str]:
    raw = os.getenv("KALSHI_BTC_TICKERS", "")
    toks = re.split(r"[,\s;]+", raw)
    return [t.strip() for t in toks if t and t.strip()]


def _kalshi_btc_series_candidates() -> List[str]:
    raw = os.getenv("KALSHI_BTC_SERIES", "KXBTCD,KXBTC,KXBT")
    toks = re.split(r"[,\s;]+", raw)
    out: List[str] = []
    seen = set()
    for t in toks:
        tt = t.strip().upper()
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


def fetch_kalshi_btc_markets(fast_mode: bool = True) -> List[Dict[str, object]]:
    # Manual override: direct list of market tickers from Kalshi UI/API.
    manual = _manual_kalshi_btc_tickers()
    manual_rows: List[Dict[str, object]] = []
    for t in manual:
        try:
            mm = fetch_kalshi_market(t)
            mm["source"] = "manual_ticker_override"
            manual_rows.append(mm)
        except Exception:
            pass

    markets: List[Dict[str, object]] = []
    targeted_queries = [
        {"status": "open", "limit": "1000", "search": "bitcoin"},
        {"status": "open", "limit": "1000", "search": "btc"},
        {"status": "active", "limit": "1000", "search": "bitcoin"},
        {"status": "active", "limit": "1000", "search": "btc"},
        {"limit": "1000", "search": "bitcoin"},
        {"limit": "1000", "search": "btc"},
    ]
    for series in _kalshi_btc_series_candidates():
        targeted_queries.extend(
            [
                {"status": "open", "limit": "1000", "series_ticker": series},
                {"status": "active", "limit": "1000", "series_ticker": series},
                {"limit": "1000", "series_ticker": series},
            ]
        )
    for q in targeted_queries:
        try:
            rows = _fetch_kalshi_markets_with_params(q)
            if rows:
                markets.extend(rows)
        except Exception:
            pass
    if (not markets) or (not fast_mode):
        # Broad scan fallback (slower).
        try:
            markets.extend(fetch_kalshi_all_open_markets())
        except Exception:
            pass

    keywords = ["BTC", "BITCOIN", "XBT", "KXBTC", "KXBT", "KXBTCD", "CRYPTO", "BTCUSD", "XBTUSD"]
    title_regex = os.getenv(
        "KALSHI_BTC_TITLE_REGEX",
        r"BITCOIN\s+PRICE\s+TODAY\s+AT\s+\d{1,2}\s*(AM|PM)\s*(ET|EST)?",
    ).strip()
    ticker_regex = os.getenv("KALSHI_BTC_TICKER_REGEX", r"^(KXBT|KXBTC|KXBTCD)").strip()
    sport_noise = [
        "NBA",
        "NHL",
        "NFL",
        "MLB",
        "NCAA",
        "SOCCER",
        "TENNIS",
        "SABRES",
        "LAKERS",
        "LEBRON",
        "DONCIC",
        "TOURNAMENT",
        "GOALS",
        "POINTS",
        "REBOUNDS",
    ]
    out: List[Dict[str, object]] = []
    for m in markets:
        raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
        title = str(raw.get("title", "")).upper()
        subtitle = str(raw.get("subtitle", "")).upper()
        event_ticker = str(raw.get("event_ticker", "") or raw.get("eventTicker", "")).upper()
        series_ticker = str(raw.get("series_ticker", "") or raw.get("seriesTicker", "")).upper()
        market_ticker = str(m.get("market_ticker", "")).upper()
        txt = " ".join(
            str(x)
            for x in [
                market_ticker,
                str(raw.get("ticker", "")).upper(),
                str(raw.get("market_ticker", "")).upper(),
                title,
                subtitle,
                event_ticker,
                series_ticker,
                raw.get("underlying", ""),
                raw.get("underlying_asset", ""),
                raw.get("category", ""),
            ]
        ).upper()
        if any(s in txt for s in sport_noise):
            continue
        # Exact hourly BTC title pattern first (user-requested market style).
        if title_regex:
            try:
                if re.search(title_regex, title):
                    out.append(m)
                    continue
            except re.error:
                pass
        # Keep only contracts likely related to BTC/bitcoin.
        if any(k in txt for k in keywords):
            out.append(m)
            continue
        # Additional strict ticker fallback for known Kalshi BTC contract prefixes.
        if ticker_regex:
            try:
                if re.search(ticker_regex, market_ticker):
                    out.append(m)
                    continue
            except re.error:
                pass
        if market_ticker.startswith("KXBT") or market_ticker.startswith("KXBTC"):
            out.append(m)
    uniq: Dict[str, Dict[str, object]] = {}
    for m in [*manual_rows, *out]:
        t = m.get("market_ticker")
        if t:
            uniq[str(t)] = m
    return list(uniq.values())


def _extract_strike_from_market(m: Dict[str, object]) -> Optional[float]:
    s = m.get("strike")
    if isinstance(s, (int, float)):
        return float(s)
    raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
    for key in ["strike_price", "strikePrice", "floor_strike", "floorStrike", "cap_strike", "capStrike"]:
        v = raw.get(key)
        if isinstance(v, (int, float)):
            fv = float(v)
            # Some feeds store strike in cents.
            if fv > 300000:
                fv /= 100.0
            if 5000 <= fv <= 300000:
                return fv
    txt = " ".join(str(m.get(k, "")) for k in ["market_ticker", "raw", "strike"])
    mm = re.findall(r"(?<!\d)(\d{4,7})(?!\d)", txt)
    if not mm:
        return None
    vals = [float(x) for x in mm]
    # choose value in plausible BTC strike range
    plausible = [v for v in vals if 5000 <= v <= 300000]
    if plausible:
        return float(plausible[-1])
    return float(vals[-1])


def _canonicalize_strike_value(v: float) -> float:
    x = float(v)
    frac = x - math.floor(x)
    # Kalshi tickers often encode threshold as 66499.99 for 66,500.
    if abs(frac - 0.99) <= 0.011:
        return float(round(x + 0.01, 2))
    if abs(frac - 0.01) <= 0.011:
        return float(round(x - 0.01, 2))
    return float(round(x, 2))


def _is_supported_threshold_market(market: Dict[str, object]) -> bool:
    raw = market.get("raw", {}) if isinstance(market.get("raw"), dict) else {}
    title = str(raw.get("title", "")).upper()
    subtitle = str(raw.get("subtitle", "")).upper()
    txt = " ".join(
        str(x).upper()
        for x in [
            market.get("market_ticker", ""),
            title,
            subtitle,
            raw.get("event_ticker", ""),
            raw.get("eventTicker", ""),
        ]
    )
    # Reject range contracts; model is built for threshold events.
    if any(k in txt for k in ["RANGE", "BETWEEN", " TO ", "-TO-", "-B", " BRACKET", " BAND "]):
        return False
    # Keep threshold style contracts.
    if ("OR ABOVE" in txt) or ("OR BELOW" in txt) or ("ABOVE" in txt) or ("BELOW" in txt):
        return True
    # T-strike tickers are usually threshold ladders.
    if re.search(r"-T\d{4,8}(\.\d+)?$", str(market.get("market_ticker", "")).upper()):
        return True
    return False


def _is_directional_up_market(m: Dict[str, object]) -> bool:
    raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
    txt = " ".join(
        str(x).upper()
        for x in [
            m.get("market_ticker", ""),
            raw.get("title", ""),
            raw.get("subtitle", ""),
            raw.get("event_ticker", ""),
            raw.get("eventTicker", ""),
        ]
    )
    return ("BTC PRICE UP IN NEXT" in txt) or ("KXBTC15M" in str(m.get("market_ticker", "")).upper())


def _extract_expiry_minutes(m: Dict[str, object], now: pd.Timestamp) -> Optional[float]:
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
    return float((ts - now).total_seconds() / 60.0)

def _parse_spread_fraction(v: object) -> Optional[float]:
    if v is None:
        return None
    x = float(v)
    if x > 1.0:
        x /= 100.0
    return float(np.clip(x, 0.0, 1.0))


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


def _select_dynamic_interval(intervals: Sequence[str], minutes_left: float) -> str:
    mins = sorted(((_interval_minutes(iv), iv) for iv in intervals), key=lambda x: x[0])
    if not mins:
        raise ValueError("No model intervals discovered")
    if minutes_left > 50:
        return mins[-1][1]
    if minutes_left >= 30:
        five = [iv for m, iv in mins if m == 5]
        if five:
            return five[0]
        return min(mins, key=lambda x: abs(x[0] - 5))[1]
    return mins[0][1]


def _pick_model_for_cycle(catalog: List[Dict[str, object]], minutes_left: float) -> Dict[str, object]:
    desired_iv = _select_dynamic_interval([str(r["interval"]) for r in catalog], minutes_left=minutes_left)
    scoped = [r for r in catalog if str(r["interval"]).lower() == str(desired_iv).lower()]
    if not scoped:
        scoped = catalog
    return min(scoped, key=lambda r: abs(float(r["horizon_minutes"]) - float(minutes_left)))


def _update_spot_samples(samples: List[Tuple[pd.Timestamp, float]], ts: pd.Timestamp, spot: float, max_minutes: int = 180) -> List[Tuple[pd.Timestamp, float]]:
    out = [*samples, (pd.Timestamp(ts), float(spot))]
    cutoff = pd.Timestamp(ts) - pd.Timedelta(minutes=max_minutes)
    out = [(t, p) for t, p in out if t >= cutoff]
    if len(out) > 2000:
        out = out[-2000:]
    return out


def _rolling_realized_vol_1h(samples: List[Tuple[pd.Timestamp, float]], now: pd.Timestamp) -> Optional[float]:
    cutoff = now - pd.Timedelta(minutes=60)
    recent = [(t, p) for t, p in samples if t >= cutoff and p > 0]
    if len(recent) < 5:
        return None
    prices = np.array([p for _, p in sorted(recent, key=lambda x: x[0])], dtype=float)
    rets = np.diff(np.log(prices))
    if len(rets) < 3:
        return None
    sigma = float(np.sqrt(np.sum(np.square(rets))))
    return float(max(0.0, sigma))


def _expected_move_from_rv(spot: float, minutes_left: float, rv_1h: Optional[float], fallback_sigma_1h: float) -> float:
    sigma_1h = float(rv_1h if rv_1h is not None else fallback_sigma_1h)
    horizon_scale = math.sqrt(max(float(minutes_left), 1.0) / 60.0)
    return float(max(spot * sigma_1h * horizon_scale, 10.0))


def _parse_hourly_title_key(title: str) -> Optional[str]:
    t = str(title).strip().upper()
    m = re.search(r"BITCOIN\s+PRICE\s+(TODAY\s+)?AT\s+(\d{1,2})\s*(AM|PM)\s*(ET|EST)?\??", t)
    if not m:
        m = re.search(r"\b(\d{1,2})\s*(AM|PM)\s*(ET|EST)\b", t)
        if not m:
            return None
        return f"{m.group(1)}{m.group(2)}"
    # Group 2/3 when "today" is optional in regex above.
    return f"{m.group(2)}{m.group(3)}"


def _select_active_hourly_group(markets: List[Dict[str, object]], now: pd.Timestamp, min_expiry_min: float, max_expiry_min: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    fallback_rows: List[Dict[str, object]] = []
    for m in markets:
        raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
        title = str(raw.get("title", "")).strip()
        title_key = _parse_hourly_title_key(title)
        exp_min = _extract_expiry_minutes(m, now=now)
        if title_key is None or exp_min is None:
            continue
        if exp_min < float(min_expiry_min) or exp_min > float(max_expiry_min):
            continue
        strike = _extract_strike_from_market(m)
        if strike is None:
            continue
        mm = dict(m)
        mm["expiry_minutes"] = float(exp_min)
        mm["title_key"] = title_key
        mm["strike_value"] = _canonicalize_strike_value(float(strike))
        txt = " ".join(
            str(x).upper()
            for x in [
                m.get("market_ticker", ""),
                raw.get("title", ""),
                raw.get("subtitle", ""),
                raw.get("event_ticker", ""),
                raw.get("eventTicker", ""),
                raw.get("series_ticker", ""),
                raw.get("seriesTicker", ""),
            ]
        )
        if ("BTC" in txt) or ("BITCOIN" in txt) or str(m.get("market_ticker", "")).upper().startswith(("KXBT", "KXBTC")):
            fallback_rows.append(mm)
        if title_key is not None:
            rows.append(mm)

    base_rows = rows if rows else fallback_rows
    if not base_rows:
        return []
    active_expiry = min(float(r["expiry_minutes"]) for r in base_rows)
    # Keep only the current hourly event (same expiry bucket).
    return [r for r in base_rows if abs(float(r["expiry_minutes"]) - active_expiry) <= 2.5]


def _select_active_btc_strike_group(markets: List[Dict[str, object]], now: pd.Timestamp, min_expiry_min: float, max_expiry_min: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for m in markets:
        raw = m.get("raw", {}) if isinstance(m.get("raw"), dict) else {}
        txt = " ".join(
            str(x).upper()
            for x in [
                m.get("market_ticker", ""),
                raw.get("title", ""),
                raw.get("subtitle", ""),
                raw.get("event_ticker", ""),
                raw.get("eventTicker", ""),
                raw.get("series_ticker", ""),
                raw.get("seriesTicker", ""),
                raw.get("underlying", ""),
                raw.get("underlying_asset", ""),
            ]
        )
        if ("BTC" not in txt) and ("BITCOIN" not in txt) and (not str(m.get("market_ticker", "")).upper().startswith(("KXBT", "KXBTC"))):
            continue
        exp_min = _extract_expiry_minutes(m, now=now)
        if exp_min is None or exp_min < float(min_expiry_min) or exp_min > float(max_expiry_min):
            continue
        strike = _extract_strike_from_market(m)
        if strike is None:
            continue
        mm = dict(m)
        mm["expiry_minutes"] = float(exp_min)
        mm["strike_value"] = _canonicalize_strike_value(float(strike))
        rows.append(mm)

    if not rows:
        return []
    active_expiry = min(float(r["expiry_minutes"]) for r in rows)
    return [r for r in rows if abs(float(r["expiry_minutes"]) - active_expiry) <= 3.0]


def _append_runner_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    cols = list(row.keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _build_discord_alert_content(payload: Dict[str, object]) -> str:
    return (
        f"**BTC Mispricing Alert**\n"
        f"Market: `{payload.get('market_title')}`\n"
        f"Ticker: `{payload.get('market_ticker')}`\n"
        f"Strike: `{payload.get('strike')}`  Side: **{payload.get('side')}**\n"
        f"Model P(YES): **{float(payload.get('model_probability', 0.0))*100:.2f}%**\n"
        f"Market P(YES): **{float(payload.get('market_probability', 0.0))*100:.2f}%**\n"
        f"Edge: **{float(payload.get('edge', 0.0))*100:+.2f}%**\n"
        f"Confidence: `{payload.get('confidence')}` ({float(payload.get('confidence_score', 0.0)):.4f})\n"
        f"Time Remaining: `{float(payload.get('minutes_left', 0.0)):.1f}m`\n"
        f"BTC Spot: `${float(payload.get('spot_price', 0.0)):.2f}`"
    )


def _send_discord_alert(webhook: str, payload: Dict[str, object]) -> None:
    body = {"content": _build_discord_alert_content(payload)}
    _safe_post_json(webhook, body)


def _pct(v: object) -> str:
    try:
        x = float(v)
    except Exception:
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x*100:.1f}%"


def _stoploss_reason_lines(
    *,
    move_against: float,
    drop_threshold: float,
    curr_side_prob: float,
    abs_threshold: float,
    mins_left: Optional[float],
    late_minutes: float,
    late_prob: float,
    model_side_prob: Optional[float],
    model_threshold: float,
    entry_market_prob: float,
) -> List[str]:
    lines: List[str] = []
    if move_against >= drop_threshold:
        lines.append(
            f"Chance moved against your side by {_pct(move_against)} "
            f"(entry {_pct(entry_market_prob)} -> now {_pct(curr_side_prob)})."
        )
    if curr_side_prob <= abs_threshold:
        lines.append(
            f"Current chance for your side is {_pct(curr_side_prob)}, below safety line {_pct(abs_threshold)}."
        )
    if mins_left is not None and mins_left <= late_minutes and curr_side_prob <= late_prob:
        lines.append(
            f"Only {mins_left:.1f}m left and chance is {_pct(curr_side_prob)}, below late-stage line {_pct(late_prob)}."
        )
    if model_side_prob is not None and model_side_prob <= model_threshold:
        lines.append(
            f"Model now shows {_pct(model_side_prob)} for your side, below model safety line {_pct(model_threshold)}."
        )
    return lines


def _build_discord_stoploss_content(payload: Dict[str, object]) -> str:
    why = payload.get("why_lines")
    if isinstance(why, list) and why:
        why_text = "\n".join(f"- {str(x)}" for x in why)
    else:
        why_text = f"- {payload.get('reason')}"
    return (
        f"**BTC Risk Alert (Stop-Loss Check)**\n"
        f"Market: `{payload.get('market_title')}`\n"
        f"Ticker: `{payload.get('market_ticker')}`\n"
        f"Strike: `{payload.get('strike')}`  Side: **{payload.get('side')}**\n"
        f"Entry Market P: **{float(payload.get('entry_market_probability', 0.0))*100:.2f}%**\n"
        f"Current Market P: **{float(payload.get('current_market_probability', 0.0))*100:.2f}%**\n"
        f"Move Against: **{float(payload.get('move_against', 0.0))*100:.2f}%**\n"
        f"Time Remaining: `{float(payload.get('minutes_left', 0.0)):.1f}m`\n"
        f"BTC Spot: `${float(payload.get('spot_price', 0.0)):.2f}`\n"
        f"Why stop-loss fired:\n{why_text}"
    )


def _send_discord_stoploss_alert(webhook: str, payload: Dict[str, object]) -> None:
    body = {"content": _build_discord_stoploss_content(payload)}
    _safe_post_json(webhook, body)


def _build_discord_position_update_content(payload: Dict[str, object]) -> str:
    return (
        f"**BTC Position Update**\n"
        f"Market: `{payload.get('market_title')}`\n"
        f"Ticker: `{payload.get('market_ticker')}`\n"
        f"Strike: `{payload.get('strike')}`  Side: **{payload.get('side')}**\n"
        f"Model P(YES): **{float(payload.get('model_probability', 0.0))*100:.2f}%**\n"
        f"Market P(YES): **{float(payload.get('market_yes_probability', 0.0))*100:.2f}%**\n"
        f"Model P(side): **{float(payload.get('model_side_probability', 0.0))*100:.2f}%**\n"
        f"Market P(side): **{float(payload.get('market_side_probability', 0.0))*100:.2f}%**\n"
        f"Edge(side): **{float(payload.get('edge_side', 0.0))*100:+.2f}%**\n"
        f"Time Remaining: `{float(payload.get('minutes_left', 0.0)):.1f}m`\n"
        f"BTC Spot: `${float(payload.get('spot_price', 0.0)):.2f}`"
    )


def _send_discord_position_update_alert(webhook: str, payload: Dict[str, object]) -> None:
    body = {"content": _build_discord_position_update_content(payload)}
    _safe_post_json(webhook, body)


def _confidence_rank(conf: object) -> int:
    s = str(conf or "").strip().lower()
    if s == "high":
        return 3
    if s == "medium":
        return 2
    if s == "low":
        return 1
    return 0


def _confidence_threshold_rank(name: str) -> int:
    s = str(name or "").strip().lower()
    if s not in {"low", "medium", "high"}:
        return 2
    return _confidence_rank(s)


def evaluate_signal(pred: Dict[str, object], kalshi: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    p = float(pred["probability_above_target"])
    yes = float(kalshi["yes_prob"])
    no = float(kalshi["no_prob"])
    edge_yes = p - yes
    edge_no = (1.0 - p) - no
    nd = float(pred["normalized_distance"])
    dbg = pred.get("debug", {}) if isinstance(pred.get("debug"), dict) else {}

    critical_missing = dbg.get("critical_missing", [])
    if critical_missing:
        return {"signal": False, "reason": f"critical missing data: {critical_missing}"}
    live_q = dbg.get("live_factor_quality", {}) if isinstance(dbg.get("live_factor_quality"), dict) else {}
    cov = live_q.get("recent_coverage_ratio", {}) if isinstance(live_q.get("recent_coverage_ratio"), dict) else {}
    liq_cov = float(
        min(
            float(cov.get("liquidations_long_usd", 0.0)),
            float(cov.get("liquidations_short_usd", 0.0)),
        )
    )
    funding_cov = float(cov.get("funding_rate", 0.0))
    min_liq_cov = float(getattr(args, "min_liquidations_coverage", 0.10))
    min_funding_cov = float(getattr(args, "min_funding_coverage", 0.10))
    if liq_cov < min_liq_cov:
        return {"signal": False, "reason": f"liquidations coverage too low ({liq_cov:.2f})"}
    if funding_cov < min_funding_cov:
        return {"signal": False, "reason": f"funding coverage too low ({funding_cov:.2f})"}

    prob_jump = dbg.get("prob_jump")
    if prob_jump is not None and float(prob_jump) > float(args.max_prob_jump):
        return {"signal": False, "reason": f"probability jump too large ({float(prob_jump):.4f})"}

    raw_p = dbg.get("raw_model_prob")
    cal_p = dbg.get("calibrated_prob")
    if raw_p is not None and cal_p is not None and abs(float(raw_p) - float(cal_p)) > float(args.max_calibration_delta):
        return {"signal": False, "reason": "calibration instability"}

    liq = float(kalshi.get("liquidity", 0.0))
    spread = kalshi.get("spread")
    spread_ok = True
    if spread is not None:
        s = float(spread)
        if s > 1.0:
            s /= 100.0
        spread_ok = s <= float(args.max_spread)

    liq_ok = liq >= float(args.min_kalshi_liquidity)

    direction = None
    ev = None
    # Policy: trade on model conviction + positive edge, without Kalshi-probability cutoffs.
    if p >= float(args.yes_prob_min) and edge_yes > float(args.min_edge):
        direction = "YES"
        ev = edge_yes
    if p <= float(args.no_prob_max) and edge_no > float(args.min_edge):
        direction = "NO"
        ev = edge_no

    if direction is None:
        return {"signal": False, "reason": "edge/probability thresholds not met"}

    if abs(nd) >= 1.0:
        return {"signal": False, "reason": "normalized_distance too extreme"}

    if not spread_ok:
        return {"signal": False, "reason": "spread too wide"}
    if not liq_ok:
        return {"signal": False, "reason": "insufficient liquidity"}

    align = _structural_alignment(pd.Series(dbg.get("structural_snapshot", {})), direction_yes=(direction == "YES"))
    edge = edge_yes if direction == "YES" else edge_no
    confidence_score = float(edge * align)
    if confidence_score >= 0.08:
        conf = "High"
    elif confidence_score >= 0.04:
        conf = "Medium"
    else:
        conf = "Low"

    return {
        "signal": True,
        "direction": direction,
        "edge": float(edge),
        "expected_value": float(ev if ev is not None else edge),
        "alignment": float(align),
        "confidence_score": confidence_score,
        "confidence": conf,
    }


def alert(payload: Dict[str, object], args: argparse.Namespace) -> None:
    print("\n=== EDGE DETECTED ===")
    print(json.dumps(payload, indent=2))
    if args.alert_webhook:
        try:
            _safe_post_json(args.alert_webhook, payload)
        except Exception as e:
            print(f"[warn] webhook failed: {e}")


def _monitor_open_positions(
    state: RunnerState,
    now: pd.Timestamp,
    markets: List[Dict[str, object]],
    spot: float,
    catalog: List[Dict[str, object]],
    args: argparse.Namespace,
    webhook: str,
) -> None:
    by_ticker = {str(m.get("market_ticker", "")): m for m in markets if m.get("market_ticker")}
    remove_keys: List[str] = []
    snapshot_by_interval: Dict[str, Dict[str, object]] = {}
    for pos_key, pos in list(state.open_positions.items()):
        ticker = str(pos.get("market_ticker", ""))
        side = str(pos.get("side", "YES")).upper()
        m = by_ticker.get(ticker)
        if m is None:
            continue
        mins_left = _extract_expiry_minutes(m, now=now)
        if mins_left is not None and mins_left <= 0:
            remove_keys.append(pos_key)
            continue

        curr_yes = float(m.get("yes_prob", np.nan))
        if not np.isfinite(curr_yes):
            continue
        entry_market_prob = float(pos.get("entry_market_probability", np.nan))
        if not np.isfinite(entry_market_prob):
            continue

        curr_side_prob = curr_yes if side == "YES" else float(np.clip(1.0 - curr_yes, 0.0, 1.0))
        move_against = float(entry_market_prob - curr_side_prob)

        model_side_prob = None
        model_yes_prob = None
        try:
            if mins_left is not None and mins_left > 0:
                bundle_item = _pick_model_for_cycle(catalog, minutes_left=float(mins_left))
                bundle = bundle_item["bundle"]
                interval = str(bundle["interval"])
                if interval not in snapshot_by_interval:
                    snapshot_by_interval[interval] = _build_live_snapshot(
                        interval=interval,
                        refresh_period=args.refresh_period,
                        require_live_factors=args.require_live_factors,
                        critical_factor_max_age_minutes=args.critical_factor_max_age_minutes,
                        max_staleness_minutes=args.max_staleness_minutes,
                    )
                yes_is_above = _infer_yes_is_above(m)
                strike = float(pos.get("strike", np.nan))
                monitor_key = f"{pos_key}:monitor"
                pred, next_state = _calc_contract_probability(
                    bundle=bundle,
                    snapshot=snapshot_by_interval[interval],
                    target_price=float(strike),
                    current_price=float(spot),
                    contract_key=monitor_key,
                    prev_state=state.contract_prev.get(monitor_key),
                )
                state.contract_prev[monitor_key] = next_state
                model_yes_prob = float(pred["probability_above_target"])
                if not yes_is_above:
                    model_yes_prob = float(np.clip(1.0 - model_yes_prob, 0.0, 1.0))
                model_side_prob = model_yes_prob if side == "YES" else float(np.clip(1.0 - model_yes_prob, 0.0, 1.0))
        except Exception as e:
            print(f"[runner][warn] position model refresh failed for {ticker}: {e}")

        why_lines = _stoploss_reason_lines(
            move_against=move_against,
            drop_threshold=float(args.stoploss_drop_threshold),
            curr_side_prob=curr_side_prob,
            abs_threshold=float(args.stoploss_absolute_prob),
            mins_left=float(mins_left) if mins_left is not None else None,
            late_minutes=float(args.stoploss_time_hard_minutes),
            late_prob=float(args.stoploss_late_prob),
            model_side_prob=float(model_side_prob) if model_side_prob is not None else None,
            model_threshold=float(args.stoploss_model_side_prob),
            entry_market_prob=entry_market_prob,
        )

        # Periodic probability update for active positions.
        if bool(args.enable_position_update_alert):
            prev_upd = state.last_position_update_at.get(pos_key)
            if prev_upd is None or float((now - prev_upd).total_seconds()) >= float(args.position_update_minutes) * 60.0:
                upd_payload = {
                    "timestamp": _fmt_ts(now),
                    "market_title": (m.get("raw", {}) or {}).get("title"),
                    "market_ticker": ticker,
                    "strike": float(pos.get("strike", np.nan)),
                    "side": side,
                    "model_probability": float(model_yes_prob) if model_yes_prob is not None else float("nan"),
                    "market_yes_probability": curr_yes,
                    "model_side_probability": float(model_side_prob) if model_side_prob is not None else float("nan"),
                    "market_side_probability": curr_side_prob,
                    "edge_side": (
                        float(model_side_prob - curr_side_prob)
                        if model_side_prob is not None
                        else float("nan")
                    ),
                    "minutes_left": float(mins_left) if mins_left is not None else float("nan"),
                    "spot_price": float(spot),
                }
                print("\n=== POSITION UPDATE ===")
                print(json.dumps(upd_payload, indent=2))
                if webhook:
                    try:
                        _send_discord_position_update_alert(webhook, upd_payload)
                    except Exception as e:
                        print(f"[warn] discord position-update webhook failed: {e}")
                state.last_position_update_at[pos_key] = now

        if not why_lines:
            continue

        dedup_key = f"{ticker}:{side}"
        prev = state.stoploss_dedup.get(dedup_key)
        if prev is not None and float((now - prev).total_seconds()) < float(args.stoploss_alert_cooldown_minutes) * 60.0:
            continue

        payload = {
            "timestamp": _fmt_ts(now),
            "market_title": (m.get("raw", {}) or {}).get("title"),
            "market_ticker": ticker,
            "strike": float(pos.get("strike", np.nan)),
            "side": side,
            "entry_market_probability": entry_market_prob,
            "current_market_probability": curr_side_prob,
            "move_against": move_against,
            "model_side_probability": float(model_side_prob) if model_side_prob is not None else float("nan"),
            "minutes_left": float(mins_left) if mins_left is not None else float("nan"),
            "spot_price": float(spot),
            "reason": why_lines[0],
            "why_lines": why_lines,
        }
        print("\n=== RISK ALERT ===")
        print("Stop-loss reason(s):")
        for line in why_lines:
            print(f"  - {line}")
        print(json.dumps(payload, indent=2))
        if webhook:
            try:
                _send_discord_stoploss_alert(webhook, payload)
            except Exception as e:
                print(f"[warn] discord stoploss webhook failed: {e}")
        state.stoploss_dedup[dedup_key] = now
        if bool(args.close_position_on_stoploss_alert):
            remove_keys.append(pos_key)

    for k in remove_keys:
        p = state.open_positions.pop(k, None)
        state.last_position_update_at.pop(k, None)
        if p is not None and state.focus_market_ticker and str(p.get("market_ticker", "")) == str(state.focus_market_ticker):
            state.focus_market_ticker = None


def run_loop(args: argparse.Namespace) -> None:
    catalog = _discover_model_catalog(args.model_path)
    state = RunnerState(
        contract_prev={},
        active_snapshot=None,
        signal_persistence={},
        alert_dedup={},
        stoploss_dedup={},
        open_positions={},
        focus_market_ticker=None,
        last_position_update_at={},
        strike_last_eval={},
        spot_samples=[],
        market_cache=[],
        market_cache_updated_at=None,
    )

    stop = {"requested": False}

    def _request_stop(sig_num: int, _frame: object) -> None:
        stop["requested"] = True
        print(f"[runner] stop requested via signal {sig_num}; finishing current cycle...")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    webhook = args.alert_webhook or os.getenv("ALERT_WEBHOOK_URL", "").strip()
    last_spot_ts: Optional[pd.Timestamp] = None
    last_spot: Optional[float] = None

    while not stop["requested"]:
        cycle_t0 = time.time()
        now = _now_utc()
        cycle_note = "no-op"
        try:
            # Basic spot caching to avoid excessive ticker calls.
            use_cached_spot = False
            if last_spot is not None and last_spot_ts is not None:
                age_sec = float((now - last_spot_ts).total_seconds())
                use_cached_spot = age_sec <= float(args.spot_cache_seconds)
            if use_cached_spot:
                spot = float(last_spot)
            else:
                spot = float(fetch_coinbase_spot())
                last_spot = spot
                last_spot_ts = now
            state.spot_samples = _update_spot_samples(state.spot_samples, ts=now, spot=spot)
            rv_1h = _rolling_realized_vol_1h(state.spot_samples, now=now)

            # Market refresh no more than once per configured interval.
            refresh_needed = True
            if state.market_cache_updated_at is not None:
                age_sec = float((now - state.market_cache_updated_at).total_seconds())
                refresh_needed = age_sec >= float(args.market_refresh_seconds)
            if refresh_needed:
                KALSHI_LAST_ERRORS.clear()
                state.market_cache = fetch_kalshi_btc_markets(fast_mode=bool(args.fast_market_scan))
                state.market_cache_updated_at = now
            if bool(args.enable_position_monitor):
                _monitor_open_positions(
                    state=state,
                    now=now,
                    markets=state.market_cache,
                    spot=float(spot),
                    catalog=catalog,
                    args=args,
                    webhook=webhook,
                )

            hourly_group = _select_active_hourly_group(
                state.market_cache,
                now=now,
                min_expiry_min=float(args.min_expiry_minutes),
                max_expiry_min=float(args.max_expiry_minutes),
            )
            used_fallback_group = False
            used_extended_expiry_fallback = False
            if not hourly_group:
                hourly_group = _select_active_btc_strike_group(
                    state.market_cache,
                    now=now,
                    min_expiry_min=float(args.min_expiry_minutes),
                    max_expiry_min=float(args.max_expiry_minutes),
                )
                used_fallback_group = bool(hourly_group)
            if not hourly_group:
                # If account feed exposes non-hourly BTC strike events (e.g., daily KXBTCD),
                # still pick nearest expiry bucket so daemon remains actionable.
                if not bool(args.hourly_only):
                    hourly_group = _select_active_btc_strike_group(
                        state.market_cache,
                        now=now,
                        min_expiry_min=float(args.min_expiry_minutes),
                        max_expiry_min=float(args.max_expiry_minutes_extended_fallback),
                    )
                    used_fallback_group = bool(hourly_group)
                    used_extended_expiry_fallback = bool(hourly_group)
            if not hourly_group:
                cycle_note = "no_active_hourly_btc_group"
                _append_runner_log(
                    Path(args.runner_log),
                    {
                        "ts_utc": now.isoformat(),
                        "spot": spot,
                        "rv_1h": rv_1h,
                        "status": "skipped",
                        "reason": cycle_note,
                        "market_cache_size": len(state.market_cache),
                        "kalshi_errors": " | ".join(KALSHI_LAST_ERRORS[-3:]) if KALSHI_LAST_ERRORS else "",
                    },
                )
                continue

            minutes_left = float(min(float(m["expiry_minutes"]) for m in hourly_group))
            expected_move = _expected_move_from_rv(
                spot=spot,
                minutes_left=minutes_left,
                rv_1h=rv_1h,
                fallback_sigma_1h=float(args.fallback_sigma_1h),
            )

            candidates: List[Dict[str, object]] = []
            skipped_counts: Dict[str, int] = {}
            for m in hourly_group:
                ticker = str(m.get("market_ticker", ""))
                strike = float(m["strike_value"])
                yes_prob = float(m.get("yes_prob", np.nan))
                no_prob = float(m.get("no_prob", np.nan))
                spread = _parse_spread_fraction(m.get("spread"))
                liq = float(m.get("liquidity", 0.0))
                if not _is_supported_threshold_market(m):
                    skipped_counts["unsupported_contract"] = skipped_counts.get("unsupported_contract", 0) + 1
                    continue

                # Some list endpoints omit liquidity/open_interest; only enforce
                # liquidity gate when value is actually present.
                if liq > 0.0 and liq < float(args.min_kalshi_liquidity):
                    skipped_counts["liq"] = skipped_counts.get("liq", 0) + 1
                    continue
                if spread is not None and spread > float(args.max_spread):
                    skipped_counts["spread"] = skipped_counts.get("spread", 0) + 1
                    continue
                if yes_prob <= float(args.implied_prob_floor) or yes_prob >= float(args.implied_prob_ceiling):
                    skipped_counts["implied_extreme"] = skipped_counts.get("implied_extreme", 0) + 1
                    continue

                dist = abs(strike - spot)
                if dist > float(args.max_dist_sigma) * expected_move:
                    skipped_counts["distance"] = skipped_counts.get("distance", 0) + 1
                    continue

                max_yes_edge = float(args.model_prob_upper_bound) - yes_prob
                max_no_edge = (1.0 - float(args.model_prob_lower_bound)) - no_prob
                if max(max_yes_edge, max_no_edge) <= float(args.min_edge):
                    skipped_counts["no_ev_path"] = skipped_counts.get("no_ev_path", 0) + 1
                    continue

                eval_key = f"{ticker}:{int(round(strike))}"
                prev_eval = state.strike_last_eval.get(eval_key)
                if prev_eval is not None:
                    if float((now - prev_eval).total_seconds()) < float(args.strike_eval_cooldown_seconds):
                        skipped_counts["eval_cooldown"] = skipped_counts.get("eval_cooldown", 0) + 1
                        continue

                # Prioritize ATM and balanced implied probabilities.
                rank_score = (dist / max(expected_move, EPS)) + abs(yes_prob - 0.50)
                mm = dict(m)
                mm["rank_score"] = float(rank_score)
                mm["eval_key"] = eval_key
                mm["strike"] = strike
                candidates.append(mm)

            if not candidates:
                cycle_note = f"no_candidate_after_prune:{json.dumps(skipped_counts, sort_keys=True)}"
                _append_runner_log(
                    Path(args.runner_log),
                    {
                        "ts_utc": now.isoformat(),
                        "spot": spot,
                        "rv_1h": rv_1h,
                        "minutes_left": minutes_left,
                        "expected_move": expected_move,
                        "status": "skipped",
                        "reason": cycle_note,
                        "selection_mode": (
                            "fallback_btc_strike_extended_expiry"
                            if used_extended_expiry_fallback
                            else ("fallback_btc_strike" if used_fallback_group else "hourly_title")
                        ),
                        "candidate_pool_size": len(hourly_group),
                        "prune_counts": json.dumps(skipped_counts, sort_keys=True),
                    },
                )
                if bool(args.debug_rejections):
                    print(
                        f"[runner][prune] pool={len(hourly_group)} accepted=0 "
                        f"counts={json.dumps(skipped_counts, sort_keys=True)}"
                    )
                continue

            if bool(args.focus_on_medium_confidence) and state.focus_market_ticker:
                focused = [c for c in candidates if str(c.get("market_ticker", "")) == str(state.focus_market_ticker)]
                if focused:
                    candidates = focused
                else:
                    # Focused market no longer available; return to normal scanning.
                    state.focus_market_ticker = None

            # Evaluate a single highest-priority strike each cycle.
            candidate = sorted(candidates, key=lambda x: float(x["rank_score"]))[0]
            ticker = str(candidate.get("market_ticker", ""))
            strike = float(candidate["strike"])
            state.strike_last_eval[str(candidate["eval_key"])] = now
            bundle_item = _pick_model_for_cycle(catalog, minutes_left=minutes_left)
            bundle = bundle_item["bundle"]
            interval = str(bundle["interval"])
            yes_is_above = _infer_yes_is_above(candidate)
            contract_key = f"{ticker}:{int(round(strike))}"

            snapshot = _build_live_snapshot(
                interval=interval,
                refresh_period=args.refresh_period,
                require_live_factors=args.require_live_factors,
                critical_factor_max_age_minutes=args.critical_factor_max_age_minutes,
                max_staleness_minutes=args.max_staleness_minutes,
            )
            pred, next_state = _calc_contract_probability(
                bundle=bundle,
                snapshot=snapshot,
                target_price=float(strike),
                current_price=float(spot),
                contract_key=contract_key,
                prev_state=state.contract_prev.get(contract_key),
            )
            state.contract_prev[contract_key] = next_state

            # Align model probability with market YES semantics.
            model_yes = float(pred["probability_above_target"])
            if not yes_is_above:
                model_yes = float(np.clip(1.0 - model_yes, 0.0, 1.0))

            eval_pred = dict(pred)
            eval_pred["probability_above_target"] = model_yes
            eval_pred["probability_below_or_equal_target"] = float(np.clip(1.0 - model_yes, 0.0, 1.0))
            decision = evaluate_signal(eval_pred, candidate, args)

            out_row = {
                "ts_utc": now.isoformat(),
                "status": "evaluated",
                "market_ticker": ticker,
                "selection_mode": (
                    "fallback_btc_strike_extended_expiry"
                    if used_extended_expiry_fallback
                    else ("fallback_btc_strike" if used_fallback_group else "hourly_title")
                ),
                "market_title": (candidate.get("raw", {}) or {}).get("title"),
                "strike": strike,
                "spot": spot,
                "rv_1h": rv_1h,
                "minutes_left": minutes_left,
                "expected_move": expected_move,
                "interval": interval,
                "horizon_minutes": int(bundle["horizon_minutes"]),
                "candidate_pool_size": len(hourly_group),
                "rank_score": float(candidate.get("rank_score", np.nan)),
                "prune_counts": json.dumps(skipped_counts, sort_keys=True),
                "market_yes_prob": float(candidate.get("yes_prob", np.nan)),
                "market_yes_ask_prob": candidate.get("yes_ask_prob"),
                "market_yes_bid_prob": candidate.get("yes_bid_prob"),
                "market_no_ask_prob": candidate.get("no_ask_prob"),
                "market_no_bid_prob": candidate.get("no_bid_prob"),
                "model_yes_prob": model_yes,
                "signal": bool(decision.get("signal", False)),
                "reason": decision.get("reason"),
                "edge": decision.get("edge"),
                "confidence": decision.get("confidence"),
                "confidence_score": decision.get("confidence_score"),
                "critical_missing": json.dumps((pred.get("debug", {}) or {}).get("critical_missing", [])),
                "live_quality": json.dumps((pred.get("debug", {}) or {}).get("live_factor_quality", {})),
            }
            _append_runner_log(Path(args.runner_log), out_row)
            if bool(args.debug_rejections):
                print(
                    f"[runner][select] mode={out_row.get('selection_mode')} pool={len(hourly_group)} "
                    f"pick={ticker} strike={strike:.2f} yes_mkt={float(candidate.get('yes_prob', np.nan)):.3f} "
                    f"yes_model={model_yes:.3f} reason={decision.get('reason')}"
                )

            latest_path = Path(args.runner_out)
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(json.dumps(out_row, indent=2))

            if bool(decision.get("signal", False)) and float(decision.get("confidence_score", 0.0)) >= float(args.min_confidence_score):
                side = str(decision.get("direction", ""))
                conf_label = str(decision.get("confidence", ""))
                conf_rank = _confidence_rank(conf_label)
                track_rank_threshold = _confidence_threshold_rank(str(args.min_track_confidence))
                tracked_count = len(state.open_positions)
                slots_full = tracked_count >= int(args.max_active_alert_positions)
                if bool(args.pause_new_entries_when_full) and slots_full:
                    cycle_note = f"entry_suppressed_capacity:{tracked_count}/{int(args.max_active_alert_positions)}"
                    continue
                alert_key = f"{ticker}:{int(round(strike))}:{side}"
                prev_alert = state.alert_dedup.get(alert_key)
                is_dup = False
                if prev_alert is not None:
                    is_dup = float((now - prev_alert).total_seconds()) < float(args.alert_dedup_minutes) * 60.0
                if not is_dup:
                    payload = {
                        "timestamp": _fmt_ts(now),
                        "market_title": (candidate.get("raw", {}) or {}).get("title"),
                        "market_ticker": ticker,
                        "strike": float(strike),
                        "side": side,
                        "model_probability": model_yes,
                        "market_probability": float(candidate.get("yes_prob", np.nan)),
                        "edge": float(decision.get("edge", 0.0)),
                        "minutes_left": minutes_left,
                        "spot_price": spot,
                        "confidence": decision.get("confidence"),
                        "confidence_score": float(decision.get("confidence_score", 0.0)),
                    }
                    print("\n=== EDGE DETECTED ===")
                    print(json.dumps(payload, indent=2))
                    if webhook:
                        try:
                            _send_discord_alert(webhook, payload)
                        except Exception as e:
                            print(f"[warn] discord webhook failed: {e}")
                    pos_key = f"{ticker}:{side}"
                    if conf_rank >= track_rank_threshold:
                        state.open_positions[pos_key] = {
                            "opened_at": now.isoformat(),
                            "market_ticker": ticker,
                            "market_title": (candidate.get("raw", {}) or {}).get("title"),
                            "strike": float(strike),
                            "side": side,
                            "entry_market_probability": float(candidate.get("yes_prob", np.nan))
                            if side == "YES"
                            else float(np.clip(1.0 - float(candidate.get("yes_prob", np.nan)), 0.0, 1.0)),
                            "entry_model_side_probability": float(model_yes if side == "YES" else float(np.clip(1.0 - model_yes, 0.0, 1.0))),
                            "entry_confidence": conf_label,
                        }
                        if bool(args.focus_on_medium_confidence) and str(decision.get("confidence", "")).lower() == "medium":
                            state.focus_market_ticker = ticker
                    state.alert_dedup[alert_key] = now
                    cycle_note = "alert_sent"
                else:
                    cycle_note = "signal_dedup_suppressed"
            else:
                cycle_note = f"no_signal:{decision.get('reason', 'thresholds')}"

        except Exception as e:
            cycle_note = f"error:{e}"
            print(f"[runner][error] {e}")
            _append_runner_log(
                Path(args.runner_log),
                {
                    "ts_utc": _now_utc().isoformat(),
                    "status": "error",
                    "reason": str(e),
                },
            )
            time.sleep(float(args.error_backoff_seconds))
        finally:
            elapsed = time.time() - cycle_t0
            sleep_sec = max(0.0, float(args.model_cycle_seconds) - elapsed)
            print(f"[runner] cycle complete in {elapsed:.2f}s; status={cycle_note}; sleeping {sleep_sec:.2f}s")
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    print("[runner] stopped gracefully.")


def manual_eval(args: argparse.Namespace) -> None:
    catalog = _discover_model_catalog(args.model_path)

    minutes_left = args.minutes_left
    if minutes_left is None:
        minutes_left = int(_parse_float_input(input("Minutes left to expiry (e.g. 15, 60): ").strip()))
    minutes_left = max(1, int(minutes_left))

    preferred_interval = args.interval
    if preferred_interval is None:
        suggested = _suggest_interval_from_minutes(minutes_left)
        iv_in = input(f"Candle interval to use [{suggested}] (e.g. 1m, 5m, 15m, 1h): ").strip()
        preferred_interval = iv_in if iv_in else suggested

    model_item, model_pick_note = _pick_model_for_manual(catalog, minutes_left=minutes_left, preferred_interval=preferred_interval)
    bundle = model_item["bundle"]
    interval = str(bundle["interval"])

    target_price = args.target_price
    if target_price is None:
        target_price = _parse_float_input(input("Target BTC price (e.g. 68000 or $68,000): ").strip())

    market_yes = args.market_yes_prob
    if market_yes is None:
        market_yes = _parse_prob_input(input("Kalshi YES probability (0-1 or 0-100): ").strip())
    else:
        market_yes = _parse_prob_input(str(market_yes))

    market_no = args.market_no_prob
    if market_no is None:
        market_no = float(np.clip(1.0 - market_yes, 0.0, 1.0))
    else:
        market_no = _parse_prob_input(str(market_no))

    snapshot = _build_live_snapshot(
        interval=interval,
        refresh_period=args.refresh_period,
        require_live_factors=args.require_live_factors,
        critical_factor_max_age_minutes=args.critical_factor_max_age_minutes,
        max_staleness_minutes=args.max_staleness_minutes,
    )
    current_price = float(args.current_price) if args.current_price is not None else fetch_coinbase_spot()

    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    prev_state = None
    if state_path.exists():
        prev_state = json.loads(state_path.read_text())

    pred, next_state = _calc_contract_probability(
        bundle=bundle,
        snapshot=snapshot,
        target_price=float(target_price),
        current_price=float(current_price),
        contract_key="manual",
        prev_state=prev_state,
    )
    state_path.write_text(json.dumps(next_state, indent=2))

    kalshi = {
        "market_ticker": "MANUAL",
        "yes_prob": float(market_yes),
        "no_prob": float(market_no),
        "spread": float(args.market_spread) if args.market_spread is not None else None,
        "liquidity": float(args.market_liquidity),
    }
    ev = evaluate_signal(pred, kalshi, args)

    summary = {
        "timestamp": _fmt_ts(_now_utc()),
        "minutes_left_input": int(minutes_left),
        "preferred_interval_input": str(preferred_interval),
        "model_path_used": str(model_item.get("path")),
        "model_pick_note": model_pick_note,
        "horizon_minutes": int(bundle["horizon_minutes"]),
        "interval": interval,
        "current_price": float(current_price),
        "target_price": float(target_price),
        "model_probability_above_target": float(pred["probability_above_target"]),
        "kalshi_yes_probability": float(market_yes),
        "kalshi_no_probability": float(market_no),
        "edge_yes": float(pred["probability_above_target"] - market_yes),
        "edge_no": float(pred["probability_below_or_equal_target"] - market_no),
        "normalized_distance": float(pred["normalized_distance"]),
        "signal": bool(ev.get("signal", False)),
        "decision": ev.get("direction"),
        "reason": ev.get("reason"),
        "confidence": ev.get("confidence"),
        "confidence_score": ev.get("confidence_score"),
        "alignment": ev.get("alignment"),
        "debug": pred.get("debug", {}),
    }
    dbg = summary.get("debug", {}) if isinstance(summary.get("debug"), dict) else {}
    live_q = dbg.get("live_factor_quality", {}) if isinstance(dbg.get("live_factor_quality"), dict) else {}
    struct = dbg.get("structural_snapshot", {}) if isinstance(dbg.get("structural_snapshot"), dict) else {}
    edge_yes = float(summary["edge_yes"])
    edge_no = float(summary["edge_no"])

    print("")
    print("========== MANUAL EVALUATION ==========")
    print(f"Time: {summary['timestamp']}")
    print(f"Minutes Left Input: {summary['minutes_left_input']}m   Preferred Interval: {summary['preferred_interval_input']}")
    print(f"Model Horizon: {summary['horizon_minutes']}m   Candle Interval: {summary['interval']}")
    if summary.get("model_pick_note"):
        print(f"Model Selection Note: {summary['model_pick_note']}")
    print("")
    print("Input")
    print(f"  Current BTC: {summary['current_price']:.2f}")
    print(f"  Target BTC:  {summary['target_price']:.2f}")
    print(f"  Kalshi YES:  {summary['kalshi_yes_probability']*100:.2f}%")
    print(f"  Kalshi NO:   {summary['kalshi_no_probability']*100:.2f}%")
    print("")
    print("Model")
    print(f"  P(BTC > target): {summary['model_probability_above_target']*100:.2f}%")
    print(f"  Normalized Distance: {summary['normalized_distance']:.3f}")
    print("")
    print("Edge")
    print(f"  YES Edge: {edge_yes*100:+.2f}%")
    print(f"  NO Edge:  {edge_no*100:+.2f}%")
    print("")
    print("Decision")
    print(f"  Signal: {summary['signal']}")
    print(f"  Direction: {summary['decision']}")
    print(f"  Reason: {summary['reason']}")
    print(f"  Confidence: {summary['confidence']} (score={summary['confidence_score']})")
    if summary.get("alignment") is not None:
        print(f"  Structural Alignment: {summary['alignment']}")
    print("")
    print("Data Health")
    print(f"  Latest Candle Age (sec): {dbg.get('seconds_since_latest_candle')}")
    print(f"  Critical Missing: {dbg.get('critical_missing')}")
    if live_q:
        print(f"  Factor Window (min): {live_q.get('recent_window_minutes')}")
        print(f"  Factor Max Age (min): {live_q.get('critical_factor_max_age_minutes')}")
    if struct:
        print("  Structural Snapshot:")
        print(f"    depth_ratio={struct.get('depth_ratio')}")
        print(f"    flow_imbalance_smooth={struct.get('trade_flow_imbalance_smooth')}")
        print(f"    liq_z={struct.get('liq_z')}")
        print(f"    funding_slope={struct.get('funding_slope')}")
        print(f"    vol_rank={struct.get('vol_rank')}")
    print("")
    print("Raw JSON")
    print(json.dumps(summary, indent=2))
    print("=======================================")


def doctor(args: argparse.Namespace) -> None:
    out: Dict[str, object] = {"env": {}, "api_checks": {}, "kalshi_base_candidates": _kalshi_base_candidates()}
    required = ["COINBASE_PRODUCT_ID", "COINALYZE_API_KEY", "COINALYZE_SYMBOL", "FRED_API_KEY"]
    for k in required:
        out["env"][k] = bool(os.getenv(k, "").strip())
    out["env"]["KALSHI_API_KEY"] = bool(os.getenv("KALSHI_API_KEY", "").strip())
    out["env"]["KALSHI_MARKET_TICKER"] = bool(os.getenv("KALSHI_MARKET_TICKER", "").strip())
    out["env"]["KALSHI_BTC_TICKERS"] = bool(os.getenv("KALSHI_BTC_TICKERS", "").strip())
    manual_tickers = _manual_kalshi_btc_tickers()
    out["env"]["KALSHI_BTC_TICKERS_COUNT"] = len(manual_tickers)

    try:
        px = fetch_coinbase_spot()
        out["api_checks"]["coinbase_spot_ok"] = True
        out["api_checks"]["coinbase_spot_price"] = float(px)
    except Exception as e:
        out["api_checks"]["coinbase_spot_ok"] = False
        out["api_checks"]["coinbase_spot_error"] = str(e)

    try:
        idx = pd.date_range(end=_now_utc().floor("min"), periods=20, freq="5min", tz="UTC")
        c = fetch_coinalyze(idx, "5m")
        cov = {
            col: float(c[col].notna().mean()) if col in c.columns else 0.0
            for col in ["funding_rate", "liquidations_long_usd", "liquidations_short_usd", "open_interest"]
        }
        out["api_checks"]["coinalyze_ok"] = True
        out["api_checks"]["coinalyze_coverage"] = cov
    except Exception as e:
        out["api_checks"]["coinalyze_ok"] = False
        out["api_checks"]["coinalyze_error"] = str(e)

    kt = args.kalshi_market_ticker or os.getenv("KALSHI_MARKET_TICKER", "").strip()
    if kt:
        try:
            km = fetch_kalshi_market(kt)
            out["api_checks"]["kalshi_ok"] = True
            out["api_checks"]["kalshi_market"] = {
                "market_ticker": km.get("market_ticker"),
                "yes_prob": km.get("yes_prob"),
                "no_prob": km.get("no_prob"),
                "liquidity": km.get("liquidity"),
                "spread": km.get("spread"),
            }
        except Exception as e:
            out["api_checks"]["kalshi_ok"] = False
            out["api_checks"]["kalshi_error"] = str(e)
    else:
        out["api_checks"]["kalshi_ok"] = False
        out["api_checks"]["kalshi_error"] = "KALSHI_MARKET_TICKER not set"

    try:
        KALSHI_LAST_ERRORS.clear()
        all_mkts = fetch_kalshi_all_open_markets()
        mkts = fetch_kalshi_btc_markets(fast_mode=False)
        out["api_checks"]["kalshi_all_open_markets_found"] = int(len(all_mkts))
        out["api_checks"]["kalshi_btc_scan_ok"] = True
        out["api_checks"]["kalshi_btc_markets_found"] = int(len(mkts))
        if not mkts and all_mkts:
            out["api_checks"]["kalshi_open_market_samples"] = [
                {
                    "market_ticker": m.get("market_ticker"),
                    "title": (m.get("raw", {}) or {}).get("title"),
                    "subtitle": (m.get("raw", {}) or {}).get("subtitle"),
                    "event_ticker": (m.get("raw", {}) or {}).get("event_ticker")
                    or (m.get("raw", {}) or {}).get("eventTicker"),
                    "series_ticker": (m.get("raw", {}) or {}).get("series_ticker")
                    or (m.get("raw", {}) or {}).get("seriesTicker"),
                }
                for m in all_mkts[:25]
            ]
        if bool(args.debug_kalshi):
            lim = max(1, int(args.debug_limit))
            out["api_checks"]["kalshi_btc_debug_samples"] = [
                {
                    "market_ticker": m.get("market_ticker"),
                    "yes_prob": m.get("yes_prob"),
                    "no_prob": m.get("no_prob"),
                    "liquidity": m.get("liquidity"),
                    "strike_raw": m.get("strike"),
                    "expiry": m.get("expiry"),
                    "title": (m.get("raw", {}) or {}).get("title"),
                    "subtitle": (m.get("raw", {}) or {}).get("subtitle"),
                    "event_ticker": (m.get("raw", {}) or {}).get("event_ticker")
                    or (m.get("raw", {}) or {}).get("eventTicker"),
                    "series_ticker": (m.get("raw", {}) or {}).get("series_ticker")
                    or (m.get("raw", {}) or {}).get("seriesTicker"),
                }
                for m in mkts[:lim]
            ]
            if not mkts:
                out["api_checks"]["kalshi_debug_note"] = (
                    "No BTC-like markets found in API response. "
                    "This usually means your key/account feed currently has no BTC market access."
                )
            if manual_tickers:
                manual_ok = []
                manual_err = []
                for t in manual_tickers[:50]:
                    try:
                        mm = fetch_kalshi_market(t)
                        manual_ok.append(
                            {
                                "ticker": t,
                                "yes_prob": mm.get("yes_prob"),
                                "no_prob": mm.get("no_prob"),
                                "expiry": mm.get("expiry"),
                            }
                        )
                    except Exception as e:
                        manual_err.append({"ticker": t, "error": str(e)})
                out["api_checks"]["kalshi_manual_tickers_ok"] = manual_ok
                out["api_checks"]["kalshi_manual_tickers_error"] = manual_err
    except Exception as e:
        out["api_checks"]["kalshi_btc_scan_ok"] = False
        out["api_checks"]["kalshi_btc_scan_error"] = str(e)
    if KALSHI_LAST_ERRORS:
        out["api_checks"]["kalshi_recent_errors"] = KALSHI_LAST_ERRORS[-10:]

    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BTC structural mispricing engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train structural model")
    tr.add_argument("--interval", type=str, default="5m")
    tr.add_argument("--period", type=str, default="60d")
    tr.add_argument("--horizon-minutes", type=int, required=True)
    tr.add_argument("--out-dir", type=str, default=str(MODELS_DIR))
    tr.add_argument("--model-name", type=str, default="btc_mispricing_model.joblib")
    tr.add_argument("--calibration-method", type=str, default="platt", choices=["platt", "isotonic", "none"])
    tr.add_argument("--threshold-z-grid", type=str, default="-2,-1.5,-1,-0.5,0,0.5,1,1.5,2")
    tr.add_argument("--train-bars", type=int, default=600)
    tr.add_argument("--test-bars", type=int, default=120)
    tr.add_argument("--step-bars", type=int, default=60)
    tr.add_argument("--C", type=float, default=0.5, help="Logistic regularization inverse strength")
    tr.add_argument("--return-group-scale", type=float, default=0.5, help="Dampen return features")
    tr.add_argument("--sim-spread-bps", type=float, default=150.0)
    tr.add_argument("--require-live-factors", action="store_true")
    tr.add_argument("--critical-factor-max-age-minutes", type=int, default=90)
    tr.set_defaults(func=train)

    pr = sub.add_parser("predict", help="Single prediction")
    pr.add_argument("--model-path", type=str, required=True)
    pr.add_argument("--target-price", type=float, required=True)
    pr.add_argument("--current-price", type=float, default=None)
    pr.add_argument("--refresh-period", type=str, default="30d")
    pr.add_argument("--max-staleness-minutes", type=int, default=5)
    pr.add_argument("--require-live-factors", action="store_true")
    pr.add_argument("--critical-factor-max-age-minutes", type=int, default=90)
    pr.add_argument("--state-path", type=str, default=str(STATE_DIR / "mispricing_state.json"))

    def _predict_cmd(args: argparse.Namespace) -> None:
        r = predict_once(args)
        print(json.dumps(r, indent=2))

    pr.set_defaults(func=_predict_cmd)

    rn = sub.add_parser("run", help="Continuous BTC-Kalshi mispricing daemon")
    rn.add_argument("--model-path", type=str, required=True, help="Path to one model .joblib or a directory of model .joblib files")
    rn.add_argument("--current-price", type=float, default=None)
    rn.add_argument("--refresh-period", type=str, default="30d")
    rn.add_argument("--max-staleness-minutes", type=int, default=5)
    rn.add_argument("--require-live-factors", action="store_true")
    rn.add_argument("--critical-factor-max-age-minutes", type=int, default=90)
    rn.add_argument("--state-path", type=str, default=str(STATE_DIR / "mispricing_state.json"))

    rn.add_argument("--kalshi-cycle-seconds", type=int, default=60, help="Alias for loop speed; kept for backward compatibility")
    rn.add_argument("--model-cycle-seconds", type=int, default=60, help="Daemon cycle interval in seconds (full evaluation every cycle)")
    rn.add_argument("--market-refresh-seconds", type=int, default=60, help="Refresh Kalshi market list at most once per N seconds")
    rn.add_argument("--fast-market-scan", dest="fast_market_scan", action="store_true", help="Use targeted BTC market queries first (much faster)")
    rn.add_argument("--slow-market-scan", dest="fast_market_scan", action="store_false", help="Include broad all-open market scan every refresh (slower)")
    rn.set_defaults(fast_market_scan=True)
    rn.add_argument("--spot-cache-seconds", type=float, default=2.0, help="Reuse spot quote for this many seconds")
    rn.add_argument("--error-backoff-seconds", type=float, default=2.0)

    rn.add_argument("--market-prob-min", type=float, default=0.10)
    rn.add_argument("--market-prob-max", type=float, default=0.90)
    rn.add_argument("--min-expiry-minutes", type=float, default=0.2)
    rn.add_argument("--max-expiry-minutes", type=float, default=70.0)
    rn.add_argument("--max-expiry-minutes-extended-fallback", type=float, default=1500.0, help="If strict/hourly selection fails, allow nearest BTC strike event up to this horizon")
    rn.add_argument("--hourly-only", action="store_true", help="Require hourly-style selection only (disable extended expiry fallback)")
    rn.add_argument("--max-horizon-model-gap-minutes", type=int, default=20)
    rn.add_argument("--max-abs-normalized-distance", type=float, default=1.5)
    rn.add_argument("--max-prob-jump", type=float, default=0.20)
    rn.add_argument("--max-calibration-delta", type=float, default=0.30)
    rn.add_argument("--fallback-sigma-1h", type=float, default=0.0075, help="Fallback 1h sigma if rolling RV not yet available")
    rn.add_argument("--max-dist-sigma", type=float, default=3.0, help="Prune strikes farther than this multiple of expected move")
    rn.add_argument("--implied-prob-floor", type=float, default=0.05)
    rn.add_argument("--implied-prob-ceiling", type=float, default=0.95)
    rn.add_argument("--model-prob-lower-bound", type=float, default=0.05, help="Lower bound of model plausibility band for EV-path pruning")
    rn.add_argument("--model-prob-upper-bound", type=float, default=0.95, help="Upper bound of model plausibility band for EV-path pruning")

    rn.add_argument("--min-edge", type=float, default=0.0)
    rn.add_argument("--yes-prob-min", type=float, default=0.63)
    rn.add_argument("--yes-kalshi-max", type=float, default=0.55)
    rn.add_argument("--no-prob-max", type=float, default=0.37)
    rn.add_argument("--no-kalshi-min", type=float, default=0.45)
    rn.add_argument("--min-kalshi-liquidity", type=float, default=2000.0)
    rn.add_argument("--max-spread", type=float, default=0.06)
    rn.add_argument("--min-liquidations-coverage", type=float, default=0.10, help="Minimum recent coverage ratio for liquidation features (0-1)")
    rn.add_argument("--min-funding-coverage", type=float, default=0.10, help="Minimum recent coverage ratio for funding_rate feature (0-1)")
    rn.add_argument("--strike-eval-cooldown-seconds", type=float, default=20.0)
    rn.add_argument("--signal-persist-checks", type=int, default=2)
    rn.add_argument("--alert-dedup-minutes", type=float, default=15.0)
    rn.add_argument("--min-confidence-score", type=float, default=0.04)
    rn.add_argument("--max-alerts-per-scan", type=int, default=2)
    rn.add_argument("--alert-webhook", type=str, default=None)
    rn.add_argument("--enable-position-monitor", action="store_true", default=True, help="Track alerted positions and emit risk/stop-loss alerts")
    rn.add_argument("--disable-position-monitor", dest="enable_position_monitor", action="store_false")
    rn.add_argument("--max-active-alert-positions", type=int, default=2, help="Max concurrently tracked/active alerted positions")
    rn.add_argument("--min-track-confidence", type=str, default="medium", choices=["low", "medium", "high"], help="Only alerts at/above this confidence consume active position slots")
    rn.add_argument("--pause-new-entries-when-full", action="store_true", default=True, help="When active position slots are full, suppress new entry alerts")
    rn.add_argument("--allow-new-entries-when-full", dest="pause_new_entries_when_full", action="store_false")
    rn.add_argument("--focus-on-medium-confidence", action="store_true", default=False, help="Optional: after Medium-confidence entry alert, temporarily focus on that market")
    rn.add_argument("--disable-focus-on-medium-confidence", dest="focus_on_medium_confidence", action="store_false")
    rn.add_argument("--enable-position-update-alert", action="store_true", default=True, help="Send periodic model-vs-market probability updates for open positions")
    rn.add_argument("--disable-position-update-alert", dest="enable_position_update_alert", action="store_false")
    rn.add_argument("--position-update-minutes", type=float, default=1.0, help="Cadence for open-position probability updates")
    rn.add_argument("--stoploss-drop-threshold", type=float, default=0.20, help="Trigger risk alert if side probability drops by this amount vs entry")
    rn.add_argument("--stoploss-absolute-prob", type=float, default=0.15, help="Trigger risk alert if side probability falls below this absolute level")
    rn.add_argument("--stoploss-model-side-prob", type=float, default=0.45, help="Trigger risk alert if model side probability drops below this level")
    rn.add_argument("--stoploss-time-hard-minutes", type=float, default=12.0, help="Late-stage window for harder stop checks")
    rn.add_argument("--stoploss-late-prob", type=float, default=0.30, help="Within late-stage window, trigger if side prob below this")
    rn.add_argument("--stoploss-alert-cooldown-minutes", type=float, default=10.0)
    rn.add_argument("--close-position-on-stoploss-alert", action="store_true", default=True, help="After risk alert, stop monitoring that position")
    rn.add_argument("--keep-monitoring-after-stoploss-alert", dest="close_position_on_stoploss_alert", action="store_false")
    rn.add_argument("--runner-out", type=str, default=str(LATEST_DIR / "mispricing_top_contract.json"))
    rn.add_argument("--runner-log", type=str, default=str(LOGS_DIR / "mispricing_runner_scan_log.csv"))
    rn.add_argument("--debug-rejections", action="store_true")
    rn.add_argument("--market-cache-max-age-seconds", type=int, default=1800)
    rn.set_defaults(func=run_loop)

    mn = sub.add_parser("manual", help="Manual threshold + market probability evaluation")
    mn.add_argument("--model-path", type=str, required=True, help="Path to one model .joblib or a directory of model .joblib files")
    mn.add_argument("--minutes-left", type=int, default=None, help="Minutes left until market expiry")
    mn.add_argument("--interval", type=str, default=None, help="Preferred candle interval (1m/5m/15m/1h)")
    mn.add_argument("--target-price", type=float, default=None, help="BTC threshold price")
    mn.add_argument("--current-price", type=float, default=None, help="Optional override, else live Coinbase spot")
    mn.add_argument("--market-yes-prob", type=str, default=None, help="Kalshi YES prob (0-1 or 0-100)")
    mn.add_argument("--market-no-prob", type=str, default=None, help="Kalshi NO prob (0-1 or 0-100), optional")
    mn.add_argument("--market-liquidity", type=float, default=0.0)
    mn.add_argument("--market-spread", type=float, default=None, help="0-1 or 0-100 format")
    mn.add_argument("--refresh-period", type=str, default="30d")
    mn.add_argument("--max-staleness-minutes", type=int, default=5)
    mn.add_argument("--require-live-factors", action="store_true")
    mn.add_argument("--critical-factor-max-age-minutes", type=int, default=90)
    mn.add_argument("--state-path", type=str, default=str(STATE_DIR / "manual_state.json"))

    # Keep same decision thresholds used by evaluate_signal
    mn.add_argument("--market-prob-min", type=float, default=0.10)
    mn.add_argument("--market-prob-max", type=float, default=0.90)
    mn.add_argument("--min-edge", type=float, default=0.0)
    mn.add_argument("--yes-prob-min", type=float, default=0.63)
    mn.add_argument("--yes-kalshi-max", type=float, default=0.55)
    mn.add_argument("--no-prob-max", type=float, default=0.37)
    mn.add_argument("--no-kalshi-min", type=float, default=0.45)
    mn.add_argument("--min-kalshi-liquidity", type=float, default=0.0)
    mn.add_argument("--max-spread", type=float, default=1.0)
    mn.add_argument("--min-liquidations-coverage", type=float, default=0.10)
    mn.add_argument("--min-funding-coverage", type=float, default=0.10)
    mn.add_argument("--max-prob-jump", type=float, default=0.20)
    mn.add_argument("--max-calibration-delta", type=float, default=0.30)
    mn.set_defaults(func=manual_eval)

    dc = sub.add_parser("doctor", help="Check env + API connectivity")
    dc.add_argument("--kalshi-market-ticker", type=str, default=None)
    dc.add_argument("--debug-kalshi", action="store_true")
    dc.add_argument("--debug-limit", type=int, default=15)
    dc.set_defaults(func=doctor)

    return p


def main() -> None:
    load_env()
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "cmd", "") == "run":
        # Backward-compatible alias: if caller uses --kalshi-cycle-seconds only, apply it.
        alias_v = int(getattr(args, "kalshi_cycle_seconds", 60))
        if int(getattr(args, "model_cycle_seconds", 60)) == 60 and alias_v != 60:
            args.model_cycle_seconds = alias_v
    args.func(args)


if __name__ == "__main__":
    main()
