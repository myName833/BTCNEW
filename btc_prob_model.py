#!/usr/bin/env python3
"""BTC threshold probability model with train/predict CLI.

Goal:
Estimate P(BTC_price_at_t+h > target_price) for user-provided target_price and horizon.

Design:
- Train-time: time-series features + threshold-conditioned classifier
- Inference-time: convert target_price into threshold return and predict probability
- Validation: walk-forward, chronological only, Brier/log loss/calibration

Research/engineering notes:
- Order-flow/liquidation/book-depth data are not freely available at quality across long history.
- This script supports optional external factors CSV so those feeds can be added without code changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    load_dotenv = None

EPS = 1e-8
FACTOR_ALIASES = {
    "book_imbalance": "order_book_imbalance",
    "liquidation_long_usd": "liquidations_long_usd",
    "liquidation_short_usd": "liquidations_short_usd",
}
COINBASE_EXCHANGE_BASE = "https://api.exchange.coinbase.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
FRED_BASE = "https://api.stlouisfed.org/fred"
BLS_CALENDAR_ICS = "https://www.bls.gov/schedule/news_release/bls.ics"
KAIKO_BASE = "https://us.market-api.kaiko.io"
AMBERDATA_BASE = "https://web3api.io"


@dataclass
class WalkForwardConfig:
    train_bars: int
    test_bars: int
    step_bars: int


def _horizon_tag_minutes(horizon_minutes: int) -> str:
    return f"{int(horizon_minutes)}m"


def _interval_tag(interval: str) -> str:
    return interval.strip().lower()


def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Unsupported interval '{interval}'. Use Xm or Xh (e.g., 5m, 15m, 1h).")


def _horizon_to_bars(horizon_minutes: int, interval_minutes: int) -> int:
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be > 0")
    return int(np.ceil(horizon_minutes / interval_minutes))


def load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv()


def _period_to_days(period: str) -> int:
    s = period.strip().lower()
    if s.endswith("d"):
        return int(float(s[:-1]))
    if s.endswith("mo"):
        return int(float(s[:-2]) * 30)
    if s.endswith("y"):
        return int(float(s[:-1]) * 365)
    raise ValueError("Unsupported period format. Use Nd/Nmo/Ny, e.g. 180d, 12mo, 2y.")


def _coinbase_granularity(interval: str) -> int:
    minutes = _interval_to_minutes(interval)
    sec = minutes * 60
    allowed = {60, 300, 900, 3600, 21600, 86400}
    if sec not in allowed:
        raise ValueError(
            f"Coinbase candles do not support interval={interval}. "
            "Use one of: 1m, 5m, 15m, 1h, 6h, 1d."
        )
    return sec


def _coinalyze_interval(interval: str) -> str:
    minutes = _interval_to_minutes(interval)
    mapping = {
        1: "1min",
        5: "5min",
        15: "15min",
        30: "30min",
        60: "1hour",
        240: "4hour",
        1440: "daily",
    }
    if minutes in mapping:
        return mapping[minutes]
    if minutes < 60:
        return "15min"
    return "1hour"


def _safe_get_json(url: str, params: Dict[str, str] | None = None, headers: Dict[str, str] | None = None) -> object:
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _extract_first_numeric(obj: object, key_candidates: Sequence[str]) -> float | None:
    keys = {k.lower() for k in key_candidates}
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str) and k.lower() in keys:
                    try:
                        return float(v)
                    except Exception:
                        pass
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _parse_ics_events(text: str) -> List[Tuple[pd.Timestamp, str]]:
    events: List[Tuple[pd.Timestamp, str]] = []
    in_event = False
    dt_token = None
    summary = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            in_event = True
            dt_token = None
            summary = ""
            continue
        if line == "END:VEVENT":
            if in_event and dt_token:
                if re.fullmatch(r"\d{8}", dt_token):
                    ts = pd.to_datetime(dt_token, format="%Y%m%d", utc=True, errors="coerce")
                elif re.fullmatch(r"\d{8}T\d{6}Z", dt_token):
                    ts = pd.to_datetime(dt_token, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
                else:
                    ts = pd.to_datetime(dt_token, utc=True, errors="coerce")
                if pd.notna(ts):
                    events.append((ts, summary or "BLS event"))
            in_event = False
            continue
        if not in_event:
            continue
        if line.startswith("DTSTART"):
            dt_token = line.split(":", 1)[1].strip() if ":" in line else None
        elif line.startswith("SUMMARY"):
            summary = line.split(":", 1)[1].strip() if ":" in line else ""

    return events


def parse_threshold_z_grid(spec: str) -> List[float]:
    vals = [s.strip() for s in spec.split(",") if s.strip()]
    if not vals:
        raise ValueError("threshold-z-grid must contain at least one numeric value")
    out = [float(v) for v in vals]
    return sorted(out)


def _append_latest_coinbase_minute_candle(df: pd.DataFrame, product_id: str, strict_live: bool) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")
    start = now_utc - pd.Timedelta(minutes=20)
    params = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": now_utc.isoformat().replace("+00:00", "Z"),
        "granularity": "60",
    }
    try:
        rows = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product_id}/candles", params=params)
        if not isinstance(rows, list) or not rows:
            raise ValueError("No recent 1-minute candles returned from Coinbase.")
        raw = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        raw["time"] = pd.to_datetime(raw["time"], unit="s", utc=True)
        latest = raw.sort_values("time").iloc[-1]
        ts = pd.Timestamp(latest["time"])
        vals = {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
        }
        out = df.copy()
        out.loc[ts, ["open", "high", "low", "close", "volume"]] = [
            vals["open"],
            vals["high"],
            vals["low"],
            vals["close"],
            vals["volume"],
        ]
        return out.sort_index()
    except Exception as e:
        if strict_live:
            raise ValueError(f"Failed to fetch latest live 1-minute Coinbase candle: {e}") from e
        return df


def _assert_market_fresh(df: pd.DataFrame, max_staleness_minutes: int) -> None:
    if df.empty:
        raise ValueError("Market data is empty.")
    last_ts = pd.Timestamp(df.index.max())
    now_utc = pd.Timestamp.now(tz="UTC")
    age_min = (now_utc - last_ts).total_seconds() / 60.0
    if age_min > float(max_staleness_minutes):
        raise ValueError(
            f"Market candle is stale ({age_min:.2f} min old). "
            f"Max allowed is {max_staleness_minutes} min."
        )


def load_market_data(
    symbol: str,
    period: str,
    interval: str,
    allow_yahoo_fallback: bool = True,
    strict_live: bool = False,
    max_staleness_minutes: int | None = None,
) -> pd.DataFrame:
    # Prefer Coinbase Exchange public candles (no API key required).
    product_id = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    if symbol.upper() == "BTC-USD":
        symbol = product_id

    try:
        granularity = _coinbase_granularity(interval)
        days = _period_to_days(period)
        end = pd.Timestamp.utcnow().floor("min")
        start = end - pd.Timedelta(days=days)

        rows: List[list] = []
        max_points = 300
        step = pd.Timedelta(seconds=granularity * max_points)
        cur_start = start
        while cur_start < end:
            cur_end = min(cur_start + step, end)
            params = {
                "start": cur_start.isoformat().replace("+00:00", "Z"),
                "end": cur_end.isoformat().replace("+00:00", "Z"),
                "granularity": str(granularity),
            }
            data = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{symbol}/candles", params=params)
            if isinstance(data, list) and data:
                rows.extend(data)
            cur_start = cur_end

        if rows:
            # Coinbase format: [time, low, high, open, close, volume]
            raw = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
            raw["time"] = pd.to_datetime(raw["time"], unit="s", utc=True)
            df = raw.drop_duplicates(subset=["time"]).set_index("time").sort_index()
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            if not df.empty:
                df = _append_latest_coinbase_minute_candle(df, product_id=symbol, strict_live=strict_live)
                if max_staleness_minutes is not None:
                    _assert_market_fresh(df, max_staleness_minutes=max_staleness_minutes)
                return df
    except Exception:
        if strict_live:
            raise

    if not allow_yahoo_fallback:
        raise ValueError("Coinbase market fetch failed and Yahoo fallback is disabled.")

    # Fallback to Yahoo when explicitly allowed.
    import yfinance as yf

    hist = yf.download(
        symbol if symbol.endswith("USD") else "BTC-USD",
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        timeout=20,
    )
    if hist.empty:
        raise ValueError("No market data returned from Coinbase or Yahoo. Check network/DNS and period/interval.")

    if isinstance(hist.columns, pd.MultiIndex):
        cols = {}
        for f in ["Open", "High", "Low", "Close", "Volume"]:
            if f in set(hist.columns.get_level_values(0)):
                x = hist.xs(f, axis=1, level=0)
                cols[f.lower()] = x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
        df = pd.DataFrame(cols)
    else:
        required = ["Open", "High", "Low", "Close"]
        miss = [c for c in required if c not in hist.columns]
        if miss:
            raise ValueError(f"Missing required OHLC columns: {miss}")
        df = pd.DataFrame(
            {
                "open": hist["Open"],
                "high": hist["High"],
                "low": hist["Low"],
                "close": hist["Close"],
                "volume": hist["Volume"] if "Volume" in hist.columns else np.nan,
            }
        )

    df.index = pd.to_datetime(df.index, utc=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    if max_staleness_minutes is not None:
        _assert_market_fresh(df, max_staleness_minutes=max_staleness_minutes)
    return df


def load_equity_context(index: pd.DatetimeIndex, interval: str, period: str, enabled: bool = True) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(index=index)

    import yfinance as yf

    try:
        eq = yf.download(
            ["NQ=F", "ES=F"],
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            timeout=20,
        )
    except Exception:
        return pd.DataFrame(index=index)

    if eq.empty:
        return pd.DataFrame(index=index)

    out = pd.DataFrame(index=index)

    if isinstance(eq.columns, pd.MultiIndex):
        for ticker, name in [("NQ=F", "nq"), ("ES=F", "es")]:
            try:
                close = eq[("Close", ticker)]
            except Exception:
                continue
            close.index = pd.to_datetime(close.index, utc=True)
            out = out.join(close.rename(f"{name}_close"), how="left")
    else:
        if "Close" in eq.columns:
            close = eq["Close"]
            close.index = pd.to_datetime(close.index, utc=True)
            out = out.join(close.rename("nq_close"), how="left")

    for c in list(out.columns):
        out[f"{c.split('_')[0]}_ret_1"] = np.log(out[c]).diff()

    out = out.ffill()
    ret_cols = [c for c in out.columns if c.endswith("_ret_1")]
    return out[ret_cols]


def _coinalyze_history_series(payload: object, field_candidates: Sequence[str]) -> pd.DataFrame:
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
    for f in field_candidates:
        if f in df.columns:
            out[f] = pd.to_numeric(df[f], errors="coerce")
    return out


def _resolve_coinalyze_symbol(api_key: str, preferred_symbol: str) -> str:
    try:
        payload = _safe_get_json(
            f"{COINALYZE_BASE}/future-markets",
            params={"api_key": api_key},
        )
        if not isinstance(payload, list) or not payload:
            return preferred_symbol
        symbols = []
        for row in payload:
            if isinstance(row, dict):
                s = row.get("symbol")
                if isinstance(s, str):
                    symbols.append(s)
        if preferred_symbol in symbols:
            return preferred_symbol
        # Fallback: first BTC perpetual aggregate-like symbol.
        btc_perp = [s for s in symbols if "BTC" in s and "PERP" in s]
        if btc_perp:
            if "BTCUSDT_PERP.A" in btc_perp:
                return "BTCUSDT_PERP.A"
            return btc_perp[0]
        return preferred_symbol
    except Exception:
        return preferred_symbol


def fetch_coinalyze_factors(index: pd.DatetimeIndex, interval: str) -> pd.DataFrame:
    api_key = os.getenv("COINALYZE_API_KEY", "").strip()
    symbol = os.getenv("COINALYZE_SYMBOL", "BTCUSDT_PERP.A")
    out = pd.DataFrame(index=index)
    for c in ["funding_rate", "liquidations_long_usd", "liquidations_short_usd", "open_interest", "oi_change"]:
        out[c] = np.nan
    if not api_key:
        return out

    from_ts = int(index.min().timestamp())
    to_ts = int(index.max().timestamp())
    iv = _coinalyze_interval(interval)
    symbol = _resolve_coinalyze_symbol(api_key, symbol)

    common = {"symbols": symbol, "interval": iv, "from": str(from_ts), "to": str(to_ts), "api_key": api_key}

    try:
        p = _safe_get_json(f"{COINALYZE_BASE}/funding-rate-history", params=common)
        x = _coinalyze_history_series(p, ["c", "value", "funding_rate", "rate", "r", "fr", "f"])
        if not x.empty:
            src_col = "c" if "c" in x.columns else list(x.columns)[0]
            out = out.join(x[[src_col]].rename(columns={src_col: "funding_rate"}), how="left")
    except Exception:
        pass

    # Fallback when funding history is sparse/unavailable for the selected range:
    # use latest available funding rate snapshot and propagate forward.
    if out["funding_rate"].notna().mean() < 0.01:
        try:
            snap = _safe_get_json(
                f"{COINALYZE_BASE}/funding-rate",
                params={"symbols": symbol, "api_key": api_key},
            )
            if isinstance(snap, list) and snap:
                row = snap[0] if isinstance(snap[0], dict) else {}
                val = None
                for k in ["rate", "r", "funding_rate", "value", "c", "fr", "f"]:
                    if k in row:
                        try:
                            val = float(row[k])
                            break
                        except Exception:
                            continue
                if val is not None:
                    out["funding_rate"] = out["funding_rate"].fillna(val)
        except Exception:
            pass

    try:
        liq_params = dict(common)
        liq_params["convert_to_usd"] = "true"
        p = _safe_get_json(f"{COINALYZE_BASE}/liquidation-history", params=liq_params)
        x = _coinalyze_history_series(p, ["l", "s", "longs", "shorts"])
        if not x.empty:
            if "l" in x.columns:
                out["liquidations_long_usd"] = x["l"]
            elif "longs" in x.columns:
                out["liquidations_long_usd"] = x["longs"]
            if "s" in x.columns:
                out["liquidations_short_usd"] = x["s"]
            elif "shorts" in x.columns:
                out["liquidations_short_usd"] = x["shorts"]
    except Exception:
        pass

    try:
        p = _safe_get_json(f"{COINALYZE_BASE}/open-interest-history", params=common)
        x = _coinalyze_history_series(p, ["c", "value", "open_interest"])
        if not x.empty:
            src_col = "c" if "c" in x.columns else list(x.columns)[0]
            out = out.join(x[[src_col]].rename(columns={src_col: "open_interest"}), how="left")
    except Exception:
        pass

    if "open_interest" in out.columns:
        out["oi_change"] = out["open_interest"].diff()
    return out.sort_index().ffill()


def fetch_coinbase_order_book_imbalance(index: pd.DatetimeIndex) -> pd.DataFrame:
    product_id = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    out = pd.DataFrame(index=index)
    try:
        data = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product_id}/book", params={"level": "2"})
        bids = data.get("bids", []) if isinstance(data, dict) else []
        asks = data.get("asks", []) if isinstance(data, dict) else []
        bid_notional = sum(float(px) * float(sz) for px, sz, *_ in bids[:200])
        ask_notional = sum(float(px) * float(sz) for px, sz, *_ in asks[:200])
        imb = (bid_notional - ask_notional) / (bid_notional + ask_notional + EPS)
        out["order_book_imbalance"] = float(np.clip(imb, -1.0, 1.0))
    except Exception:
        out["order_book_imbalance"] = np.nan
    return out.ffill()


def fetch_coinbase_spot_price() -> float | None:
    product_id = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    try:
        data = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product_id}/ticker")
        if isinstance(data, dict) and "price" in data:
            return float(data["price"])
    except Exception:
        return None
    return None


def fetch_coinbase_trade_flow(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fetch recent Coinbase trades and derive short-horizon order-flow features.

    Note: Coinbase public trades endpoint is recent-history only. These features are
    most informative for live prediction and may be sparse historically.
    """
    product_id = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD")
    lookback_minutes = int(float(os.getenv("COINBASE_TRADES_LOOKBACK_MINUTES", "60")))
    out = pd.DataFrame(index=index)
    out["trade_flow_imbalance"] = np.nan
    out["trade_flow_buy_share"] = np.nan
    out["trade_flow_notional_usd"] = np.nan
    try:
        rows = _safe_get_json(f"{COINBASE_EXCHANGE_BASE}/products/{product_id}/trades", params={"limit": "1000"})
        if not isinstance(rows, list) or not rows:
            return out
        df = pd.DataFrame(rows)
        if "time" not in df.columns or "price" not in df.columns or "size" not in df.columns:
            return out
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=lookback_minutes)
        df = df[df["time"] >= cutoff]
        if df.empty:
            return out
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df = df.dropna(subset=["price", "size"])
        if df.empty:
            return out
        df["notional"] = df["price"] * df["size"]
        if "side" in df.columns:
            side = df["side"].astype(str).str.lower()
        else:
            side = pd.Series([""] * len(df), index=df.index)
        buy = df[side == "buy"]["notional"].sum()
        sell = df[side == "sell"]["notional"].sum()
        total = float(buy + sell)
        if total <= 0:
            return out
        imbalance = float((buy - sell) / (total + EPS))
        buy_share = float(buy / (total + EPS))
        out["trade_flow_imbalance"] = imbalance
        out["trade_flow_buy_share"] = buy_share
        out["trade_flow_notional_usd"] = total
        return out
    except Exception:
        return out


def validate_required_provider_keys() -> None:
    missing = []
    for key in ["KAIKO_API_KEY", "AMBERDATA_API_KEY"]:
        if not os.getenv(key, "").strip():
            missing.append(key)
    if missing:
        raise ValueError(
            "Missing required API keys in .env: "
            f"{missing}. Set these only if you want to enable those optional premium feeds."
        )


def fetch_kaiko_factors(index: pd.DatetimeIndex) -> pd.DataFrame:
    api_key = os.getenv("KAIKO_API_KEY", "").strip()
    if not api_key:
        raise ValueError("KAIKO_API_KEY is required.")
    instrument = os.getenv("KAIKO_INSTRUMENT", "btc-usd")
    exchange = os.getenv("KAIKO_EXCHANGE", "cbse")
    url = os.getenv(
        "KAIKO_ORDERBOOK_URL",
        f"{KAIKO_BASE}/v2/data/order_book_snapshots.v1/exchanges/{exchange}/spot/{instrument}",
    )
    params = {"page_size": "1"}
    headers = {"X-Api-Key": api_key}
    payload = _safe_get_json(url, params=params, headers=headers)
    mid = _extract_first_numeric(payload, ["mid_price", "mid", "price"])
    spread_bps = _extract_first_numeric(payload, ["spread_bps", "spread"])
    if mid is None:
        raise ValueError("Kaiko response missing usable mid-price field.")

    out = pd.DataFrame(index=index)
    out["kaiko_mid_price"] = float(mid)
    out["kaiko_spread_bps"] = float(spread_bps) if spread_bps is not None else 0.0
    return out


def fetch_amberdata_factors(index: pd.DatetimeIndex) -> pd.DataFrame:
    api_key = os.getenv("AMBERDATA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("AMBERDATA_API_KEY is required.")
    pair = os.getenv("AMBERDATA_PAIR", "BTC-USD")
    exchange = os.getenv("AMBERDATA_EXCHANGE", "coinbase")
    url = os.getenv(
        "AMBERDATA_SPOT_URL",
        f"{AMBERDATA_BASE}/api/v2/market/spot/prices/latest?pair={pair}&exchange={exchange}",
    )
    headers = {"x-api-key": api_key}
    payload = _safe_get_json(url, headers=headers)
    spot = _extract_first_numeric(payload, ["price", "last_price", "mark_price", "mid_price"])
    spread_bps = _extract_first_numeric(payload, ["spread_bps", "spread"])
    if spot is None:
        raise ValueError("Amberdata response missing usable spot price field.")

    out = pd.DataFrame(index=index)
    out["amber_spot_price"] = float(spot)
    out["amber_spread_bps"] = float(spread_bps) if spread_bps is not None else 0.0
    return out


def fetch_macro_event_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    details_by_date: Dict[object, List[str]] = {}

    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key:
        try:
            params = {
                "api_key": fred_key,
                "file_type": "json",
                "realtime_start": index.min().strftime("%Y-%m-%d"),
                "realtime_end": index.max().strftime("%Y-%m-%d"),
                "limit": "1000",
            }
            payload = _safe_get_json(f"{FRED_BASE}/releases/dates", params=params)
            rel = payload.get("release_dates", []) if isinstance(payload, dict) else []
            for row in rel:
                d = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
                if pd.notna(d):
                    details_by_date.setdefault(d.date(), []).append("FRED release day")
        except Exception:
            pass

    try:
        txt = requests.get(BLS_CALENDAR_ICS, timeout=20).text
        for ts, summary in _parse_ics_events(txt):
            details_by_date.setdefault(ts.date(), []).append(f"BLS: {summary}")
    except Exception:
        pass

    out = pd.DataFrame(index=index)
    out["macro_event_flag"] = [1.0 if ts.date() in details_by_date else 0.0 for ts in index]
    out["macro_event_detail"] = [
        "; ".join(sorted(set(details_by_date.get(ts.date(), [])))) if ts.date() in details_by_date else ""
        for ts in index
    ]
    return out


def load_optional_factors(csv_path: str | None, index: pd.DatetimeIndex | None, interval: str) -> pd.DataFrame:
    """Load optional external factors CSV.

    Expected:
    - one timestamp column (time/date/timestamp-like name)
    - numeric factor columns, e.g.:
      order_flow_imbalance, book_imbalance, liquidations_long_usd, liquidations_short_usd,
      funding_rate, open_interest, oi_change, basis
    """
    base_index = index if index is not None else pd.DatetimeIndex([], tz="UTC")

    # Live factors from configured APIs.
    live = pd.DataFrame(index=base_index)
    if len(base_index) > 0:
        live = live.join(fetch_coinalyze_factors(base_index, interval=interval), how="left")
        live = live.join(fetch_coinbase_order_book_imbalance(base_index), how="left")
        live = live.join(fetch_coinbase_trade_flow(base_index), how="left")
        if os.getenv("KAIKO_API_KEY", "").strip():
            try:
                live = live.join(fetch_kaiko_factors(base_index), how="left")
            except Exception:
                pass
        if os.getenv("AMBERDATA_API_KEY", "").strip():
            try:
                live = live.join(fetch_amberdata_factors(base_index), how="left")
            except Exception:
                pass
        live = live.join(fetch_macro_event_flags(base_index), how="left")
        no_ffill_cols = {"macro_event_flag", "macro_event_detail"}
        ffill_cols = [c for c in live.columns if c not in no_ffill_cols]
        if ffill_cols:
            live[ffill_cols] = live[ffill_cols].ffill()
        if "macro_event_flag" in live.columns:
            live["macro_event_flag"] = live["macro_event_flag"].fillna(0.0)
        if "macro_event_detail" in live.columns:
            live["macro_event_detail"] = live["macro_event_detail"].fillna("")

    if not csv_path:
        return live.sort_index()

    df = pd.read_csv(csv_path)
    time_col = None
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "date" in cl or "timestamp" in cl:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Factors CSV requires a timestamp column.")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col])

    numeric_cols = [c for c in df.columns if c != time_col]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.set_index(time_col).rename(columns=FACTOR_ALIASES)
    out = out.ffill()
    if len(base_index) == 0:
        return out
    merged = pd.merge_asof(
        live.sort_index(),
        out.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    no_ffill_cols = {"macro_event_flag", "macro_event_detail"}
    ffill_cols = [c for c in merged.columns if c not in no_ffill_cols]
    if ffill_cols:
        merged[ffill_cols] = merged[ffill_cols].ffill()
    if "macro_event_flag" in merged.columns:
        merged["macro_event_flag"] = merged["macro_event_flag"].fillna(0.0)
    if "macro_event_detail" in merged.columns:
        merged["macro_event_detail"] = merged["macro_event_detail"].fillna("")
    return merged.sort_index()


def enrich_external_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure requested external factors exist and add robust engineered variants."""
    out = df.copy()

    needed_defaults = {
        "order_book_imbalance": 0.0,
        "trade_flow_imbalance": 0.0,
        "trade_flow_buy_share": 0.5,
        "trade_flow_notional_usd": 0.0,
        "kaiko_mid_price": 0.0,
        "kaiko_spread_bps": 0.0,
        "amber_spot_price": 0.0,
        "amber_spread_bps": 0.0,
        "order_flow_imbalance": 0.0,
        "funding_rate": 0.0,
        "liquidations_long_usd": 0.0,
        "liquidations_short_usd": 0.0,
        "macro_event_flag": 0.0,
        "nq_ret_1": 0.0,
        "es_ret_1": 0.0,
    }
    for col, default in needed_defaults.items():
        if col not in out.columns:
            out[col] = default

    out["order_book_imbalance"] = out["order_book_imbalance"].clip(-1, 1)
    out["order_flow_imbalance"] = out["order_flow_imbalance"].clip(-1, 1)
    out["funding_rate"] = out["funding_rate"].fillna(0.0)
    out["macro_event_flag"] = (out["macro_event_flag"].fillna(0.0) > 0).astype(float)

    liq_winsor_q = float(os.getenv("LIQ_WINSOR_Q", "0.99"))
    long_raw = pd.to_numeric(out["liquidations_long_usd"], errors="coerce").clip(lower=0.0)
    short_raw = pd.to_numeric(out["liquidations_short_usd"], errors="coerce").clip(lower=0.0)
    out["liq_long_missing_flag"] = long_raw.isna().astype(float)
    out["liq_short_missing_flag"] = short_raw.isna().astype(float)

    # Use recent carries and median fallback instead of hard zero fill.
    long_liq = long_raw.ffill()
    short_liq = short_raw.ffill()
    long_med = long_raw.median()
    short_med = short_raw.median()
    long_liq = long_liq.fillna(0.0 if pd.isna(long_med) else float(long_med))
    short_liq = short_liq.fillna(0.0 if pd.isna(short_med) else float(short_med))
    out["liquidations_long_usd"] = long_liq
    out["liquidations_short_usd"] = short_liq
    if len(long_liq) > 10:
        long_cap = float(long_liq.quantile(liq_winsor_q))
        short_cap = float(short_liq.quantile(liq_winsor_q))
        long_liq = long_liq.clip(upper=max(long_cap, 0.0))
        short_liq = short_liq.clip(upper=max(short_cap, 0.0))
    out["liq_long_log"] = np.log1p(long_liq)
    out["liq_short_log"] = np.log1p(short_liq)
    out["liq_total_log"] = np.log1p(long_liq + short_liq)
    out["liq_imbalance"] = (long_liq - short_liq) / (long_liq + short_liq + EPS)

    if "open_interest" in out.columns:
        oi_missing = out["open_interest"].isna().astype(float)
        out["open_interest"] = out["open_interest"].ffill()
        oi_med = out["open_interest"].median()
        out["open_interest"] = out["open_interest"].fillna(0.0 if pd.isna(oi_med) else float(oi_med))
        out["oi_missing_flag"] = oi_missing
    else:
        out["open_interest"] = 0.0
        out["oi_missing_flag"] = 1.0

    if "oi_change" in out.columns:
        out["oi_change"] = out["oi_change"].fillna(0.0)
    else:
        out["oi_change"] = 0.0

    fr_mean = out["funding_rate"].rolling(96, min_periods=24).mean()
    fr_std = out["funding_rate"].rolling(96, min_periods=24).std()
    out["funding_z_96"] = (out["funding_rate"] - fr_mean) / (fr_std + EPS)

    return out


def validate_live_factor_coverage(
    factors: pd.DataFrame,
    require: bool,
    min_coverage: float = 0.10,
    allow_recent_for_predict: bool = False,
) -> None:
    if not require:
        return
    must_have = [
        "order_book_imbalance",
        "funding_rate",
        "liquidations_long_usd",
        "liquidations_short_usd",
        "macro_event_flag",
    ]
    missing = [c for c in must_have if c not in factors.columns]
    if missing:
        raise ValueError(
            "Live factor schema is incomplete. "
            f"Missing columns: {missing}. "
            "Check Coinalyze API key/symbol and loader configuration."
        )
    low_cov = []
    for c in must_have:
        cov = factors[c].notna().mean()
        if cov < min_coverage:
            if allow_recent_for_predict and not factors[c].dropna().empty:
                continue
            low_cov.append(c)
    if low_cov:
        raise ValueError(
            "Required live factors have very low coverage. "
            f"Check API keys/endpoints for columns: {low_cov}. "
            "Expected env: COINALYZE_API_KEY and COINALYZE_SYMBOL (e.g., BTCUSDT_PERP.A)."
        )


def validate_recent_live_factors(
    factors: pd.DataFrame,
    max_age_minutes: int = 30,
) -> None:
    must_have_recent = [
        "funding_rate",
        "liquidations_long_usd",
        "liquidations_short_usd",
    ]
    optional_recent = ["open_interest"]
    if factors.empty:
        raise ValueError("Factors dataframe is empty; cannot validate recent live factors.")
    end_ts = pd.Timestamp(factors.index.max())
    start_ts = end_ts - pd.Timedelta(minutes=max_age_minutes)
    recent = factors.loc[factors.index >= start_ts]
    if recent.empty:
        raise ValueError(f"No factor rows in the last {max_age_minutes} minutes.")
    missing_recent = []
    for c in must_have_recent:
        if c not in recent.columns or recent[c].dropna().empty:
            missing_recent.append(c)
    if missing_recent:
        raise ValueError(
            "Recent live factor data missing for columns: "
            f"{missing_recent} in last {max_age_minutes} minutes."
        )
    missing_optional = []
    for c in optional_recent:
        if c not in recent.columns or recent[c].dropna().empty:
            missing_optional.append(c)
    if missing_optional:
        print(
            "[warn] optional recent factors missing: "
            f"{missing_optional} in last {max_age_minutes} minutes."
        )


def build_prediction_debug_payload(
    market: pd.DataFrame,
    factors: pd.DataFrame,
    latest: pd.Series,
    feature_cols: List[str],
    model_prob_raw: float,
    model_prob_calibrated: float,
    calibration_method: str,
) -> Dict[str, object]:
    must_have = [
        "order_book_imbalance",
        "funding_rate",
        "liquidations_long_usd",
        "liquidations_short_usd",
        "macro_event_flag",
    ]
    coverage = {}
    for c in must_have:
        if c in factors.columns and len(factors) > 0:
            coverage[c] = float(factors[c].notna().mean())
        else:
            coverage[c] = 0.0

    key_latest = {}
    for c in must_have + [
        "trade_flow_imbalance",
        "trade_flow_buy_share",
        "trade_flow_notional_usd",
        "open_interest",
        "oi_change",
        "nq_ret_1",
        "es_ret_1",
        "vol_24",
        "threshold_z",
    ]:
        if c in latest.index:
            try:
                key_latest[c] = float(latest[c])
            except Exception:
                key_latest[c] = None
        else:
            key_latest[c] = None

    nan_count = int(pd.Series([latest.get(c, np.nan) for c in feature_cols]).isna().sum())

    last_ts = market.index.max() if len(market) > 0 else None
    age_seconds = None
    if last_ts is not None:
        now_utc = pd.Timestamp.now(tz="UTC")
        age_seconds = float((now_utc - pd.Timestamp(last_ts)).total_seconds())

    return {
        "raw_model_probability_above_target": float(model_prob_raw),
        "calibrated_model_probability_above_target": float(model_prob_calibrated),
        "calibration_method": calibration_method,
        "feature_nan_count_in_final_row": nan_count,
        "required_factor_coverage": coverage,
        "latest_key_features": key_latest,
        "macro_event_detail_latest_row": str(latest.get("macro_event_detail", "")),
        "data_freshness": {
            "latest_market_candle_timestamp_utc": str(last_ts) if last_ts is not None else None,
            "seconds_since_latest_candle": age_seconds,
        },
    }


def _apply_calibrator(calibrator: object, p_raw: np.ndarray) -> np.ndarray:
    p = np.asarray(p_raw, dtype=float).reshape(-1)
    if calibrator is None:
        return np.clip(p, 1e-6, 1 - 1e-6)

    # New artifact format.
    if isinstance(calibrator, dict):
        method = str(calibrator.get("method", "none")).lower()
        model = calibrator.get("model")
        if method == "isotonic" and model is not None:
            return np.clip(model.predict(p), 1e-6, 1 - 1e-6)
        if method == "platt" and model is not None:
            return np.clip(model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        return np.clip(p, 1e-6, 1 - 1e-6)

    # Backward compatibility with old artifacts that stored IsotonicRegression directly.
    if hasattr(calibrator, "predict"):
        try:
            return np.clip(calibrator.predict(p), 1e-6, 1 - 1e-6)
        except Exception:
            pass
    return np.clip(p, 1e-6, 1 - 1e-6)


def compute_local_feature_influence(
    clf: HistGradientBoostingClassifier,
    calibrator: object,
    x_row: np.ndarray,
    feature_cols: List[str],
    baseline_vals: np.ndarray,
) -> Dict[str, object]:
    """Approximate local feature impact via one-feature replacement to baseline."""
    p_raw = float(np.clip(clf.predict_proba(x_row.reshape(1, -1))[:, 1][0], 1e-6, 1 - 1e-6))
    p_full = float(_apply_calibrator(calibrator, np.array([p_raw]))[0])

    rows = []
    for i, name in enumerate(feature_cols):
        x_alt = x_row.copy()
        x_alt[i] = baseline_vals[i]
        p_alt_raw = float(np.clip(clf.predict_proba(x_alt.reshape(1, -1))[:, 1][0], 1e-6, 1 - 1e-6))
        p_alt = float(_apply_calibrator(calibrator, np.array([p_alt_raw]))[0])
        rows.append(
            {
                "feature": name,
                "delta_prob": float(p_full - p_alt),
                "value": float(x_row[i]),
                "baseline": float(baseline_vals[i]),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: abs(r["delta_prob"]), reverse=True)
    top = rows_sorted[:10]
    positive = [r for r in top if r["delta_prob"] > 0]
    negative = [r for r in top if r["delta_prob"] < 0]
    return {
        "top_feature_influences": top,
        "top_positive_influences": positive[:5],
        "top_negative_influences": negative[:5],
    }


def append_prediction_debug_log(out_dir: Path, result: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prediction_debug_log.csv"
    row = {
        "logged_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "prediction_timestamp": result.get("timestamp"),
        "horizon_minutes": result.get("horizon_minutes"),
        "target_price": result.get("target_price"),
        "current_price": result.get("current_price"),
        "prob_above": result.get("probability_price_above_target"),
        "prob_below_or_equal": result.get("probability_price_below_or_equal_target"),
        "threshold_z": result.get("threshold_z"),
    }
    dbg = result.get("debug", {})
    if isinstance(dbg, dict):
        row["raw_prob"] = dbg.get("raw_model_probability_above_target")
        row["calibrated_prob"] = dbg.get("calibrated_model_probability_above_target")
        cov = dbg.get("required_factor_coverage", {})
        if isinstance(cov, dict):
            row["cov_funding_rate"] = cov.get("funding_rate")
            row["cov_liq_long"] = cov.get("liquidations_long_usd")
            row["cov_liq_short"] = cov.get("liquidations_short_usd")
            row["cov_order_book"] = cov.get("order_book_imbalance")
            row["cov_macro_flag"] = cov.get("macro_event_flag")
        row["macro_event_detail"] = dbg.get("macro_event_detail_latest_row")
        row["calibration_method"] = dbg.get("calibration_method")
        row["calibration_plateau_flag"] = dbg.get("calibration_plateau_flag")
        top = dbg.get("top_feature_influences", [])
        if isinstance(top, list) and top:
            row["top_feature_1"] = top[0].get("feature")
            row["top_feature_1_delta"] = top[0].get("delta_prob")
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        out = pd.concat([prev, df], ignore_index=True)
    else:
        out = df
    out.to_csv(path, index=False)


def detect_calibration_plateau(out_dir: Path, current_prob: float, n: int = 4, tol: float = 1e-12) -> bool:
    path = out_dir / "prediction_debug_log.csv"
    vals: List[float] = []
    if path.exists():
        try:
            prev = pd.read_csv(path)
            if "calibrated_prob" in prev.columns:
                series = pd.to_numeric(prev["calibrated_prob"], errors="coerce").dropna()
                vals = series.tail(max(n - 1, 0)).tolist()
        except Exception:
            vals = []
    vals.append(float(current_prob))
    if len(vals) < n:
        return False
    return (max(vals[-n:]) - min(vals[-n:])) <= tol


def clean_old_reports(out_dir: Path) -> None:
    patterns = [
        "*_train_report.json",
        "*_walk_forward_metrics.csv",
        "*_walk_forward_predictions.csv",
        "train_report.json",
        "walk_forward_metrics.csv",
        "walk_forward_predictions.csv",
        "latest_probability_output.json",
    ]
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def build_base_features(
    market: pd.DataFrame,
    horizon_bars: int,
    interval: str,
    period: str,
    factors: pd.DataFrame | None,
    include_equity_context: bool = True,
) -> pd.DataFrame:
    df = market.copy()

    # Core short-term state
    df["ret_1"] = np.log(df["close"]).diff()
    df["ret_3_raw"] = np.log(df["close"] / df["close"].shift(3))
    df["ret_12"] = np.log(df["close"] / df["close"].shift(12))
    df["vol_24"] = df["ret_1"].rolling(24).std()
    df["vol_72"] = df["ret_1"].rolling(72).std()
    df["trend_12"] = df["ret_1"].rolling(12).mean()

    # Candle microstructure proxies (when true L2/L3 unavailable)
    df["hl_spread_raw"] = (df["high"] - df["low"]) / df["close"].clip(lower=EPS)
    df["co_move"] = np.log(df["close"] / df["open"].clip(lower=EPS))

    # Smooth high-variance short-horizon drivers to reduce one-bar overreaction.
    smooth_w = int(float(os.getenv("FEATURE_SMOOTH_WINDOW", "3")))
    smooth_w = max(1, smooth_w)
    ret3_clip = float(os.getenv("RET3_CLIP", "0.03"))
    spread_clip = float(os.getenv("HL_SPREAD_CLIP", "0.05"))
    df["ret_3"] = df["ret_3_raw"].clip(-ret3_clip, ret3_clip).rolling(smooth_w, min_periods=1).mean()
    df["hl_spread"] = df["hl_spread_raw"].clip(0.0, spread_clip).rolling(smooth_w, min_periods=1).mean()

    if "volume" in df.columns:
        vol_log = np.log1p(df["volume"].clip(lower=0))
        df["vol_z_24"] = (vol_log - vol_log.rolling(24).mean()) / (vol_log.rolling(24).std() + EPS)

    # External equity context
    eq = load_equity_context(
        df.index,
        interval=interval,
        period=period,
        enabled=include_equity_context,
    )
    df = df.join(eq, how="left")

    # Optional external factors: order flow, liquidation, funding, OI, etc.
    if factors is not None and not factors.empty:
        df = pd.merge_asof(
            df.sort_index(),
            factors.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )

    # Requested real-world factors (or neutral defaults if feed absent)
    df = enrich_external_factors(df)

    # Forward return at horizon h bars
    df["fwd_ret_h"] = np.log(df["close"].shift(-horizon_bars) / df["close"])
    return df


def make_threshold_training_set(
    feat_df: pd.DataFrame,
    threshold_z_grid: Sequence[float],
) -> pd.DataFrame:
    """Expand each timestamp into multiple threshold-conditioned samples.

    A sample predicts P(fwd_ret_h > threshold_ret | state_t, threshold_ret).
    threshold_ret is parameterized by z * vol_24(t) so one model supports arbitrary target prices.
    """
    needed_cols = ["fwd_ret_h", "vol_24"]
    base = feat_df.dropna(subset=needed_cols).copy()

    rows = []
    for z in threshold_z_grid:
        tmp = base.copy()
        tmp["threshold_z"] = float(z)
        tmp["threshold_ret"] = tmp["vol_24"].clip(lower=EPS) * float(z)
        # Extra target-distance features to anchor prediction to asked price threshold.
        tmp["threshold_abs"] = tmp["threshold_ret"].abs()
        tmp["threshold_ret_sq"] = tmp["threshold_ret"] ** 2
        tmp["threshold_z_abs"] = tmp["threshold_z"].abs()
        tmp["y_event"] = (tmp["fwd_ret_h"] > tmp["threshold_ret"]).astype(int)
        rows.append(tmp)

    out = pd.concat(rows, axis=0)
    out = out.sort_index()
    return out


def split_walk_forward(n: int, cfg: WalkForwardConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while True:
        tr_end = start + cfg.train_bars
        te_end = tr_end + cfg.test_bars
        if te_end > n:
            break
        splits.append((np.arange(start, tr_end), np.arange(tr_end, te_end)))
        start += cfg.step_bars
    if not splits:
        raise ValueError("No walk-forward splits. Increase history or reduce window sizes.")
    return splits


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "event_rate": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
    }


def _fit_calibrated_classifier(
    x: np.ndarray,
    y: np.ndarray,
    calibration_method: str = "platt",
) -> Tuple[HistGradientBoostingClassifier, object]:
    # Chronological split for calibration (no random CV leakage)
    cut = max(int(0.8 * len(x)), 1)
    x_fit, y_fit = x[:cut], y[:cut]
    x_cal, y_cal = x[cut:], y[cut:]

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=40,
        random_state=42,
    )
    clf.fit(x_fit, y_fit)

    if len(x_cal) < 100 or len(np.unique(y_cal)) < 2:
        return clf, {"method": "none", "model": None}

    p_cal = clf.predict_proba(x_cal)[:, 1]
    method = calibration_method.strip().lower()
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        return clf, {"method": "isotonic", "model": iso}
    if method == "platt":
        platt = LogisticRegression(solver="lbfgs")
        platt.fit(p_cal.reshape(-1, 1), y_cal)
        return clf, {"method": "platt", "model": platt}
    return clf, {"method": "none", "model": None}


def _predict_calibrated(
    clf: HistGradientBoostingClassifier,
    calibrator: object,
    x: np.ndarray,
) -> np.ndarray:
    p_raw = clf.predict_proba(x)[:, 1]
    return _apply_calibrator(calibrator, p_raw)


def walk_forward_eval(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: WalkForwardConfig,
    calibration_method: str = "platt",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = train_df.dropna(subset=feature_cols + ["y_event", "threshold_ret", "vol_24"]).copy()
    base = base.sort_index()

    n = len(base)
    splits = split_walk_forward(n, cfg)
    if verbose:
        print(f"[walk-forward] rows={n}, folds={len(splits)}")
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        if verbose:
            print(f"[walk-forward] fold {fold}/{len(splits)}")
        tr = base.iloc[tr_idx]
        te = base.iloc[te_idx]

        xtr = tr[feature_cols].to_numpy()
        ytr = tr["y_event"].to_numpy()
        xte = te[feature_cols].to_numpy()
        yte = te["y_event"].to_numpy()

        clf, calibrator = _fit_calibrated_classifier(xtr, ytr, calibration_method=calibration_method)
        p_model = _predict_calibrated(clf, calibrator, xte)

        # Baseline: normal CDF under zero-drift with current vol
        z = te["threshold_ret"].to_numpy() / np.clip(te["vol_24"].to_numpy(), EPS, None)
        p_base = 1.0 - norm.cdf(z)
        p_base = np.clip(p_base, 1e-6, 1 - 1e-6)

        fold_rows.append(
            pd.DataFrame(
                {
                    "timestamp": te.index,
                    "fold": fold,
                    "y_true": yte,
                    "p_model": p_model,
                    "p_baseline": p_base,
                    "threshold_z": te["threshold_z"].to_numpy(),
                }
            )
        )

    oos = pd.concat(fold_rows, ignore_index=True).sort_values("timestamp")

    metrics = []
    for name, col in [("model", "p_model"), ("baseline", "p_baseline")]:
        m = _metrics(oos["y_true"].to_numpy(), oos[col].to_numpy())
        m["name"] = name
        metrics.append(m)

    metrics_df = pd.DataFrame(metrics).set_index("name")
    return oos, metrics_df


def train_final_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    calibration_method: str = "platt",
) -> Tuple[HistGradientBoostingClassifier, object]:
    fit_df = train_df.dropna(subset=feature_cols + ["y_event"]).sort_index()
    x = fit_df[feature_cols].to_numpy()
    y = fit_df["y_event"].to_numpy()
    return _fit_calibrated_classifier(x, y, calibration_method=calibration_method)


def parse_horizon_minutes(raw: str) -> int:
    minutes = int(np.ceil(float(raw.strip())))
    if minutes <= 0:
        raise ValueError("Horizon minutes must be > 0")
    return minutes


def parse_target_price_input(raw: str) -> float:
    cleaned = raw.strip().replace("$", "").replace(",", "")
    price = float(cleaned)
    if price <= 0:
        raise ValueError("Target price must be > 0")
    return price


def parse_interval_minutes_input(raw: str) -> str:
    minutes = int(np.ceil(float(raw.strip())))
    if minutes <= 0:
        raise ValueError("Interval minutes must be > 0")
    allowed = {1, 5, 15, 60, 360, 1440}
    if minutes not in allowed:
        raise ValueError(
            f"Unsupported interval minutes: {minutes}. "
            "Use one of: 1, 5, 15, 60, 360, 1440."
        )
    return f"{minutes}m"


def cmd_train(args: argparse.Namespace) -> None:
    interval_minutes = _interval_to_minutes(args.interval)
    horizon_minutes = int(args.horizon_minutes)
    if horizon_minutes <= 0:
        raise ValueError("--horizon-minutes must be > 0")
    horizon_bars = _horizon_to_bars(horizon_minutes, interval_minutes)

    if args.verbose:
        print("[train] loading market data...")
    market = load_market_data(symbol=args.symbol, period=args.period, interval=args.interval)
    if args.verbose:
        print(f"[train] market rows: {len(market)}")

    if args.verbose:
        print("[train] loading factors (APIs + optional CSV)...")
    factors = load_optional_factors(args.factors_csv, index=market.index, interval=args.interval)
    if args.verbose:
        print(f"[train] factor rows: {len(factors)}")
    validate_live_factor_coverage(
        factors,
        require=args.require_api_factors,
        min_coverage=0.10,
        allow_recent_for_predict=False,
    )

    if args.verbose:
        print("[train] building features...")
    feat = build_base_features(
        market=market,
        horizon_bars=horizon_bars,
        interval=args.interval,
        period=args.period,
        factors=factors,
        include_equity_context=not args.no_equity_context,
    )

    threshold_z_grid = parse_threshold_z_grid(args.threshold_z_grid)
    if args.verbose:
        print(f"[train] threshold_z_grid={threshold_z_grid}")
    train_df = make_threshold_training_set(feat, threshold_z_grid=threshold_z_grid)

    excluded = {
        "fwd_ret_h",
        "y_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    feature_cols = [c for c in train_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(train_df[c])]
    # Drop sparse features so walk-forward does not collapse to zero usable rows.
    min_cov = 0.30
    dense_feature_cols = [c for c in feature_cols if train_df[c].notna().mean() >= min_cov]
    if "threshold_z" in feature_cols and "threshold_z" not in dense_feature_cols:
        dense_feature_cols.append("threshold_z")
    dropped_sparse = sorted(set(feature_cols) - set(dense_feature_cols))
    feature_cols = dense_feature_cols
    if args.verbose and dropped_sparse:
        print(f"[train] dropped sparse features (<{int(min_cov*100)}% coverage): {dropped_sparse}")

    base_for_splits = train_df.dropna(subset=feature_cols + ["y_event", "threshold_ret", "vol_24"]).copy()
    usable_n = len(base_for_splits)
    train_bars = int(args.train_bars)
    test_bars = int(args.test_bars)
    step_bars = int(args.step_bars)

    # Auto-shrink walk-forward windows when data is shorter than configured defaults.
    if usable_n > 0 and (train_bars + test_bars > usable_n):
        train_bars = max(100, int(usable_n * 0.6))
        test_bars = max(40, int(usable_n * 0.2))
        step_bars = max(20, int(test_bars * 0.5))
        if train_bars + test_bars > usable_n:
            train_bars = max(60, int(usable_n * 0.55))
            test_bars = max(20, int(usable_n * 0.15))
            step_bars = max(10, int(test_bars * 0.5))
        if args.verbose:
            print(
                "[train] auto-adjusted walk-forward windows: "
                f"train_bars={train_bars}, test_bars={test_bars}, step_bars={step_bars}, usable_rows={usable_n}"
            )

    wf = WalkForwardConfig(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
    )
    if args.verbose:
        print("[train] running walk-forward evaluation...")
    oos, metrics = walk_forward_eval(
        train_df,
        feature_cols=feature_cols,
        cfg=wf,
        calibration_method=args.calibration_method,
        verbose=args.verbose,
    )

    if args.verbose:
        print("[train] fitting final model...")
    clf, calibrator = train_final_model(
        train_df,
        feature_cols=feature_cols,
        calibration_method=args.calibration_method,
    )

    artifact = {
        "model": clf,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "symbol": args.symbol,
        "interval": args.interval,
        "period": args.period,
        "horizon": f"{horizon_minutes}m",
        "horizon_minutes": horizon_minutes,
        "horizon_bars": horizon_bars,
        "calibration_method": args.calibration_method,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_old_reports:
        clean_old_reports(out_dir)

    model_stem = Path(args.model_name).stem
    model_path = out_dir / args.model_name
    report_path = out_dir / f"{model_stem}_train_report.json"
    metrics_path = out_dir / f"{model_stem}_walk_forward_metrics.csv"
    preds_path = out_dir / f"{model_stem}_walk_forward_predictions.csv"

    joblib.dump(artifact, model_path)
    metrics.to_csv(metrics_path)
    oos.to_csv(preds_path, index=False)

    report = {
        "symbol": args.symbol,
        "interval": args.interval,
        "horizon": f"{horizon_minutes}m",
        "horizon_bars": horizon_bars,
        "rows_market": int(len(market)),
        "rows_training": int(len(train_df)),
        "features": feature_cols,
        "metrics": metrics.reset_index().to_dict(orient="records"),
        "factors_csv": args.factors_csv,
        "calibration_method": args.calibration_method,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print("=== Walk-Forward Metrics (lower is better) ===")
    print(metrics[["brier", "log_loss", "event_rate", "mean_pred"]])
    print("\nSaved:")
    print(model_path)
    print(report_path)
    print(metrics_path)
    print(preds_path)


def predict_once(
    model_path: str,
    target_price: float,
    current_price: float | None,
    factors_csv: str | None,
    refresh_period: str | None,
    no_equity_context: bool,
    require_api_factors: bool,
    max_staleness_minutes: int,
    verbose: bool,
) -> Dict[str, float | str]:
    artifact = joblib.load(model_path)

    symbol = artifact["symbol"]
    interval = artifact["interval"]
    period = refresh_period or artifact["period"]
    horizon_minutes = artifact["horizon_minutes"]
    horizon_bars = artifact["horizon_bars"]
    feature_cols = artifact["feature_cols"]

    if verbose:
        print("[predict] loading market data...")
    market = load_market_data(
        symbol=symbol,
        period=period,
        interval=interval,
        allow_yahoo_fallback=False,
        strict_live=True,
        max_staleness_minutes=max_staleness_minutes,
    )
    if verbose:
        print(f"[predict] market rows: {len(market)}")

    if verbose:
        print("[predict] loading factors (APIs + optional CSV)...")
    factors = load_optional_factors(factors_csv, index=market.index, interval=interval)
    validate_live_factor_coverage(
        factors,
        require=require_api_factors,
        min_coverage=0.01,
        allow_recent_for_predict=True,
    )
    if require_api_factors:
        validate_recent_live_factors(factors, max_age_minutes=max(10, max_staleness_minutes * 3))

    if verbose:
        print("[predict] building latest features...")
    feat = build_base_features(
        market=market,
        horizon_bars=horizon_bars,
        interval=interval,
        period=period,
        factors=factors,
        include_equity_context=not no_equity_context,
    )

    if feat.empty:
        raise ValueError("No feature rows available for prediction. Check market data source and interval.")

    pred_df = feat.copy()
    for c in feature_cols:
        if c not in pred_df.columns:
            pred_df[c] = np.nan

    # Prefer complete recent rows, but fall back to robust imputation if none exist.
    valid = pred_df.dropna(subset=feature_cols)
    if not valid.empty:
        latest = valid.iloc[-1].copy()
    else:
        latest = pred_df.iloc[-1].copy()
        col_medians = pred_df[feature_cols].median(numeric_only=True)
        for c in feature_cols:
            v = latest.get(c, np.nan)
            if pd.isna(v):
                mv = col_medians.get(c, np.nan)
                latest[c] = 0.0 if pd.isna(mv) else float(mv)
    market_close_price = float(latest["close"])
    if current_price is not None:
        current_price = float(current_price)
    else:
        spot = fetch_coinbase_spot_price()
        current_price = float(spot) if spot is not None else market_close_price
    target_price = float(target_price)

    threshold_ret = float(np.log(target_price / current_price))
    vol = float(max(latest.get("vol_24", np.nan), EPS))
    threshold_z = threshold_ret / vol

    latest["threshold_z"] = threshold_z
    latest["threshold_ret"] = threshold_ret
    latest["threshold_abs"] = abs(threshold_ret)
    latest["threshold_ret_sq"] = threshold_ret ** 2
    latest["threshold_z_abs"] = abs(threshold_z)

    missing = [c for c in feature_cols if c not in latest.index]
    if missing:
        raise ValueError(f"Prediction features missing in latest row: {missing}")

    x_vec = latest[feature_cols].to_numpy(dtype=float)
    x = x_vec.reshape(1, -1)
    clf = artifact["model"]
    calibrator = artifact["calibrator"]
    calibration_method = str(artifact.get("calibration_method", "unknown"))

    p_raw = float(np.clip(clf.predict_proba(x)[:, 1][0], 1e-6, 1 - 1e-6))
    p = float(_apply_calibrator(calibrator, np.array([p_raw], dtype=float))[0])

    debug = build_prediction_debug_payload(
        market=market,
        factors=factors,
        latest=latest,
        feature_cols=feature_cols,
        model_prob_raw=p_raw,
        model_prob_calibrated=p,
        calibration_method=calibration_method,
    )
    baseline_vals = pred_df[feature_cols].median(numeric_only=True).reindex(feature_cols).fillna(0.0).to_numpy(dtype=float)
    debug.update(
        compute_local_feature_influence(
            clf=clf,
            calibrator=calibrator,
            x_row=x_vec,
            feature_cols=feature_cols,
            baseline_vals=baseline_vals,
        )
    )
    # Local target-sensitivity probe: if this is near zero, model may be too flat around current input.
    target_sensitivity = None
    if "threshold_z" in feature_cols:
        idx = feature_cols.index("threshold_z")
        x_up = x_vec.copy()
        x_dn = x_vec.copy()
        x_up[idx] = x_up[idx] + 0.10
        x_dn[idx] = x_dn[idx] - 0.10
        p_up = float(_apply_calibrator(calibrator, clf.predict_proba(x_up.reshape(1, -1))[:, 1])[0])
        p_dn = float(_apply_calibrator(calibrator, clf.predict_proba(x_dn.reshape(1, -1))[:, 1])[0])
        target_sensitivity = float((p_up - p_dn) / 0.20)
    debug["target_sensitivity_per_0p1_z"] = target_sensitivity
    z_abs = abs(float(latest.get("threshold_z", 0.0)))
    debug["low_target_sensitivity_flag"] = bool(
        target_sensitivity is not None and z_abs >= 0.2 and abs(target_sensitivity) < 0.01
    )

    result = {
        "symbol": symbol,
        "timestamp": str(latest.name),
        "interval": interval,
        "horizon_minutes": horizon_minutes,
        "target_price": target_price,
        "current_price": current_price,
        "market_close_price": market_close_price,
        "threshold_return": threshold_ret,
        "threshold_z": threshold_z,
        "probability_price_above_target": p,
        "probability_price_below_or_equal_target": float(1.0 - p),
        "debug": debug,
    }
    if verbose:
        print(f"[predict] current_price={result['current_price']}")
        print(
            "[predict] raw_prob="
            f"{result['debug']['raw_model_probability_above_target']}, "
            "calibrated_prob="
            f"{result['debug']['calibrated_model_probability_above_target']}"
        )
    return result


def cmd_predict(args: argparse.Namespace) -> None:
    result = predict_once(
        model_path=args.model_path,
        target_price=args.target_price,
        current_price=args.current_price,
        factors_csv=args.factors_csv,
        refresh_period=args.refresh_period,
        no_equity_context=args.no_equity_context,
        require_api_factors=args.require_api_factors,
        max_staleness_minutes=args.max_staleness_minutes,
        verbose=args.verbose,
    )
    debug_dir = Path(args.output_file).parent if args.output_file else Path("model_artifacts")
    result.setdefault("debug", {})
    if isinstance(result["debug"], dict):
        result["debug"]["calibration_plateau_flag"] = detect_calibration_plateau(
            debug_dir,
            float(result["probability_price_above_target"]),
        )
    append_prediction_debug_log(debug_dir, result)
    output = json.dumps(result, indent=2)
    print(output)
    print(f"Current price: {result['current_price']}")
    print(f"Debug log updated: {debug_dir / 'prediction_debug_log.csv'}")
    if args.output_file:
        Path(args.output_file).write_text(output + "\n")
        print(f"[predict] wrote output to {args.output_file}")


def cmd_run(args: argparse.Namespace) -> None:
    run_verbose = True
    print("[run] interactive mode started")
    interval_raw = input(f"Candle interval minutes? (default from --interval={args.interval}): ").strip()
    interval = args.interval if interval_raw == "" else parse_interval_minutes_input(interval_raw)
    horizon_minutes_raw = input("How many minutes is the forecast? (e.g. 15, 60, 120): ")
    horizon_minutes = parse_horizon_minutes(horizon_minutes_raw)
    target_price_raw = input("What BTC target price do you want to test? (float allowed): ")
    target_price = parse_target_price_input(target_price_raw)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_old_reports:
        clean_old_reports(model_dir)
    tag = f"{_interval_tag(interval)}_{_horizon_tag_minutes(horizon_minutes)}"
    model_name = f"btc_prob_model_{tag}.joblib"
    model_path = model_dir / model_name
    generic_model_path = model_dir / "btc_prob_model.joblib"

    if (not model_path.exists()) and generic_model_path.exists():
        try:
            generic_artifact = joblib.load(generic_model_path)
            if (
                int(generic_artifact.get("horizon_minutes", -1)) == horizon_minutes
                and str(generic_artifact.get("interval", "")).lower() == interval.lower()
            ):
                model_path.write_bytes(generic_model_path.read_bytes())
                print(f"[run] reused existing model: {generic_model_path}")
        except Exception:
            pass

    if not model_path.exists():
        print(f"[run] no model found for horizon={horizon_minutes} minutes. Training now...")
        train_args = argparse.Namespace(
            symbol=args.symbol,
            period=args.period,
            interval=interval,
            horizon_minutes=horizon_minutes,
            factors_csv=args.factors_csv,
            no_equity_context=args.no_equity_context,
            verbose=run_verbose,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            out_dir=str(model_dir),
            model_name=model_name,
            keep_old_reports=args.keep_old_reports,
            threshold_z_grid=args.threshold_z_grid,
            require_api_factors=args.require_api_factors,
            calibration_method="platt",
        )
        try:
            cmd_train(train_args)
        except ValueError as e:
            msg = str(e)
            if args.require_api_factors and "very low coverage" in msg:
                fallback_period = "60d"
                print(
                    "[run] live factor coverage too low for current training period. "
                    f"Retrying with shorter period={fallback_period}..."
                )
                train_args.period = fallback_period
                cmd_train(train_args)
            else:
                raise

    result = predict_once(
        model_path=str(model_path),
        target_price=target_price,
        current_price=None,
        factors_csv=args.factors_csv,
        refresh_period=args.refresh_period,
        no_equity_context=args.no_equity_context,
        require_api_factors=args.require_api_factors,
        max_staleness_minutes=args.max_staleness_minutes,
        verbose=run_verbose,
    )

    output_file = model_dir / "latest_probability_output.json"
    result.setdefault("debug", {})
    if isinstance(result["debug"], dict):
        result["debug"]["calibration_plateau_flag"] = detect_calibration_plateau(
            model_dir,
            float(result["probability_price_above_target"]),
        )
    output = json.dumps(result, indent=2)
    output_file.write_text(output + "\n")
    append_prediction_debug_log(model_dir, result)
    print("\n=== Probability Output ===")
    print(output)
    print(f"Current price: {result['current_price']}")
    print(f"Debug log updated: {model_dir / 'prediction_debug_log.csv'}")
    print(f"\nSaved output file: {output_file}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/predict BTC threshold probability model")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train model and produce walk-forward metrics")
    tr.add_argument("--symbol", type=str, default="BTC-USD")
    tr.add_argument("--period", type=str, default="730d")
    tr.add_argument("--interval", type=str, default="1h")
    tr.add_argument("--horizon-minutes", type=int, default=60, help="Forecast horizon in minutes, e.g. 15, 60")
    tr.add_argument("--factors-csv", type=str, default=None, help="Optional timestamped external factors CSV")
    tr.add_argument("--no-equity-context", action="store_true", help="Skip NQ/ES context download")
    tr.add_argument("--verbose", action="store_true", help="Print progress logs")
    tr.add_argument("--require-api-factors", action="store_true", help="Fail if live API factor coverage is poor")
    tr.add_argument(
        "--calibration-method",
        type=str,
        default="platt",
        choices=["platt", "isotonic", "none"],
        help="Probability calibration method",
    )
    tr.add_argument(
        "--threshold-z-grid",
        type=str,
        default="-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0",
        help="Comma-separated threshold z grid; fewer points run faster",
    )

    tr.add_argument("--train-bars", type=int, default=24 * 90)
    tr.add_argument("--test-bars", type=int, default=24 * 14)
    tr.add_argument("--step-bars", type=int, default=24 * 7)

    tr.add_argument("--out-dir", type=str, default="model_artifacts")
    tr.add_argument("--model-name", type=str, default="btc_prob_model.joblib")
    tr.add_argument("--keep-old-reports", action="store_true", help="Keep existing report/output files")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Predict probability price will exceed target at horizon")
    pr.add_argument("--model-path", type=str, default="model_artifacts/btc_prob_model.joblib")
    pr.add_argument("--target-price", type=float, required=True)
    pr.add_argument("--current-price", type=float, default=None, help="Optional override; default latest close")
    pr.add_argument("--factors-csv", type=str, default=None, help="Optional latest external factors CSV")
    pr.add_argument("--refresh-period", type=str, default=None, help="Optional fresh fetch period, e.g. 60d")
    pr.add_argument("--no-equity-context", action="store_true", help="Skip NQ/ES context download")
    pr.add_argument("--verbose", action="store_true", help="Print progress logs")
    pr.add_argument("--require-api-factors", action="store_true", help="Fail if live API factor coverage is poor")
    pr.add_argument("--max-staleness-minutes", type=int, default=5, help="Maximum allowed age of latest candle")
    pr.add_argument("--output-file", type=str, default=None, help="Optional JSON output path")
    pr.set_defaults(func=cmd_predict)

    rn = sub.add_parser("run", help="Interactive mode: asks horizon and target price, writes one output file")
    rn.add_argument("--symbol", type=str, default="BTC-USD")
    rn.add_argument("--period", type=str, default="730d")
    rn.add_argument("--interval", type=str, default="1h")
    rn.add_argument("--factors-csv", type=str, default=None, help="Optional timestamped external factors CSV")
    rn.add_argument("--refresh-period", type=str, default=None, help="Optional fresh fetch period for prediction")
    rn.add_argument("--model-dir", type=str, default="model_artifacts")
    rn.add_argument("--train-bars", type=int, default=24 * 90)
    rn.add_argument("--test-bars", type=int, default=24 * 14)
    rn.add_argument("--step-bars", type=int, default=24 * 7)
    rn.add_argument("--no-equity-context", action="store_true", help="Skip NQ/ES context download")
    rn.add_argument("--verbose", action="store_true", help="Print progress logs")
    rn.add_argument("--require-api-factors", action="store_true", help="Fail if live API factor coverage is poor")
    rn.add_argument("--max-staleness-minutes", type=int, default=5, help="Maximum allowed age of latest candle")
    rn.add_argument("--keep-old-reports", action="store_true", help="Keep existing report/output files")
    rn.add_argument(
        "--threshold-z-grid",
        type=str,
        default="-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0",
        help="Comma-separated threshold z grid; fewer points run faster",
    )
    rn.set_defaults(func=cmd_run)

    return p


def main() -> None:
    load_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
