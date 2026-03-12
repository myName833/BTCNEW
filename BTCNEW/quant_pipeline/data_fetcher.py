from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import ensure_dir, read_csv_time_index

COINBASE_BASE = "https://api.exchange.coinbase.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"


def _fetch_coinbase_candles(
    product: str,
    start: datetime,
    end: datetime,
    granularity_seconds: int,
    headers: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    max_points = 300
    chunk_seconds = granularity_seconds * max_points
    cur = start
    rows: List[List[float]] = []

    while cur < end:
        nxt = min(cur + timedelta(seconds=chunk_seconds), end)
        params = {
            "start": cur.isoformat().replace("+00:00", "Z"),
            "end": nxt.isoformat().replace("+00:00", "Z"),
            "granularity": str(granularity_seconds),
        }
        resp = requests.get(f"{COINBASE_BASE}/products/{product}/candles", params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
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


def fetch_coinbase_ohlcv(
    product: str,
    start: str,
    end: str,
    candle_minutes: int,
    out_path: Path,
) -> Path:
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("Invalid start/end date")
    if end_dt <= start_dt:
        raise ValueError("End must be after start")

    granularity = candle_minutes * 60
    df = _fetch_coinbase_candles(product, start_dt.to_pydatetime(), end_dt.to_pydatetime(), granularity, headers=None)
    ensure_dir(out_path.parent)
    df.reset_index().to_csv(out_path, index=False)
    return out_path


def load_or_fetch_coinbase(
    product: str,
    start: str,
    end: str,
    candle_minutes: int,
    out_dir: Path,
) -> pd.DataFrame:
    fname = f"coinbase_{product}_{start}_{end}_{candle_minutes}m.csv".replace(":", "")
    path = out_dir / fname
    if path.exists():
        return read_csv_time_index(path)
    fetch_coinbase_ohlcv(product, start, end, candle_minutes, path)
    return read_csv_time_index(path)


def fetch_derivatives_placeholders(
    start: str,
    end: str,
    candle_minutes: int,
) -> pd.DataFrame:
    # Placeholder for funding rate / OI / liquidations / basis.
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    idx = pd.date_range(start=start_dt, end=end_dt, freq=f"{candle_minutes}min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["funding_rate"] = 0.0
    df["open_interest_change"] = 0.0
    df["liquidation_pressure"] = 0.0
    df["liquidation_imbalance"] = 0.0
    df["basis_spread"] = 0.0
    df["perp_premium"] = 0.0
    df["long_short_ratio"] = 0.0
    df["volume_delta"] = 0.0
    df["orderbook_imbalance"] = 0.0
    return df


def _interval_for_minutes(minutes: int) -> str:
    if minutes <= 1:
        return "1min"
    if minutes <= 5:
        return "5min"
    if minutes <= 15:
        return "15min"
    if minutes <= 30:
        return "30min"
    return "1hour"


def _coinalyze_history(payload: object) -> pd.DataFrame:
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


def fetch_derivatives_coinalyze(
    start: str,
    end: str,
    candle_minutes: int,
    symbol: str,
    api_key: str,
) -> pd.DataFrame:
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    idx = pd.date_range(start=start_dt, end=end_dt, freq=f"{candle_minutes}min", tz="UTC")
    out = pd.DataFrame(index=idx)

    headers = {"api_key": api_key}
    interval = _interval_for_minutes(candle_minutes)
    params = {
        "symbols": symbol,
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
        "interval": interval,
    }

    try:
        fr = requests.get(f"{COINALYZE_BASE}/funding-rate-history", params=params, headers=headers, timeout=10).json()
        fr_df = _coinalyze_history(fr)
        if not fr_df.empty:
            fr_df["t"] = pd.to_datetime(fr_df["t"], unit="s", utc=True)
            out = out.join(fr_df.set_index("t")["c"].rename("funding_rate"), how="left")
    except Exception:
        out["funding_rate"] = 0.0

    try:
        oi = requests.get(f"{COINALYZE_BASE}/open-interest-history", params=params, headers=headers, timeout=10).json()
        oi_df = _coinalyze_history(oi)
        if not oi_df.empty:
            oi_df["t"] = pd.to_datetime(oi_df["t"], unit="s", utc=True)
            oi_series = oi_df.set_index("t")["c"].rename("open_interest")
            out = out.join(oi_series, how="left")
            out["open_interest_change"] = out["open_interest"].pct_change().fillna(0.0)
        else:
            out["open_interest_change"] = 0.0
    except Exception:
        out["open_interest_change"] = 0.0

    try:
        liq = requests.get(f"{COINALYZE_BASE}/liquidation-history", params=params, headers=headers, timeout=10).json()
        liq_df = _coinalyze_history(liq)
        if not liq_df.empty:
            liq_df["t"] = pd.to_datetime(liq_df["t"], unit="s", utc=True)
            long_col = next((c for c in ["long_liquidation_usd", "longs", "long", "l"] if c in liq_df.columns), None)
            short_col = next((c for c in ["short_liquidation_usd", "shorts", "short", "s"] if c in liq_df.columns), None)
            if long_col and short_col:
                longs = pd.to_numeric(liq_df[long_col], errors="coerce").fillna(0.0)
                shorts = pd.to_numeric(liq_df[short_col], errors="coerce").fillna(0.0)
                liq_df["liquidation_pressure"] = (shorts - longs) / (shorts + longs + 1e-12)
                liq_df["liquidation_imbalance"] = (longs - shorts) / (shorts + longs + 1e-12)
                out = out.join(liq_df.set_index("t")[["liquidation_pressure", "liquidation_imbalance"]], how="left")
        if "liquidation_pressure" not in out.columns:
            out["liquidation_pressure"] = 0.0
        if "liquidation_imbalance" not in out.columns:
            out["liquidation_imbalance"] = 0.0
    except Exception:
        out["liquidation_pressure"] = 0.0
        out["liquidation_imbalance"] = 0.0

    try:
        pm = requests.get(f"{COINALYZE_BASE}/premium-index-history", params=params, headers=headers, timeout=10).json()
        pm_df = _coinalyze_history(pm)
        if not pm_df.empty:
            pm_df["t"] = pd.to_datetime(pm_df["t"], unit="s", utc=True)
            out = out.join(pm_df.set_index("t")["c"].rename("basis_spread"), how="left")
            out["perp_premium"] = out["basis_spread"]
        else:
            out["basis_spread"] = 0.0
    except Exception:
        out["basis_spread"] = 0.0
        out["perp_premium"] = 0.0

    # Optional placeholders (not provided by coinalyze endpoints used above)
    if "long_short_ratio" not in out.columns:
        out["long_short_ratio"] = 0.0
    if "volume_delta" not in out.columns:
        out["volume_delta"] = 0.0
    if "orderbook_imbalance" not in out.columns:
        out["orderbook_imbalance"] = 0.0

    out = out.fillna(0.0)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Coinbase OHLCV data")
    parser.add_argument("--product", default="BTC-USD", help="Coinbase product id")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--candle-minutes", default=1, type=int, help="Candle interval minutes")
    parser.add_argument("--out-dir", default="BTCNEW/artifacts/data", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    load_or_fetch_coinbase(args.product, args.start, args.end, int(args.candle_minutes), out_dir)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
