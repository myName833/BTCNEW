from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.data_fetcher import fetch_derivatives_placeholders, fetch_derivatives_coinalyze, load_or_fetch_coinbase
from quant_pipeline.feature_engineering import build_feature_set, feature_columns, normalize_features
from quant_pipeline.utils import ensure_dir, read_csv_time_index


def create_targets(df: pd.DataFrame, horizons: List[int], candle_minutes: int, move_threshold: float) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        bars = max(1, int(round(h / candle_minutes)))
        out[f"future_return_{h}m"] = out["close"].shift(-bars) / out["close"] - 1.0
        out[f"target_cls_{h}m"] = (out[f"future_return_{h}m"] > 0).astype(int)
        out[f"target_dir_{h}m"] = 0
        out.loc[out[f"future_return_{h}m"] > move_threshold, f"target_dir_{h}m"] = 1
        out.loc[out[f"future_return_{h}m"] < -move_threshold, f"target_dir_{h}m"] = -1
    return out


def build_dataset(
    start: str,
    end: str,
    candle_minutes: int,
    horizons: List[int],
    product: str,
    out_dir: Path,
    move_threshold: float,
) -> Path:
    ensure_dir(out_dir)
    ohlcv = load_or_fetch_coinbase(product, start, end, candle_minutes, out_dir)
    api_key = os.getenv("COINALYZE_API_KEY", "").strip()
    symbol = os.getenv("COINALYZE_SYMBOL", "BTCUSDT_PERP.A").strip()
    if api_key:
        deriv = fetch_derivatives_coinalyze(start, end, candle_minutes, symbol, api_key)
    else:
        deriv = fetch_derivatives_placeholders(start, end, candle_minutes)

    df = ohlcv.join(deriv, how="left").fillna(0.0)
    df = build_feature_set(df)
    feats = feature_columns(df)
    df = normalize_features(df, feats, window=200)
    df = create_targets(df, horizons, candle_minutes, move_threshold)

    # Drop rows with future info leakage for the max horizon
    max_h = max(horizons) if horizons else 0
    max_bars = max(1, int(round(max_h / candle_minutes))) if max_h else 0
    if max_bars > 0:
        df = df.iloc[:-max_bars]

    df = df.dropna()
    out_path = out_dir / f"dataset_{product}_{start}_{end}_{candle_minutes}m.csv".replace(":", "")
    df.to_csv(out_path, index=True)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ML dataset with features/targets")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--candle-minutes", default=1, type=int, help="Candle interval minutes")
    parser.add_argument("--horizons", default="5,15,30,60", help="Prediction horizons in minutes")
    parser.add_argument("--product", default="BTC-USD", help="Coinbase product id")
    parser.add_argument("--move-threshold", default=0.002, type=float, help="Directional target threshold (e.g. 0.002 = 0.2%)")
    parser.add_argument("--out-dir", default="BTCNEW/artifacts/data", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    path = build_dataset(
        args.start,
        args.end,
        int(args.candle_minutes),
        horizons,
        args.product,
        out_dir,
        float(args.move_threshold),
    )
    print("Saved dataset:", path)


if __name__ == "__main__":
    main()
