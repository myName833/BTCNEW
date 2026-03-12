from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from quant_pipeline.utils import read_csv_time_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC futures walk-forward with regime filtering")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--model", required=True, help="Base classifier (.joblib)")
    parser.add_argument("--target", default="target_cls_30m", help="Target column")
    parser.add_argument("--leverage", default=10.0, type=float, help="Leverage for futures backtest")
    parser.add_argument("--fee", default=0.0004, type=float, help="Fee per side")
    parser.add_argument("--slippage", default=0.0001, type=float, help="Slippage per side")
    parser.add_argument("--horizon", default=30, type=int, help="Holding period in minutes")
    parser.add_argument("--candle-minutes", default=5, type=int, help="Candle interval minutes")
    parser.add_argument("--trend-threshold", default=0.0015, type=float, help="EMA trend threshold (fraction)")
    parser.add_argument("--prob-long", default=0.60, type=float, help="Prob threshold for LONG")
    parser.add_argument("--prob-short", default=0.40, type=float, help="Prob threshold for SHORT")
    parser.add_argument("--out", default="BTCNEW/artifacts/walkforward_futures.json", help="Output JSON")
    return parser


def _backtest_directional(
    df: pd.DataFrame,
    probs: np.ndarray,
    leverage: float,
    fee: float,
    slippage: float,
    hold_candles: int,
    mode: str,
) -> Dict[str, float]:
    idx = df.index
    pos = 0
    last = len(df) - 1
    trades = []
    while pos + 1 < last:
        p = probs[pos]
        row = df.iloc[pos]

        # Regime filters
        if row.get("regime_chop", False):
            pos += 1
            continue

        direction = None
        if p > 0.60:
            direction = "LONG"
        elif p < 0.40:
            direction = "SHORT"

        if direction is None:
            pos += 1
            continue

        if mode == "long_only" and direction != "LONG":
            pos += 1
            continue
        if mode == "short_only" and direction != "SHORT":
            pos += 1
            continue
        if mode == "long_bull" and not row.get("regime_bull", False):
            pos += 1
            continue
        if mode == "short_bear" and not row.get("regime_bear", False):
            pos += 1
            continue

        entry_pos = pos + 1
        if entry_pos >= last:
            break
        entry_price = float(df["open"].iloc[entry_pos])
        entry_price = entry_price * (1.0 + slippage) if direction == "LONG" else entry_price * (1.0 - slippage)

        liq_price = entry_price * (1.0 - 1.0 / leverage) if direction == "LONG" else entry_price * (1.0 + 1.0 / leverage)

        end_pos = min(entry_pos + hold_candles, last)
        exit_price = float(df["close"].iloc[end_pos])
        exit_reason = "TIME_EXIT"

        window = df.iloc[entry_pos : end_pos + 1]
        for _, r in window.iterrows():
            hi = float(r["high"])
            lo = float(r["low"])
            if direction == "LONG":
                liq_hit = lo <= liq_price
            else:
                liq_hit = hi >= liq_price
            if liq_hit:
                exit_reason = "LIQUIDATION"
                exit_price = liq_price
                break

        if direction == "LONG":
            price_return = (exit_price - entry_price) / entry_price
        else:
            price_return = (entry_price - exit_price) / entry_price

        pnl = price_return * leverage - (fee * 2.0)
        pnl = max(pnl, -1.0)

        trades.append({"direction": direction, "pnl": pnl, "exit_reason": exit_reason})
        pos = end_pos + 1

    if not trades:
        return {"trades": 0, "win_rate": None, "avg_pnl": None, "profit_factor": None, "max_drawdown": None}

    tdf = pd.DataFrame(trades)
    pnl = tdf["pnl"]
    profits = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    profit_factor = float(profits / abs(losses)) if losses != 0 else float("inf")
    win_rate = float((pnl > 0).mean())
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return {
        "trades": int(len(tdf)),
        "win_rate": win_rate,
        "avg_pnl": float(pnl.mean()),
        "profit_factor": profit_factor,
        "max_drawdown": float(drawdown.min()),
    }


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))

    # Build regime flags if not present
    if "ema20_raw" in df.columns and "ema100_raw" in df.columns and "adx14_raw" in df.columns:
        close = df["close"]
        trend_strength = (df["ema20_raw"] - df["ema100_raw"]).abs() / (close + 1e-8)
        df["regime_trend"] = (df["adx14_raw"] > 20) & (trend_strength > float(args.trend_threshold))
        df["regime_chop"] = df["adx14_raw"] < 15
        df["regime_bull"] = df["regime_trend"] & (df["ema20_raw"] > df["ema100_raw"])
        df["regime_bear"] = df["regime_trend"] & (df["ema20_raw"] < df["ema100_raw"])
    else:
        df["regime_trend"] = False
        df["regime_chop"] = False
        df["regime_bull"] = False
        df["regime_bear"] = False

    feature_cols = [c for c in df.columns if not c.startswith("future_return_") and not c.startswith("target_")]
    df = df.dropna(subset=feature_cols + [args.target])

    n = len(df)
    folds = [
        ("roll_60_20", 0, int(n * 0.60), int(n * 0.80)),
        ("roll_80_20", 0, int(n * 0.80), n),
    ]

    hold_candles = max(1, int(round(int(args.horizon) / int(args.candle_minutes))))

    results: List[Dict[str, object]] = []
    for name, start, train_end, test_end in folds:
        train = df.iloc[start:train_end]
        test = df.iloc[train_end:test_end]

        base_model = joblib.load(args.model)
        base_model.fit(train[feature_cols], train[args.target].values)
        iso = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
        iso.fit(train[feature_cols], train[args.target].values)
        probs = iso.predict_proba(test[feature_cols])[:, 1]

        fold_result = {
            "fold": name,
            "train_start": str(train.index[0]),
            "train_end": str(train.index[-1]),
            "test_start": str(test.index[0]),
            "test_end": str(test.index[-1]),
            "long_short": _backtest_directional(
                test, probs, float(args.leverage), float(args.fee), float(args.slippage), hold_candles, "long_short"
            ),
            "long_only": _backtest_directional(
                test, probs, float(args.leverage), float(args.fee), float(args.slippage), hold_candles, "long_only"
            ),
            "short_only": _backtest_directional(
                test, probs, float(args.leverage), float(args.fee), float(args.slippage), hold_candles, "short_only"
            ),
            "long_bull": _backtest_directional(
                test, probs, float(args.leverage), float(args.fee), float(args.slippage), hold_candles, "long_bull"
            ),
            "short_bear": _backtest_directional(
                test, probs, float(args.leverage), float(args.fee), float(args.slippage), hold_candles, "short_bear"
            ),
        }
        results.append(fold_result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
