from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import read_csv_time_index, write_json


def _signal_from_model(
    model: object,
    X,
    expected_return_threshold: float,
    min_confidence: float,
    min_signal_strength: float,
    realized_vol: float,
    prob_threshold: float,
    strength_threshold: Optional[float],
) -> Tuple[str, float, float, float]:
    # Classification: use probability ONLY for direction. Regression: use predicted return.
    if hasattr(model, "predict_proba"):
        X_in = X if getattr(X, "ndim", 1) == 2 else X.reshape(1, -1)
        prob_up = float(model.predict_proba(X_in)[:, 1][0])
        expected_return = 0.0
        confidence = prob_up
        signal_strength = abs(prob_up - 0.5)
        strength_gate = strength_threshold if strength_threshold is not None else min_signal_strength
        direction = "NO_TRADE"
        if confidence >= prob_threshold and signal_strength >= strength_gate:
            direction = "LONG"
        elif confidence <= (1.0 - prob_threshold) and signal_strength >= strength_gate:
            direction = "SHORT"
        return direction, expected_return, confidence, signal_strength

    X_in = X if getattr(X, "ndim", 1) == 2 else X.reshape(1, -1)
    expected_return = float(model.predict(X_in)[0])
    confidence = float(min(1.0, max(0.0, abs(expected_return) / (realized_vol + 1e-8))))
    prob_up = float(min(1.0, max(0.0, 0.5 + expected_return / (2.0 * max(realized_vol, 1e-6)))))
    signal_strength = expected_return / (realized_vol + 1e-8)
    strength_gate = strength_threshold if strength_threshold is not None else min_signal_strength
    direction = "NO_TRADE"
    if expected_return > expected_return_threshold and confidence >= min_confidence and abs(signal_strength) >= strength_gate:
        if prob_up >= prob_threshold:
            direction = "LONG"
        elif prob_up <= (1.0 - prob_threshold):
            direction = "SHORT"
    elif expected_return < -expected_return_threshold and confidence >= min_confidence and abs(signal_strength) >= strength_gate:
        if prob_up <= (1.0 - prob_threshold):
            direction = "SHORT"
        elif prob_up >= prob_threshold:
            direction = "LONG"

    return direction, expected_return, confidence, signal_strength


def _resolve_intrabar(
    direction: str,
    take_profit: float,
    stop_loss: float,
    liquidation: float,
    intrabar: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    if intrabar is None or intrabar.empty:
        return "SL_HIT"  # conservative fallback

    window = intrabar.loc[start:end]
    if window.empty:
        return "SL_HIT"

    for _, row in window.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        if direction == "LONG":
            tp_hit = hi >= take_profit
            sl_hit = lo <= stop_loss
            liq_hit = lo <= liquidation
        else:
            tp_hit = lo <= take_profit
            sl_hit = hi >= stop_loss
            liq_hit = hi >= liquidation

        if liq_hit:
            return "LIQUIDATION"
        if sl_hit and tp_hit:
            return "SL_HIT"
        if sl_hit:
            return "SL_HIT"
        if tp_hit:
            return "TP_HIT"

    return "TIME_EXIT"


def backtest(
    df: pd.DataFrame,
    model: object,
    feature_cols: list[str],
    horizon_minutes: int,
    candle_minutes: int,
    intrabar_1m: Optional[pd.DataFrame],
    leverage: float,
    fee_per_side: float,
    slippage: float,
    expected_return_threshold: float,
    min_confidence: float,
    min_signal_strength: float,
    prob_threshold: float,
    strength_quantile: Optional[float],
    stop_loss_mult: float,
    take_profit_mult: float,
) -> pd.DataFrame:
    trades = []
    equity = 1.0
    hold_candles = max(1, int(round(horizon_minutes / candle_minutes)))
    lookback = 240

    idx = df.index
    pos = max(lookback, 1)
    last = len(idx) - 1

    # Compute signal strength quantile threshold (train window only) if requested
    strength_threshold = None
    if strength_quantile is not None:
        sample = df.iloc[: max(lookback, int(len(df) * 0.70))]
        strengths = []
        for i in range(lookback, len(sample), 5):
            row = sample.iloc[i]
            X = row[feature_cols].to_frame().T
            realized_vol = float(row["realized_vol_30"]) if "realized_vol_30" in row else 0.001
            _, er, conf, s = _signal_from_model(
                model,
                X,
                expected_return_threshold,
                min_confidence,
                min_signal_strength,
                realized_vol,
                prob_threshold,
                None,
            )
            strengths.append(abs(s))
        if strengths:
            strength_threshold = float(np.quantile(strengths, strength_quantile))

    while pos + 1 < last:
        window = df.iloc[max(0, pos - lookback) : pos + 1]
        if window.empty:
            pos += 1
            continue

        X = window[feature_cols].iloc[[-1]]
        realized_vol = float(window["realized_vol_30"].iloc[-1]) if "realized_vol_30" in window.columns else 0.001

        direction, expected_return, confidence, strength = _signal_from_model(
            model,
            X,
            expected_return_threshold,
            min_confidence,
            min_signal_strength,
            realized_vol,
            prob_threshold,
            strength_threshold,
        )
        if direction == "NO_TRADE":
            pos += 1
            continue

        entry_pos = pos + 1
        if entry_pos >= last:
            break
        entry_time = idx[entry_pos]
        entry_price = float(df["open"].iloc[entry_pos])
        entry_price = entry_price * (1.0 + slippage) if direction == "LONG" else entry_price * (1.0 - slippage)

        atr = float(window["atr_14"].iloc[-1]) if "atr_14" in window.columns else entry_price * realized_vol
        atr_pct = atr / entry_price if entry_price else realized_vol
        take_profit = entry_price * (1.0 + take_profit_mult * atr_pct) if direction == "LONG" else entry_price * (1.0 - take_profit_mult * atr_pct)
        stop_loss = entry_price * (1.0 - stop_loss_mult * atr_pct) if direction == "LONG" else entry_price * (1.0 + stop_loss_mult * atr_pct)
        liquidation = entry_price * (1.0 - 1.0 / leverage) if direction == "LONG" else entry_price * (1.0 + 1.0 / leverage)

        end_pos = min(entry_pos + hold_candles, last)
        future = df.iloc[entry_pos : end_pos + 1]

        exit_reason = "TIME_EXIT"
        exit_price = float(future["close"].iloc[-1])
        exit_time = future.index[-1]

        for ts, row in future.iterrows():
            hi = float(row["high"])
            lo = float(row["low"])
            if direction == "LONG":
                tp_hit = hi >= take_profit
                sl_hit = lo <= stop_loss
                liq_hit = lo <= liquidation
            else:
                tp_hit = lo <= take_profit
                sl_hit = hi >= stop_loss
                liq_hit = hi >= liquidation

            if (tp_hit and sl_hit) or (liq_hit and (tp_hit or sl_hit)):
                exit_reason = _resolve_intrabar(
                    direction,
                    take_profit,
                    stop_loss,
                    liquidation,
                    intrabar_1m,
                    ts,
                    ts + pd.Timedelta(minutes=candle_minutes),
                )
                if exit_reason == "LIQUIDATION":
                    exit_price = liquidation
                elif exit_reason == "SL_HIT":
                    exit_price = stop_loss
                elif exit_reason == "TP_HIT":
                    exit_price = take_profit
                else:
                    exit_price = float(row["close"])
                exit_time = ts
                break
            if liq_hit:
                exit_reason = "LIQUIDATION"
                exit_price = liquidation
                exit_time = ts
                break
            if tp_hit:
                exit_reason = "TP_HIT"
                exit_price = take_profit
                exit_time = ts
                break
            if sl_hit:
                exit_reason = "SL_HIT"
                exit_price = stop_loss
                exit_time = ts
                break

        if direction == "LONG":
            price_return = (exit_price - entry_price) / entry_price
        else:
            price_return = (entry_price - exit_price) / entry_price
        leveraged_return = price_return * leverage
        pnl = leveraged_return - (fee_per_side * 2.0)
        if exit_reason == "LIQUIDATION":
            pnl = -1.0
        pnl = max(pnl, -1.0)

        trades.append(
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "liquidation_price": liquidation,
                "exit_reason": exit_reason,
                "expected_return": expected_return,
                "confidence": confidence,
                "signal_strength": strength,
                "pnl": pnl,
            }
        )

        equity *= (1.0 + pnl)
        if equity <= 0:
            break

        pos = end_pos + 1

    return pd.DataFrame(trades)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest engine")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--intrabar-1m", help="Optional 1m candles CSV for intrabar TP/SL ordering")
    parser.add_argument("--horizon", default=30, type=int, help="Holding period minutes")
    parser.add_argument("--candle-minutes", default=5, type=int, help="Candle minutes")
    parser.add_argument("--leverage", default=100.0, type=float, help="Leverage")
    parser.add_argument("--fee", default=0.0004, type=float, help="Fee per side")
    parser.add_argument("--slippage", default=0.0, type=float, help="Slippage per side")
    parser.add_argument("--return-threshold", default=0.0008, type=float, help="Expected return threshold")
    parser.add_argument("--min-confidence", default=0.55, type=float, help="Min confidence (probability)")
    parser.add_argument("--min-signal-strength", default=0.15, type=float, help="Min signal strength")
    parser.add_argument("--prob-threshold", default=0.55, type=float, help="Probability threshold for long/short")
    parser.add_argument("--strength-quantile", default=0.90, type=float, help="Quantile filter for signal strength (0-1)")
    parser.add_argument("--tp-mult", default=1.2, type=float, help="TP multiplier")
    parser.add_argument("--sl-mult", default=0.7, type=float, help="SL multiplier")
    parser.add_argument("--out", default="artifacts/backtest/backtest_trades.csv", help="Output CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    intrabar_1m = None
    if args.intrabar_1m:
        intrabar_1m = read_csv_time_index(Path(args.intrabar_1m))
    # Prefer feature list saved during training to avoid feature-count mismatches
    feature_cols = None
    model_path = Path(args.model)
    feature_path = model_path.with_name(model_path.stem + "_features.json")
    for p in (feature_path,):
        if p and p.exists():
            try:
                import json as _json

                with p.open("r") as f:
                    data = _json.load(f)
                if isinstance(data, dict) and "features" in data:
                    feature_cols = list(data["features"])
                    break
            except Exception:
                feature_cols = None

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if not c.startswith("future_return_")
            and not c.startswith("target_")
        ]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"Warning: {len(missing)} missing features in dataset. Dropping: {missing[:5]}")
            feature_cols = [c for c in feature_cols if c in df.columns]
    model_path = Path(args.model)
    model_dir = model_path.parent if model_path.parent.exists() else Path("BTCNEW/artifacts/models")
    if not model_path.exists():
        target_prefix = model_path.stem.split("_")[0]
        candidates = [
            model_dir / f"{target_prefix}_lightgbm.joblib",
            model_dir / f"{target_prefix}_xgboost.joblib",
            model_dir / f"{target_prefix}_rf.joblib",
            model_dir / f"{target_prefix}_logreg.joblib",
            model_dir / f"{target_prefix}_calibrated.joblib",
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break
        if not model_path.exists():
            # fallback: first target_cls model
            any_model = next(model_dir.glob("target_cls_*_*.joblib"), None)
            if any_model is not None:
                model_path = any_model
    if not model_path.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = joblib.load(model_path)
    print(f"Using model: {model_path}")
    trades = backtest(
        df=df,
        model=model,
        feature_cols=feature_cols,
        horizon_minutes=int(args.horizon),
        candle_minutes=int(args.candle_minutes),
        intrabar_1m=intrabar_1m,
        leverage=float(args.leverage),
        fee_per_side=float(args.fee),
        slippage=float(args.slippage),
        expected_return_threshold=float(args.return_threshold),
        min_confidence=float(args.min_confidence),
        min_signal_strength=float(args.min_signal_strength),
        prob_threshold=float(args.prob_threshold),
        strength_quantile=float(args.strength_quantile) if args.strength_quantile else None,
        stop_loss_mult=float(args.sl_mult),
        take_profit_mult=float(args.tp_mult),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
