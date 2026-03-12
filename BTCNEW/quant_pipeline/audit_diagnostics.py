from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.backtest_engine import backtest, _signal_from_model
from quant_pipeline.utils import read_csv_time_index


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("time",) and not c.startswith("future_return_") and not c.startswith("target_cls_")]


def _load_model(path: Path) -> object:
    if path.exists():
        return joblib.load(path)
    model_dir = path.parent if path.parent.exists() else Path("BTCNEW/artifacts/models")
    any_model = next(model_dir.glob("target_cls_*_*.joblib"), None)
    if any_model is None:
        raise SystemExit("No model found for diagnostics.")
    return joblib.load(any_model)


def _metrics_from_trades(trades: pd.DataFrame) -> Dict[str, float]:
    non_amb = trades[trades["exit_reason"] != "AMBIGUOUS"].copy()
    total = len(non_amb)
    tp = int((non_amb["exit_reason"] == "TP_HIT").sum())
    sl = int((non_amb["exit_reason"] == "SL_HIT").sum())
    liq = int((non_amb["exit_reason"] == "LIQUIDATION").sum())
    time_exit = int((non_amb["exit_reason"] == "TIME_EXIT").sum())
    ambiguous = int((trades["exit_reason"] == "AMBIGUOUS").sum())
    strict_win = tp / total if total else 0.0
    prof_win = float((non_amb["pnl"] > 0).mean()) if total else 0.0
    avg_return = float(non_amb["pnl"].mean()) if total else 0.0
    profit = float(non_amb.loc[non_amb["pnl"] > 0, "pnl"].sum())
    loss = float(-non_amb.loc[non_amb["pnl"] < 0, "pnl"].sum())
    profit_factor = profit / loss if loss > 0 else 0.0
    equity = (1.0 + non_amb["pnl"]).cumprod()
    if equity.empty:
        max_dd = 0.0
    else:
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(drawdown.min())
    return {
        "total_trades": total,
        "tp": tp,
        "sl": sl,
        "liquidation": liq,
        "time_exit": time_exit,
        "ambiguous": ambiguous,
        "strict_win_rate": strict_win,
        "profitability_win_rate": prof_win,
        "average_return": avg_return,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


def _prediction_quality(df: pd.DataFrame, model: object, target_col: str) -> Dict[str, float]:
    n = len(df)
    if n < 100:
        return {}
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    val = df.iloc[train_end:val_end]
    X = val[_feature_cols(df)].values
    y = val[target_col].values
    if target_col.startswith("target_dir_"):
        mask = y != 0
        X = X[mask]
        y = y[mask]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score((y > 0).astype(int), proba)
    else:
        preds = model.predict(X)
        proba = preds
        auc = 0.0
    return {
        "accuracy": float(accuracy_score((y > 0).astype(int), preds)),
        "auc": float(auc),
        "precision": float(precision_score((y > 0).astype(int), preds, zero_division=0)),
        "recall": float(recall_score((y > 0).astype(int), preds, zero_division=0)),
        "proba_mean": float(np.mean(proba)),
        "proba_p10": float(np.quantile(proba, 0.10)),
        "proba_p50": float(np.quantile(proba, 0.50)),
        "proba_p90": float(np.quantile(proba, 0.90)),
        "proba_p99": float(np.quantile(proba, 0.99)),
    }


def _signal_strength_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if "signal_strength" not in trades.columns or trades.empty:
        return {}
    s = trades["signal_strength"].dropna()
    if s.empty:
        return {}
    return {
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
    }


def _holding_time_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {}
    t1 = pd.to_datetime(trades["entry_time"])
    t2 = pd.to_datetime(trades["exit_time"])
    mins = (t2 - t1).dt.total_seconds() / 60.0
    return {"avg_minutes": float(mins.mean()), "median_minutes": float(mins.median())}


def _pnl_breakdown(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {}
    wins = trades[trades["pnl"] > 0]["pnl"]
    losses = trades[trades["pnl"] < 0]["pnl"]
    return {
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "largest_win": float(wins.max()) if not wins.empty else 0.0,
        "largest_loss": float(losses.min()) if not losses.empty else 0.0,
    }


def _long_short_distribution(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {}
    longs = (trades["direction"] == "LONG").mean()
    shorts = (trades["direction"] == "SHORT").mean()
    return {"long_pct": float(longs), "short_pct": float(shorts)}


def _always_long_baseline(
    df: pd.DataFrame,
    horizon_minutes: int,
    candle_minutes: int,
    leverage: float,
    fee: float,
) -> Dict[str, float]:
    hold_candles = max(1, int(round(horizon_minutes / candle_minutes)))
    returns = []
    for i in range(0, len(df) - hold_candles - 1, hold_candles):
        entry = float(df["open"].iloc[i + 1])
        exit_px = float(df["close"].iloc[i + 1 + hold_candles])
        price_return = (exit_px - entry) / entry
        pnl = price_return * leverage - (fee * 2.0)
        pnl = max(pnl, -1.0)
        returns.append(pnl)
    if not returns:
        return {}
    equity = (1.0 + pd.Series(returns)).cumprod()
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return {
        "total_trades": len(returns),
        "profitability_win_rate": float((np.array(returns) > 0).mean()),
        "average_return": float(np.mean(returns)),
        "profit_factor": float(np.sum(np.array(returns)[np.array(returns) > 0]) / max(1e-9, -np.sum(np.array(returns)[np.array(returns) < 0]))),
        "max_drawdown": float(dd.min()),
    }


def _random_baseline(
    df: pd.DataFrame,
    trade_rate: float,
    horizon_minutes: int,
    candle_minutes: int,
    leverage: float,
    fee: float,
    slippage: float,
    tp_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    feature_cols = _feature_cols(df)
    # Dummy model to satisfy interface
    class Dummy:
        def predict_proba(self, X):
            r = np.random.rand(X.shape[0], 2)
            r[:, 1] = np.random.rand(X.shape[0])
            r[:, 0] = 1 - r[:, 1]
            return r
    model = Dummy()

    # Override thresholds to force trade_rate with random signals
    # We simulate by turning the signal rule into probability of trade.
    trades = []
    equity = 1.0
    hold_candles = max(1, int(round(horizon_minutes / candle_minutes)))
    lookback = 240
    idx = df.index
    pos = max(lookback, 1)
    last = len(idx) - 1
    while pos + 1 < last:
        if np.random.rand() > trade_rate:
            pos += 1
            continue
        direction = "LONG" if np.random.rand() > 0.5 else "SHORT"
        entry_pos = pos + 1
        entry_price = float(df["open"].iloc[entry_pos])
        entry_time = idx[entry_pos]
        take_profit = entry_price * (1.0 + tp_mult * 0.003) if direction == "LONG" else entry_price * (1.0 - tp_mult * 0.003)
        stop_loss = entry_price * (1.0 - sl_mult * 0.003) if direction == "LONG" else entry_price * (1.0 + sl_mult * 0.003)
        liquidation = entry_price * (1.0 - 1.0 / leverage) if direction == "LONG" else entry_price * (1.0 + 1.0 / leverage)
        end_pos = min(entry_pos + hold_candles, last)
        future = df.iloc[entry_pos : end_pos + 1]

        exit_reason = "TIME_EXIT"
        exit_price = float(future["close"].iloc[-1])
        exit_time = future.index[-1]
        ambiguous = False

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
                ambiguous = True
                exit_reason = "AMBIGUOUS"
                exit_price = np.nan
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

        if np.isnan(exit_price):
            pnl = 0.0
        else:
            if direction == "LONG":
                price_return = (exit_price - entry_price) / entry_price
            else:
                price_return = (entry_price - exit_price) / entry_price
            leveraged_return = price_return * leverage
            pnl = leveraged_return - (fee * 2.0)
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
                "pnl": pnl,
            }
        )

        if not ambiguous:
            equity *= (1.0 + pnl)
            if equity <= 0:
                break
        pos = end_pos + 1

    return pd.DataFrame(trades)


def _label_shuffle_test(df: pd.DataFrame, target_col: str) -> object:
    df_shuf = df.copy()
    df_shuf[target_col] = np.random.permutation(df_shuf[target_col].values)
    X = df_shuf[_feature_cols(df_shuf)].values
    y = df_shuf[target_col].values
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit diagnostics for backtest validity")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--target", default="target_cls_30m", help="Target column")
    parser.add_argument("--trades", default="BTCNEW/artifacts/backtest/backtest_trades.csv", help="Trades CSV")
    parser.add_argument("--horizon", default=30, type=int, help="Horizon minutes")
    parser.add_argument("--candle-minutes", default=5, type=int, help="Candle minutes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    trades = pd.read_csv(args.trades)
    model = _load_model(Path(args.model))

    original_metrics = _metrics_from_trades(trades)
    trade_rate = original_metrics["total_trades"] / max(1, len(df))

    random_trades = _random_baseline(
        df=df,
        trade_rate=trade_rate,
        horizon_minutes=int(args.horizon),
        candle_minutes=int(args.candle_minutes),
        leverage=100.0,
        fee=0.0004,
        slippage=0.0,
        tp_mult=1.2,
        sl_mult=0.7,
    )
    random_metrics = _metrics_from_trades(random_trades) if not random_trades.empty else {}

    shuffled_model = _label_shuffle_test(df, args.target)
    shuffled_trades = backtest(
        df=df,
        model=shuffled_model,
        feature_cols=_feature_cols(df),
        horizon_minutes=int(args.horizon),
        candle_minutes=int(args.candle_minutes),
        leverage=100.0,
        fee_per_side=0.0004,
        slippage=0.0,
        expected_return_threshold=0.0015,
        min_confidence=0.60,
        min_signal_strength=0.02,
        prob_threshold=0.55,
        strength_quantile=0.90,
        stop_loss_mult=0.7,
        take_profit_mult=1.2,
    )
    shuffled_metrics = _metrics_from_trades(shuffled_trades) if not shuffled_trades.empty else {}

    pred_quality = _prediction_quality(df, model, args.target)
    signal_strength_stats = _signal_strength_stats(trades)
    holding_stats = _holding_time_stats(trades)
    pnl_stats = _pnl_breakdown(trades)
    long_short = _long_short_distribution(trades)
    always_long = _always_long_baseline(
        df=df,
        horizon_minutes=int(args.horizon),
        candle_minutes=int(args.candle_minutes),
        leverage=100.0,
        fee=0.0004,
    )

    print("=== Original Strategy Metrics ===")
    print(original_metrics)
    print("=== Random Baseline Metrics ===")
    print(random_metrics)
    print("=== Label Shuffle Metrics ===")
    print(shuffled_metrics)
    print("=== Long vs Short Distribution ===")
    print(long_short)
    print("=== Prediction Quality (Validation) ===")
    print(pred_quality)
    print("=== Predicted Probability Distribution ===")
    print({k: pred_quality.get(k) for k in ["proba_mean", "proba_p10", "proba_p50", "proba_p90", "proba_p99"]})
    print("=== Signal Strength Distribution ===")
    print(signal_strength_stats)
    print("=== Holding Time (minutes) ===")
    print(holding_stats)
    print("=== PnL Breakdown ===")
    print(pnl_stats)
    print("=== Always-Long Baseline ===")
    print(always_long)


if __name__ == "__main__":
    main()
