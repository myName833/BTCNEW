from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.train_model import train_time_series_models
from quant_pipeline.utils import read_csv_time_index, write_json
from quant_pipeline.backtest_engine import backtest


def walkforward_splits(df: pd.DataFrame, years: List[int]) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    splits = []
    for y in years:
        train = df[df.index.year == y]
        test = df[df.index.year == y + 1]
        if not train.empty and not test.empty:
            splits.append((train, test, f"year_{y}_to_{y+1}"))
    return splits


def rolling_splits(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    n = len(df)
    splits = []
    if n < 200:
        return splits
    # Split 60/20
    t1 = int(n * 0.60)
    t2 = int(n * 0.80)
    splits.append((df.iloc[:t1], df.iloc[t1:t2], "roll_60_20"))
    # Split 80/20
    t3 = int(n * 0.80)
    splits.append((df.iloc[:t3], df.iloc[t3:], "roll_80_20"))
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--task", choices=["clf", "reg"], required=True)
    parser.add_argument("--years", default="2022,2023,2024", help="Train years list")
    parser.add_argument("--out", default="BTCNEW/artifacts/walkforward.json", help="Output JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    feature_cols = [c for c in df.columns if c not in df.columns[df.columns.str.startswith("future_return_")].tolist()
                    and not c.startswith("target_cls_") and not c.startswith("target_dir_") and c != "time"]

    # Diagnostics
    diag: Dict[str, Any] = {}
    diag["row_count"] = int(len(df))
    diag["datetime_index"] = "index"
    diag["min_timestamp"] = str(df.index.min()) if not df.empty else None
    diag["max_timestamp"] = str(df.index.max()) if not df.empty else None
    if args.target in df.columns:
        diag["target_counts"] = df[args.target].value_counts(dropna=False).to_dict()
    else:
        diag["target_counts"] = {}

    # Drop NaNs in features/target
    work = df.dropna(subset=feature_cols + [args.target])
    diag["rows_after_filter"] = int(len(work))

    results = []
    splits = walkforward_splits(work, years)
    if not splits:
        splits = rolling_splits(work)

    if not splits:
        write_json(Path(args.out), {"results": [], "diagnostics": diag, "warning": "Dataset too short for walk-forward splits."})
        print("Saved:", args.out)
        print("WARNING: Dataset too short for walk-forward splits.")
        return

    for train, test, tag in splits:
        models, metrics = train_time_series_models(train, feature_cols, args.target, args.task)
        # Use RF if present, else any
        model = models.get("rf") or next(iter(models.values()))
        # Validation metrics on test split
        X_test = test[feature_cols].values
        y_test = test[args.target].values
        if args.task == "clf" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)
        else:
            preds = model.predict(X_test)
        acc = float(accuracy_score((y_test > 0).astype(int), preds))
        prec = float(precision_score((y_test > 0).astype(int), preds, zero_division=0))
        rec = float(recall_score((y_test > 0).astype(int), preds, zero_division=0))

        # Backtest on test split for profit factor / drawdown
        bt = backtest(
            df=test,
            model=model,
            feature_cols=feature_cols,
            horizon_minutes=30,
            candle_minutes=5,
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
        if bt.empty:
            profit_factor = 0.0
            max_dd = 0.0
        else:
            non_amb = bt[bt["exit_reason"] != "AMBIGUOUS"]
            profit = float(non_amb.loc[non_amb["pnl"] > 0, "pnl"].sum())
            loss = float(-non_amb.loc[non_amb["pnl"] < 0, "pnl"].sum())
            profit_factor = profit / loss if loss > 0 else 0.0
            equity = (1.0 + non_amb["pnl"]).cumprod()
            if equity.empty:
                max_dd = 0.0
            else:
                roll_max = equity.cummax()
                dd = (equity - roll_max) / roll_max
                max_dd = float(dd.min())

        results.append(
            {
                "fold": tag,
                "train_start": str(train.index.min()),
                "train_end": str(train.index.max()),
                "test_start": str(test.index.min()),
                "test_end": str(test.index.max()),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "profit_factor": profit_factor,
                "max_drawdown": max_dd,
            }
        )

    write_json(Path(args.out), {"results": results, "diagnostics": diag})
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
