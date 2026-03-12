from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import write_json


def compute_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {}
    non_amb = trades[trades["exit_reason"] != "AMBIGUOUS"]
    total = len(non_amb)
    tp = int((non_amb["exit_reason"] == "TP_HIT").sum())
    sl = int((non_amb["exit_reason"] == "SL_HIT").sum())
    liq = int((non_amb["exit_reason"] == "LIQUIDATION").sum())
    time_exit = int((non_amb["exit_reason"] == "TIME_EXIT").sum())
    strict_win = tp / total if total else 0.0
    prof_win = float((non_amb["pnl"] > 0).mean()) if total else 0.0

    avg_return = float(non_amb["pnl"].mean()) if total else 0.0
    gross_profit = float(non_amb.loc[non_amb["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-non_amb.loc[non_amb["pnl"] < 0, "pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

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
        "strict_win_rate": strict_win,
        "profitability_win_rate": prof_win,
        "average_return": avg_return,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute backtest metrics")
    parser.add_argument("--trades", required=True, help="Backtest trades CSV")
    parser.add_argument("--out", default="BTCNEW/artifacts/backtest/metrics.json", help="Output JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.trades)
    metrics = compute_metrics(df)
    write_json(Path(args.out), metrics)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
