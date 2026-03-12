from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import write_json


def monte_carlo_equity(returns: np.ndarray, n_sims: int = 1000) -> dict:
    if len(returns) == 0:
        return {}
    max_drawdowns = []
    ruin = 0
    for _ in range(n_sims):
        sim = np.random.permutation(returns)
        equity = (1.0 + sim).cumprod()
        roll_max = np.maximum.accumulate(equity)
        drawdown = (equity - roll_max) / roll_max
        max_drawdowns.append(drawdown.min())
        if equity.min() <= 0:
            ruin += 1
    return {
        "max_drawdown_mean": float(np.mean(max_drawdowns)),
        "max_drawdown_p5": float(np.percentile(max_drawdowns, 5)),
        "max_drawdown_p95": float(np.percentile(max_drawdowns, 95)),
        "ruin_prob": float(ruin / n_sims),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo analysis of trade returns")
    parser.add_argument("--trades", required=True, help="Backtest trades CSV")
    parser.add_argument("--sims", default=1000, type=int, help="Number of simulations")
    parser.add_argument("--out", default="BTCNEW/artifacts/monte_carlo.json", help="Output JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.trades)
    returns = df[df["exit_reason"] != "AMBIGUOUS"]["pnl"].values
    stats = monte_carlo_equity(returns, n_sims=int(args.sims))
    write_json(Path(args.out), stats)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
