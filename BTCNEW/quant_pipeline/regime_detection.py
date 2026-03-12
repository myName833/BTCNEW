from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import read_csv_time_index


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = out["realized_vol_30"]
    vol_q = vol.quantile([0.33, 0.66]).values
    out["vol_regime"] = "medium"
    out.loc[vol <= vol_q[0], "vol_regime"] = "low"
    out.loc[vol >= vol_q[1], "vol_regime"] = "high"

    trend = out["ret_60"].rolling(60).mean()
    out["trend_regime"] = "mean_reversion"
    out.loc[trend > 0, "trend_regime"] = "trend"
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regime detection")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--out", default="BTCNEW/artifacts/data/dataset_with_regimes.csv", help="Output CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    df = classify_regimes(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=True)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
