from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import read_csv_time_index, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate classifier probabilities")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--model", required=True, help="Model path (.joblib)")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--method", choices=["isotonic", "sigmoid"], default="isotonic")
    parser.add_argument("--out", default="BTCNEW/artifacts/models/calibrated.joblib", help="Output model path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    feature_cols = [c for c in df.columns if not c.startswith("future_return_") and not c.startswith("target_")]
    X = df[feature_cols]
    y = df[args.target].values

    model_path = Path(args.model)
    if not model_path.exists():
        # Try fallback order
        candidates = [
            model_path.with_name(f"{args.target}_lightgbm.joblib"),
            model_path.with_name(f"{args.target}_xgboost.joblib"),
            model_path.with_name(f"{args.target}_rf.joblib"),
            model_path.with_name(f"{args.target}_logreg.joblib"),
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break
    if not model_path.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = joblib.load(model_path)
    calibrator = CalibratedClassifierCV(model, method=args.method, cv=5)
    calibrator.fit(X, y)
    joblib.dump(calibrator, Path(args.out))
    write_json(Path(args.out).with_suffix(".json"), {"method": args.method, "features": feature_cols})
    print("Saved calibrated model:", args.out)


if __name__ == "__main__":
    main()
