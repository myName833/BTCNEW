from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import read_csv_time_index, write_json


def select_features_mutual_info(df: pd.DataFrame, feature_cols: List[str], target_col: str, top_k: int = 30) -> List[str]:
    X = df[feature_cols].values
    y = df[target_col].values
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    ranked = sorted(zip(feature_cols, mi), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k]]


def select_features_tree_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    top_k: int = 30,
) -> List[str]:
    X = df[feature_cols].values
    y = df[target_col].values
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k]]


def select_features_permutation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    top_k: int = 30,
) -> List[str]:
    X = df[feature_cols].values
    y = df[target_col].values
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    ranked = sorted(zip(feature_cols, result.importances_mean), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature selection utilities")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--top-k", default=30, type=int, help="Top K features")
    parser.add_argument("--out", default="BTCNEW/artifacts/feature_selection.json", help="Output JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    feature_cols = [c for c in df.columns if c.startswith(("ret_", "realized_", "atr_", "vol_", "mom_", "rolling_", "funding", "oi_", "liq_", "basis", "orderbook", "trade_flow", "volume_"))]

    mi = select_features_mutual_info(df, feature_cols, args.target, args.top_k)
    tree = select_features_tree_importance(df, feature_cols, args.target, args.top_k)
    perm = select_features_permutation(df, feature_cols, args.target, args.top_k)

    write_json(Path(args.out), {"mutual_info": mi, "tree_importance": tree, "permutation": perm})
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
