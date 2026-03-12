from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant_pipeline.utils import read_csv_time_index, write_json

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None


def _train_classifiers(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["logreg"] = LogisticRegression(max_iter=2000)
    models["rf"] = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    if lgb is not None:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    if xgb is not None:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
    return models


def _train_regressors(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["linreg"] = LinearRegression()
    models["rf"] = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if lgb is not None:
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    if xgb is not None:
        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    return models


def train_time_series_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: str,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    # For directional target, remove neutral samples (0)
    work = df.copy()
    if target_col.startswith("target_dir_"):
        work = work[work[target_col] != 0]

    X = work[feature_cols]
    y = work[target_col].values

    metrics: Dict[str, float] = {}
    models = _train_classifiers(X, y) if task == "clf" else _train_regressors(X, y)

    # Time-based split: 70/15/15
    n = len(work)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]

    for name, model in models.items():
        model.fit(X_train, y_train)
        if task == "clf":
            proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
            metrics[f"{name}_val_auc"] = float(roc_auc_score((y_val > 0).astype(int), proba))
            proba_t = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
            metrics[f"{name}_test_auc"] = float(roc_auc_score((y_test > 0).astype(int), proba_t))
        else:
            preds = model.predict(X_val)
            metrics[f"{name}_val_mse"] = float(mean_squared_error(y_val, preds))
            preds_t = model.predict(X_test)
            metrics[f"{name}_test_mse"] = float(mean_squared_error(y_test, preds_t))

    # Refit on train+val for final model
    X_full = X.iloc[:val_end]
    y_full = y[:val_end]
    for model in models.values():
        model.fit(X_full, y_full)

    return models, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ML models (classification/regression)")
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--task", choices=["clf", "reg"], required=True)
    parser.add_argument("--out-dir", default="BTCNEW/artifacts/models", help="Model output dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_time_index(Path(args.dataset))
    feature_cols = [
        c
        for c in df.columns
        if not c.startswith("future_return_")
        and not c.startswith("target_")
    ]

    models, metrics = train_time_series_models(df, feature_cols, args.target, args.task)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, model in models.items():
        path = out_dir / f"{args.target}_{name}.joblib"
        joblib.dump(model, path)
        # Save feature list alongside each model for consistent inference
        write_json(out_dir / f"{args.target}_{name}_features.json", {"features": feature_cols})
        saved.append(str(path))

    # Also save a shared feature list for the target
    write_json(out_dir / f"{args.target}_features.json", {"features": feature_cols})

    write_json(out_dir / f"{args.target}_metrics.json", metrics)
    print("Saved models:")
    for p in saved:
        print(" -", p)


if __name__ == "__main__":
    main()
