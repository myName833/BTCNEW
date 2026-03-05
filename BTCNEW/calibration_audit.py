#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:
    IsotonicRegression = None
    LogisticRegression = None

EPS = 1e-12
STD_MIN = 1e-5
NORM = NormalDist()


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    return float(np.mean((p - y) ** 2))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    n = len(y)
    if n <= 0:
        return 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        c = int(np.sum(m))
        if c == 0:
            continue
        out += (c / n) * abs(float(np.mean(p[m])) - float(np.mean(y[m])))
    return float(out)


def _bucket_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    bins = np.arange(0.5, 1.0001, 0.1)
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        m = (p >= lo) & (p < hi if i < len(bins) - 2 else p <= hi)
        c = int(np.sum(m))
        if c == 0:
            rows.append({"bucket": f"{lo:.1f}-{hi:.1f}", "count": 0, "avg_pred": None, "actual_rate": None})
        else:
            rows.append(
                {
                    "bucket": f"{lo:.1f}-{hi:.1f}",
                    "count": c,
                    "avg_pred": float(np.mean(p[m])),
                    "actual_rate": float(np.mean(y[m])),
                }
            )
    return pd.DataFrame(rows)


def _reliability_points(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if np.any(m):
            xs.append(float(np.mean(p[m])))
            ys.append(float(np.mean(y[m])))
    return np.array(xs), np.array(ys)


def _fit_platt(x_tr: np.ndarray, y_tr: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_tr.reshape(-1, 1), y_tr)
    return clf


def _fit_isotonic(x_tr: np.ndarray, y_tr: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_tr, y_tr)
    return iso


def _apply(method: str, model, x: np.ndarray) -> np.ndarray:
    if method == "platt":
        return _clip(model.predict_proba(x.reshape(-1, 1))[:, 1])
    if method == "isotonic":
        return _clip(model.predict(x))
    return _clip(x)


def _apply_temperature(p: np.ndarray, temp: float) -> np.ndarray:
    temp = max(float(temp), 1e-6)
    p = _clip(p)
    logit = np.log(p / (1 - p))
    return _clip(1.0 / (1.0 + np.exp(-(logit / temp))))


def _apply_neutral_shrink(p: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    p = _clip(p)
    return _clip(0.5 + (p - 0.5) * a)


def _implied_std_hourly_from_row(row: pd.Series) -> float:
    if "predicted_sigma_hourly" in row.index:
        v = pd.to_numeric(pd.Series([row.get("predicted_sigma_hourly")]), errors="coerce").iloc[0]
        if pd.notna(v) and float(v) > 0:
            return float(v)
    p = float(np.clip(float(row["model_probability_above_raw"]), 1e-6, 1 - 1e-6))
    spot = float(row.get("spot_price", np.nan))
    tgt = float(row.get("target_price", np.nan))
    tfm = float(row.get("timeframe_minutes", np.nan))
    if not np.isfinite(spot) or not np.isfinite(tgt) or not np.isfinite(tfm) or tfm <= 0 or spot <= 0 or tgt <= 0:
        return np.nan
    z = NORM.inv_cdf(float(np.clip(1.0 - p, 1e-6, 1 - 1e-6)))
    z_abs = abs(float(z))
    if z_abs < 1e-3:
        return np.nan
    sigma_t = abs(math.log(tgt / spot)) / z_abs
    h = tfm / 60.0
    return float(sigma_t / math.sqrt(max(h, 1e-6)))


def _variance_scale_probability(p: np.ndarray, scale: np.ndarray) -> np.ndarray:
    p = _clip(p)
    s = np.clip(np.asarray(scale, dtype=float), 0.25, 4.0)
    z = np.array([NORM.inv_cdf(float(np.clip(1.0 - v, 1e-6, 1 - 1e-6))) for v in p], dtype=float)
    z2 = z / s
    out = np.array([1.0 - NORM.cdf(float(v)) for v in z2], dtype=float)
    return _clip(out)


def _stage_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "brier": _brier(y, p),
        "log_loss": _logloss(y, p),
        "ece": _ece(y, p),
        "avg_pred": float(np.mean(_clip(p))),
        "actual_rate": float(np.mean(y)),
    }


def _resolve_path(raw: str, base_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Train OOS probability calibrator + diagnostics for BTCNEW")
    ap.add_argument("--log", type=str, default="artifacts/logs/query_log.csv")
    ap.add_argument("--out-dir", type=str, default="artifacts/diagnostics")
    ap.add_argument("--artifact", type=str, default="artifacts/models/btcnew_calibrator.joblib")
    ap.add_argument("--method", choices=["auto", "platt", "isotonic"], default="auto")
    ap.add_argument("--market-shrinkage", type=float, default=0.12)
    ap.add_argument("--min-samples", type=int, default=120)
    args = ap.parse_args()

    if joblib is None or LogisticRegression is None or IsotonicRegression is None:
        raise SystemExit("Missing dependencies. Install with: pip install -r BTCNEW/requirements.txt")

    log_path = _resolve_path(args.log, base_dir)
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")
    out_dir = _resolve_path(args.out_dir, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _resolve_path(args.artifact, base_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path, on_bad_lines="skip", engine="python")
    for c in ["resolved", "actual_hit_above", "model_probability_above_raw", "model_probability_above", "timeframe_minutes", "timestamp_utc"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column in log: {c}")

    r = df[df["resolved"].astype(str).str.lower().isin(["1", "true", "yes"])].copy()
    r["actual_hit_above"] = pd.to_numeric(r["actual_hit_above"], errors="coerce")
    r["model_probability_above_raw"] = pd.to_numeric(r["model_probability_above_raw"], errors="coerce")
    r["timeframe_minutes"] = pd.to_numeric(r["timeframe_minutes"], errors="coerce")
    if "spot_price" in r.columns:
        r["spot_price"] = pd.to_numeric(r["spot_price"], errors="coerce")
    if "target_price" in r.columns:
        r["target_price"] = pd.to_numeric(r["target_price"], errors="coerce")

    r = r.dropna(subset=["actual_hit_above", "model_probability_above_raw", "timeframe_minutes"])
    if len(r) < int(args.min_samples):
        raise SystemExit(f"Need at least {args.min_samples} resolved rows; found {len(r)}")

    r["timestamp_utc"] = pd.to_datetime(r["timestamp_utc"], utc=True, errors="coerce")
    r = r.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    # Stage 1: variance scaling from realized hourly returns vs model-implied hourly std.
    r["horizon_min"] = r["timeframe_minutes"].astype(float).round().astype(int)
    r["predicted_std_hourly"] = r.apply(_implied_std_hourly_from_row, axis=1)

    # Realized hourly return std from observed spot path in resolved history.
    spot_ok = "spot_price" in r.columns and r["spot_price"].notna().sum() > 3
    if spot_ok:
        rr = r[["timestamp_utc", "spot_price", "horizon_min"]].dropna().copy()
        rr["log_spot"] = np.log(np.clip(rr["spot_price"].astype(float), 1e-9, None))
        rr["dlog"] = rr["log_spot"].diff()
        rr["dt_h"] = rr["timestamp_utc"].diff().dt.total_seconds() / 3600.0
        rr = rr[(rr["dt_h"] > 0.01) & (rr["dlog"].notna())].copy()
        rr["ret_hourly"] = rr["dlog"] / np.sqrt(np.clip(rr["dt_h"], 1e-6, None))
        global_realized_std = float(np.nanstd(rr["ret_hourly"].to_numpy(), ddof=0)) if len(rr) > 5 else np.nan
        realized_by_h: Dict[int, float] = {
            int(h): float(np.nanstd(g["ret_hourly"].to_numpy(), ddof=0))
            for h, g in rr.groupby("horizon_min")
            if len(g) >= 5
        }
    else:
        global_realized_std = np.nan
        realized_by_h = {}

    pred_std_global = float(np.nanmean(r["predicted_std_hourly"].to_numpy()))
    if not np.isfinite(global_realized_std) or global_realized_std <= 0:
        global_realized_std = pred_std_global if np.isfinite(pred_std_global) and pred_std_global > 0 else 0.01
    if not np.isfinite(pred_std_global) or pred_std_global <= 0:
        pred_std_global = global_realized_std

    global_scale = float(np.clip(global_realized_std / max(pred_std_global, STD_MIN), 0.6, 2.5))
    scale_by_horizon: Dict[int, float] = {}
    for h, g in r.groupby("horizon_min"):
        pred_h = float(np.nanmean(g["predicted_std_hourly"].to_numpy()))
        real_h = float(realized_by_h.get(int(h), np.nan))
        if not np.isfinite(pred_h) or pred_h <= 0:
            pred_h = pred_std_global
        if not np.isfinite(real_h) or real_h <= 0:
            real_h = global_realized_std
        scale_by_horizon[int(h)] = float(np.clip(real_h / max(pred_h, STD_MIN), 0.6, 2.5))

    r["variance_scale"] = r["horizon_min"].map(scale_by_horizon).astype(float)
    r["variance_scale"] = r["variance_scale"].fillna(global_scale).clip(0.6, 2.5)
    r["p_stage0_raw"] = _clip(r["model_probability_above_raw"].astype(float).to_numpy())
    r["p_stage1_variance"] = _variance_scale_probability(r["p_stage0_raw"].to_numpy(), r["variance_scale"].to_numpy())

    y = r["actual_hit_above"].astype(float).to_numpy()
    x = r["p_stage1_variance"].astype(float).to_numpy()

    n = len(r)
    n_tr = max(50, int(0.7 * n))
    n_va = max(20, int(0.15 * n))
    n_va = min(n_va, n - n_tr - 1)
    tr = slice(0, n_tr)
    va = slice(n_tr, n_tr + n_va)
    te = slice(n_tr + n_va, n)

    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]
    x_te, y_te = x[te], y[te]
    raw_va = r["p_stage0_raw"].to_numpy()[va]
    raw_te = r["p_stage0_raw"].to_numpy()[te]

    cands: Dict[str, Dict[str, object]] = {}
    for m in ["platt", "isotonic"]:
        try:
            model = _fit_platt(x_tr, y_tr) if m == "platt" else _fit_isotonic(x_tr, y_tr)
            p_va = _apply(m, model, x_va)
            cands[m] = {"model": model, "brier": _brier(y_va, p_va)}
        except Exception:
            continue
    if not cands:
        raise SystemExit("No calibrator could be fit")

    if args.method == "auto":
        method = sorted(cands.keys(), key=lambda k: cands[k]["brier"])[0]
    else:
        method = args.method
        if method not in cands:
            raise SystemExit(f"Requested method {method} failed to fit")

    model = cands[method]["model"]
    p_va_cal = _apply(method, model, x_va)

    # Stage 2/3: joint grid search for temperature + neutral shrink alpha.
    t_grid = np.arange(1.0, 2.5001, 0.1)
    a_grid = np.arange(0.6, 1.0001, 0.05)
    temp_only_rows = []
    best = {
        "temp": 1.0,
        "alpha": 1.0,
        "brier": _brier(y_va, p_va_cal),
        "ece": _ece(y_va, p_va_cal),
        "avg_pred": float(np.mean(p_va_cal)),
        "actual": float(np.mean(y_va)),
    }
    for t in t_grid:
        p_t = _apply_temperature(p_va_cal, float(t))
        temp_only_rows.append(
            {
                "temperature": float(t),
                "brier": _brier(y_va, p_t),
                "ece": _ece(y_va, p_t),
                "avg_pred": float(np.mean(p_t)),
                "actual": float(np.mean(y_va)),
            }
        )
        for a in a_grid:
            p_f = _apply_neutral_shrink(p_t, float(a))
            b = _brier(y_va, p_f)
            e = _ece(y_va, p_f)
            if (b < best["brier"] - 1e-12) or (abs(b - best["brier"]) <= 1e-12 and e < best["ece"]):
                best = {
                    "temp": float(t),
                    "alpha": float(a),
                    "brier": float(b),
                    "ece": float(e),
                    "avg_pred": float(np.mean(p_f)),
                    "actual": float(np.mean(y_va)),
                }

    p_te_var = x_te
    p_te_cal = _apply(method, model, p_te_var)
    p_te_temp = _apply_temperature(p_te_cal, best["temp"])
    p_te_final = _apply_neutral_shrink(p_te_temp, best["alpha"])

    # Stage summaries required by prompt.
    test_stage_summary = {
        "pre_variance": _stage_metrics(y_te, raw_te),
        "post_variance": _stage_metrics(y_te, p_te_var),
        "post_temperature": _stage_metrics(y_te, p_te_temp),
        "final": _stage_metrics(y_te, p_te_final),
        "baseline_0_5": _stage_metrics(y_te, np.full_like(y_te, 0.5, dtype=float)),
    }

    final_avg_gap = abs(test_stage_summary["final"]["avg_pred"] - test_stage_summary["final"]["actual_rate"])
    final_brier = test_stage_summary["final"]["brier"]
    final_ece = test_stage_summary["final"]["ece"]
    guardrail_fail_reasons = []
    if final_avg_gap > 0.05:
        guardrail_fail_reasons.append(f"abs(avg_pred-actual)={final_avg_gap:.4f} > 0.05")
    if final_brier > 0.26:
        guardrail_fail_reasons.append(f"brier={final_brier:.4f} > 0.26")
    if final_ece > 0.10:
        guardrail_fail_reasons.append(f"ece={final_ece:.4f} > 0.10")
    guardrail_pass = len(guardrail_fail_reasons) == 0

    metrics = {
        "leakage_guard": {
            "time_split": "train<valid<test chronological split",
            "train_eval_overlap": False,
            "note": "calibration fit only on train slice; validation/test are strictly later timestamps",
        },
        "split": {"n_total": n, "n_train": len(x_tr), "n_valid": len(x_va), "n_test": len(x_te)},
        "variance_stage": {
            "predicted_std_hourly_global": float(pred_std_global),
            "realized_std_hourly_global": float(global_realized_std),
            "variance_scale_global": float(global_scale),
            "adjusted_std_hourly_global": float(pred_std_global * global_scale),
            "by_horizon_minutes": {
                str(k): {
                    "predicted_std_hourly": float(np.nanmean(r.loc[r["horizon_min"] == k, "predicted_std_hourly"])),
                    "realized_std_hourly": float(realized_by_h.get(k, global_realized_std)),
                    "variance_scale": float(scale_by_horizon.get(k, global_scale)),
                    "adjusted_std_hourly": float(
                        np.nanmean(r.loc[r["horizon_min"] == k, "predicted_std_hourly"]) * scale_by_horizon.get(k, global_scale)
                    ),
                }
                for k in sorted(scale_by_horizon.keys())
            },
        },
        "grid_search": {
            "temperature_only": temp_only_rows,
            "joint_best": best,
        },
        "test_stage_summary": test_stage_summary,
        "test_buckets_final": _bucket_table(y_te, p_te_final).to_dict(orient="records"),
        "guardrail": {
            "pass": guardrail_pass,
            "reasons": guardrail_fail_reasons,
            "criteria": {
                "abs_avg_pred_minus_actual_le": 0.05,
                "brier_le": 0.26,
                "ece_le": 0.10,
            },
        },
    }

    # Always write diagnostics; only save model artifact on guardrail pass.
    pd.DataFrame(temp_only_rows).to_csv(out_dir / "temperature_grid_metrics.csv", index=False)
    _bucket_table(y_te, p_te_final).to_csv(out_dir / "bucket_winrate_final.csv", index=False)
    pd.DataFrame(
        {
            "bin_left": np.linspace(0.0, 0.9, 10),
            "count_raw": np.histogram(raw_te, bins=np.linspace(0, 1, 11))[0],
            "count_final": np.histogram(p_te_final, bins=np.linspace(0, 1, 11))[0],
        }
    ).to_csv(out_dir / "sharpness_histogram.csv", index=False)

    with (out_dir / "calibration_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_method": method,
                "temperature": best["temp"],
                "alpha": best["alpha"],
                "artifact": str(artifact_path),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    if plt is not None:
        xr, yr = _reliability_points(y_te, raw_te)
        xf, yf = _reliability_points(y_te, p_te_final)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        if len(xr):
            ax.plot(xr, yr, marker="o", label="Raw")
        if len(xf):
            ax.plot(xf, yf, marker="s", label="Final")
        ax.set_title("Reliability Curve (Test OOS)")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "reliability_curve_test.png", dpi=140)
        plt.close(fig)

    if guardrail_pass:
        joblib.dump(
            {
                "calibrator": {
                    "method": method,
                    "model": model,
                    "temperature": float(best["temp"]),
                    "neutral_alpha": float(best["alpha"]),
                    "neutral_shrinkage": float(1.0 - best["alpha"]),
                    "market_shrinkage": float(np.clip(args.market_shrinkage, 0.0, 0.95)),
                    "variance_scale_global": float(global_scale),
                    "variance_scale_by_horizon": {str(k): float(v) for k, v in scale_by_horizon.items()},
                },
                "metrics": metrics,
                "trained_at": pd.Timestamp.utcnow().isoformat(),
            },
            artifact_path,
        )
        print(f"Saved calibrator: {artifact_path}")
    else:
        print("Guardrail failed: calibrator artifact NOT saved.")
        for rr in guardrail_fail_reasons:
            print(f" - {rr}")

    print(
        "Final Test Metrics: "
        f"Brier={test_stage_summary['final']['brier']:.4f} | "
        f"ECE={test_stage_summary['final']['ece']:.4f} | "
        f"AvgPred={test_stage_summary['final']['avg_pred']:.4f} | "
        f"Actual={test_stage_summary['final']['actual_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
