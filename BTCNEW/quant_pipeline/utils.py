from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_time_index(path: Path, time_col: str = "time") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[time_col])
    if time_col in df.columns:
        df = df.set_index(time_col)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / (std + 1e-8)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def clip_extremes(series: pd.Series, max_abs: float = 10.0) -> pd.Series:
    return series.clip(lower=-max_abs, upper=max_abs)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw)
    except Exception:
        return default
