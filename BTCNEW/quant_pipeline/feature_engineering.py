from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import rolling_zscore, safe_pct_change, clip_extremes


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = safe_pct_change(out["close"], 1)
    out["ret_5"] = safe_pct_change(out["close"], 5)
    out["ret_15"] = safe_pct_change(out["close"], 15)
    out["ret_30"] = safe_pct_change(out["close"], 30)
    out["ret_60"] = safe_pct_change(out["close"], 60)
    out["mom_5_15"] = out["ret_5"] - out["ret_15"]
    out["mom_15_60"] = out["ret_15"] - out["ret_60"]
    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ret_1 = safe_pct_change(out["close"], 1)
    out["realized_vol_30"] = ret_1.rolling(30).std(ddof=0)
    out["realized_vol_60"] = ret_1.rolling(60).std(ddof=0)
    out["realized_vol_120"] = ret_1.rolling(120).std(ddof=0)
    out["vol_cluster"] = out["realized_vol_30"] / (out["realized_vol_120"] + 1e-8)
    out["vol_expansion"] = out["realized_vol_30"] / (out["realized_vol_60"] + 1e-8)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    return out


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vol_ma_20"] = out["volume"].rolling(20).mean()
    out["vol_surge"] = out["volume"] / (out["vol_ma_20"] + 1e-8)
    out["vol_change"] = safe_pct_change(out["volume"], 1)
    return out


def add_derivatives_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "funding_rate" in out.columns:
        out["funding_regime"] = out["funding_rate"].rolling(48).mean()
        out["funding_divergence"] = out["funding_rate"] - out["funding_regime"]
    if "open_interest_change" in out.columns:
        out["oi_accel"] = out["open_interest_change"].diff()
        out["oi_trend"] = out["open_interest_change"].rolling(24).mean()
    if "liquidation_pressure" in out.columns:
        out["liq_pressure_z"] = rolling_zscore(out["liquidation_pressure"], 100)
        out["liq_pressure_abs"] = out["liquidation_pressure"].abs()
    if "liquidation_imbalance" in out.columns:
        out["liq_imbalance_z"] = rolling_zscore(out["liquidation_imbalance"], 100)
    if "basis_spread" in out.columns:
        out["basis_z"] = rolling_zscore(out["basis_spread"], 100)
    if "perp_premium" in out.columns:
        out["perp_premium_z"] = rolling_zscore(out["perp_premium"], 100)
    if "long_short_ratio" in out.columns:
        out["long_short_ratio_z"] = rolling_zscore(out["long_short_ratio"], 100)
    return out


def add_microstructure_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Placeholders for future integration.
    out["orderbook_imbalance"] = 0.0
    out["trade_flow_imbalance"] = 0.0
    out["volume_delta"] = 0.0
    out["liquidity_gap"] = 0.0
    return out


def add_vol_adjusted_momentum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = out["realized_vol_30"].replace(0, np.nan)
    out["mom_vol_adj_15"] = out["ret_15"] / (vol + 1e-8)
    out["mom_vol_adj_60"] = out["ret_60"] / (vol + 1e-8)
    return out


def add_rolling_sharpe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ret = safe_pct_change(out["close"], 1)
    mean = ret.rolling(60).mean()
    std = ret.rolling(60).std(ddof=0)
    out["rolling_sharpe_60"] = mean / (std + 1e-8)
    return out


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ema20_raw"] = close.ewm(span=20, adjust=False).mean()
    out["ema100_raw"] = close.ewm(span=100, adjust=False).mean()
    out["ema20_slope_raw"] = out["ema20_raw"].diff(5) / (close + 1e-8)

    high = out["high"]
    low = out["low"]
    prev_close = close.shift(1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14).sum() / (atr + 1e-8)
    minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14).sum() / (atr + 1e-8)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)) * 100
    out["adx14_raw"] = dx.rolling(14).mean()
    # Copies used for model features (will be normalized later)
    out["ema20"] = out["ema20_raw"]
    out["ema100"] = out["ema100_raw"]
    out["ema20_slope"] = out["ema20_slope_raw"]
    out["adx14"] = out["adx14_raw"]
    return out


def normalize_features(df: pd.DataFrame, feature_cols: list[str], window: int = 200) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        out[c] = rolling_zscore(out[c], window=window)
        out[c] = clip_extremes(out[c], max_abs=10.0)
    return out


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    out = add_momentum_features(df)
    out = add_volatility_features(out)
    out = add_volume_features(out)
    out = add_derivatives_features(out)
    out = add_microstructure_placeholders(out)
    out = add_vol_adjusted_momentum(out)
    out = add_rolling_sharpe(out)
    out = add_regime_features(out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"open", "high", "low", "close", "volume"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if c.startswith("future_return_") or c.startswith("target_"):
            continue
        if c.endswith("_raw"):
            continue
        cols.append(c)
    return cols
