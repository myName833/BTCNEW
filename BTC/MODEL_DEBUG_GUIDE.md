# How This Model Produces Probability

This model estimates:

`P(BTC price at horizon > target_price)`

## Step-by-step

1. Fetch market data (Coinbase candles) and live factors (Coinalyze/FRED/BLS).
2. Build features from recent market state:
   - returns, volatility, trend, candle spread, volume behavior
   - external factors: order book imbalance, funding, liquidations, OI, macro flags
3. Convert your target price into a threshold return relative to current price.
4. Convert threshold return into `threshold_z` using current volatility.
5. Predict probability with trained classifier.
6. Apply isotonic calibration (if available) to improve probability reliability.

Output fields:
- `probability_price_above_target`
- `probability_price_below_or_equal_target`

## Why it may look near 50/50

- Target price is close to current price.
- Horizon is short and market is balanced.
- Live factor coverage is weak/sparse.

## Debug: prove it is using real data

Each prediction now includes a `debug` section in JSON and appends a row to:

- `model_artifacts/prediction_debug_log.csv`

Check these fields:

1. `debug.required_factor_coverage`
   - Should be > 0 for live factors.
2. `debug.latest_key_features`
   - Shows latest factor values used for this prediction.
3. `debug.raw_model_probability_above_target`
   - Uncalibrated model output.
4. `debug.calibrated_model_probability_above_target`
   - Final probability after calibration.
5. `debug.data_freshness.seconds_since_latest_candle`
   - Lower is fresher market data.

If coverage is near zero, API data is missing and model will rely more on price-only features.

## Accuracy sanity checks

1. Compare model vs baseline using walk-forward metrics (`brier`, `log_loss`).
2. Track `prediction_debug_log.csv` over time and compare with realized outcomes.
3. If model is consistently near 50% and not beating baseline, features are likely weak or noisy for that horizon.
