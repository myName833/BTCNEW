# BTCNEW Target Alert

`btc_target_alert.py` supports:

- Kalshi-style target probability: `P(BTC final price > strike at expiry)`
- Futures directional signals: expected return + LONG/SHORT/NO_TRADE over short horizons

It evaluates edge vs market probabilities (Kalshi modes), applies optional probability calibration, logs results, and can run in manual, auto, or futures modes.

## Setup

```bash
cd BTCNEW
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Key `.env` Variables

- `COINBASE_PRODUCT_ID` (default: `BTC-USD`)
- `COINALYZE_API_KEY` (recommended)
- `FRED_API_KEY` (optional)
- `ALERT_WEBHOOK_URL` (Discord webhook)
- `KALSHI_API_KEY` / `KALSHI_API_TOKEN` / `KALSHI_API_BASE` (for live market scan)
- `CALIBRATOR_ARTIFACT_PATH` (optional override; default artifact path is used if omitted)
- `FUTURES_EXCHANGE` (optional: `binance`, `bybit`, `okx` to augment futures features)
- `FUTURES_OVERRIDE_FEATURES` (default: `true` to override funding/OI/liquidation/orderbook with futures data)
- `BINANCE_FUTURES_SYMBOL` (default: `BTCUSDT`)
- `BYBIT_FUTURES_SYMBOL` (default: `BTCUSDT`)
- `OKX_FUTURES_SYMBOL` (default: `BTC-USDT-SWAP`)

## Modes

### 1) Manual mode (single run)

```bash
python btc_target_alert.py --mode manual --target 71250 --timeframe 60
```

Interactive:

```bash
python btc_target_alert.py --mode manual --interactive
```

### 2) Auto mode (live Kalshi scan)

Scans live BTC Kalshi markets every minute, filters low-quality markets, runs BTCNEW model, and alerts Discord when edge passes threshold.

```bash
python btc_target_alert.py \
  --mode auto \
  --timeframe 60 \
  --poll-seconds 60 \
  --edge-threshold 0.04
```

Useful auto flags:
- `--market-prob-min` / `--market-prob-max` (default `0.03` / `0.97`)
- `--max-spread` (default `0.12`)
- `--min-expiry-minutes` / `--max-expiry-minutes` (default `2` / `180`)
- `--max-cycles` (for short test runs)

### 3) Futures mode (directional signal)

Generates LONG/SHORT/NO_TRADE signals using expected return, confidence, and signal strength.

```bash
python btc_target_alert.py \
  --mode futures \
  --timeframe 30 \
  --leverage 10
```

Useful futures flags:
- `--return-threshold` (expected return threshold, default `0.0015`)
- `--min-confidence` (default `0.60`)
- `--min-signal-strength` (default `0.40`)
- `--take-profit-mult` (default `1.20`)
- `--stop-loss-mult` (default `0.70`)

## Model Features (Current)

- Vol-normalized distance:
  - `distance_vol_normalized_signed`
  - `distance_vol_normalized_abs`
- Momentum:
  - `return_1m`, `return_3m`, `return_5m`, `return_10m`, `momentum_acceleration`
- Realized volatility:
  - `vol_1m`, `vol_5m`, `vol_15m`, `realized_vol_annual`, `realized_vol_expansion`
- Time decay:
  - `minutes_remaining`, `sqrt_time_remaining`, `time_adjusted_distance`
- Position/trend:
  - `distance_to_high_15m`, `distance_to_low_15m`, `distance_to_vwap`
  - `price_minus_ema_3`, `price_minus_ema_10`, `ema_3_minus_ema_10`
- Near-strike specialization:
  - dedicated near-strike weight set when distance is very small
- Distribution layer:
  - `expected_final_price`, `expected_variance`, and distribution-based probability blend

## Decision Rules

Validation action (`BET_YES` or `NO_BET`) uses distance buckets:

- `<0.15%`: require `prob >= 0.68` and `edge >= 0.07`
- `<0.30%`: require `prob >= 0.65` and `edge >= 0.05`
- `<0.60%`: require `prob >= 0.60` and `edge >= 0.04`
- `>=0.60%`: require `prob >= 0.57` and `edge >= 0.03`

Auto Discord alerts are typically gated by `--edge-threshold 0.04`.

## Confidence

Confidence now uses a reliability score (not only raw probability), based on:
- probability separation from 50/50
- vol-normalized distance strength
- agreement across model components
- short-term volatility noise
- calibration availability
- warning penalties

Outputs include:
- `confidence_score`
- `confidence_tier` (`LOW` / `MEDIUM` / `HIGH` / `VERY_HIGH`)

## Calibration

Train calibrator from resolved logs:

```bash
MPLCONFIGDIR=artifacts/tmp_mpl MPLBACKEND=Agg python calibration_audit.py \
  --log artifacts/logs/query_log.csv \
  --artifact artifacts/models/btcnew_calibrator.joblib \
  --method auto \
  --min-samples 100
```

If you want to save best calibrator even when guardrails fail:

```bash
MPLCONFIGDIR=artifacts/tmp_mpl MPLBACKEND=Agg python calibration_audit.py \
  --log artifacts/logs/query_log.csv \
  --artifact artifacts/models/btcnew_calibrator.joblib \
  --method auto \
  --min-samples 100 \
  --save-best-anyway
```

Guardrail tuning flags are available:
- `--guardrail-ece-max`
- `--guardrail-brier-max`
- `--guardrail-avg-gap-max`

## Outputs

- Terminal summary (raw + calibrated probabilities)
- Discord alert (raw/calibrated/market shown)
- Latest JSON: `artifacts/latest/latest_alert.json`
- Query log CSV: `artifacts/logs/query_log.csv`
- Futures log CSV: `artifacts/logs/futures_query_log.csv`
- Optional chart PNGs: `artifacts/charts/`
- Resolved outcome updates are written back into query log for calibration

## Python API

```python
from btc_target_alert import run_btc_target_alert

result = run_btc_target_alert(
    target_price=71250.0,
    timeframe_minutes=60,
    bankroll=1000.0,
    edge_threshold=0.04,
    plot=False,
)
```
# BTCNEW
