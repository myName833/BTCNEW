# BTC Structural Mispricing Engine

Primary engine file: `btc_mispricing_engine.py`

## Setup

```bash
cd BTC
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill your keys in .env
```

Required env values:
- `COINALYZE_API_KEY`
- `COINALYZE_SYMBOL`
- `FRED_API_KEY`
- `KALSHI_API_KEY` (required for `run`, optional for `train/predict`)

## Run Project (Quick Start)

```bash
cd "/Users/johnn/Desktop/Coding Projects/BTC Model/BTC/STRUCTURAL_BTC_ENGINE"
source ../../.venv/bin/activate
python btc_mispricing_engine.py run \
  --model-path artifacts/models \
  --model-cycle-seconds 60 \
  --market-refresh-seconds 60 \
  --require-live-factors
```

## What This Engine Does

- Structural probability model: `P(BTC > target in Y minutes)`
- Live Coinbase + Coinalyze + macro flags
- Walk-forward evaluation with Brier/log loss
- Simulated Kalshi EV metrics during training
- Platt calibration by default (smooth, no isotonic plateau by design)
- Target-distance dominance (`normalized_distance`)
- Missing-data flags (no silent zero-fill-as-real-signal)
- Regime-aware probability inertia
- Continuous runner with Kalshi edge detection + alerts

## Verify Keys + Endpoints

```bash
../.venv/bin/python btc_mispricing_engine.py doctor --kalshi-market-ticker "$KALSHI_MARKET_TICKER"
```

## Clean Retrain (Delete Old Artifacts)

```bash
rm -f model_artifacts/btc_mispricing_* model_artifacts/mispricing_*
```

## Train

```bash
../.venv/bin/python btc_mispricing_engine.py train \
  --interval 5m \
  --period 60d \
  --horizon-minutes 15 \
  --require-live-factors \
  --critical-factor-max-age-minutes 90 \
  --calibration-method platt \
  --model-name btc_mispricing_model_5m_15m.joblib
```

## Predict Once

```bash
../.venv/bin/python btc_mispricing_engine.py predict \
  --model-path model_artifacts/btc_mispricing_model_5m_15m.joblib \
  --target-price 67000 \
  --max-staleness-minutes 5 \
  --critical-factor-max-age-minutes 90 \
  --require-live-factors
```

## Continuous Runner (Model + Kalshi)

```bash
../.venv/bin/python btc_mispricing_engine.py run \
  --model-path model_artifacts \
  --model-cycle-seconds 300 \
  --kalshi-cycle-seconds 60 \
  --require-live-factors \
  --max-staleness-minutes 5 \
  --critical-factor-max-age-minutes 90 \
  --min-expiry-minutes 5 \
  --max-expiry-minutes 180 \
  --market-prob-min 0.10 \
  --market-prob-max 0.90 \
  --max-abs-normalized-distance 1.5 \
  --alert-webhook "$ALERT_WEBHOOK_URL"
```

## Important Runtime Rules

- Prediction fails if candles are stale beyond `--max-staleness-minutes`.
- Prediction fails if critical live factors are stale/missing and `--require-live-factors` is set.
- Runner scans active BTC Kalshi markets automatically (no hardcoded ticker).
- Runner ignores extreme implied-probability markets outside `(market_prob_min, market_prob_max)`.
- Runner only alerts when edge/liquidity/spread/persistence/stability rules pass.
- Runner writes cycle logs to `model_artifacts/mispricing_runner_log.csv`.
