# BTC Structural Mispricing Scanner (Kalshi)

This is isolated from your old BTC model.

## Folder layout

- `btc_mispricing_engine.py`: scanner + model engine
- `.env`: API keys
- `artifacts/models`: trained model files (`.joblib`)
- `artifacts/state`: per-contract model state/inertia state
- `artifacts/logs`: scan logs
- `artifacts/latest`: latest top scanned contract snapshot

## 1) Setup

```bash
cd "/Users/johnn/Desktop/Coding Projects/BTC Model/BTC/STRUCTURAL_BTC_ENGINE"
python3 -m venv ../../.venv
source ../../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with at least:
- `COINALYZE_API_KEY`
- `COINALYZE_SYMBOL`
- `FRED_API_KEY`
- `KALSHI_API_KEY`

## 2) Verify APIs

```bash
python btc_mispricing_engine.py doctor --kalshi-market-ticker "$KALSHI_MARKET_TICKER"
```

## 3) Train models (recommended: multiple horizons)

Train 15m model:
```bash
python btc_mispricing_engine.py train --interval 5m --period 60d --horizon-minutes 15 --require-live-factors --model-name btc_mispricing_model_5m_15m.joblib
```

Train horizon set (10,20,30,40,50,60):
```bash
for h in 10 20 30 40 50 60; do
  python btc_mispricing_engine.py train \
    --interval 5m \
    --period 60d \
    --horizon-minutes "$h" \
    --require-live-factors \
    --calibration-method platt \
    --model-name "btc_mispricing_model_5m_${h}m.joblib"
done
```

## 4) Manual mode (recommended now)

```bash
python btc_mispricing_engine.py manual \
  --model-path artifacts/models/btc_mispricing_model_5m_15m.joblib \
  --target-price 68000 \
  --market-yes-prob 47 \
  --require-live-factors
```

Notes:
- `--market-yes-prob` accepts `0-1` or `0-100`.
- If you omit `--target-price` or `--market-yes-prob`, the command will prompt you.

## 5) Run continuous scanner (auto market selection)

```bash
python btc_mispricing_engine.py run \
  --model-path artifacts/models \
  --model-cycle-seconds 60 \
  --market-refresh-seconds 60 \
  --require-live-factors \
  --max-staleness-minutes 5 \
  --critical-factor-max-age-minutes 90 \
  --min-expiry-minutes 0.2 \
  --max-expiry-minutes 70 \
  --market-prob-min 0.10 \
  --market-prob-max 0.90 \
  --max-dist-sigma 3.0 \
  --strike-eval-cooldown-seconds 20 \
  --alert-dedup-minutes 15
```

The daemon now:
- Runs continuously until stopped (`Ctrl+C`/SIGTERM) with graceful shutdown.
- Evaluates one best BTC hourly strike every cycle (default: once per minute).
- Uses rolling 1-hour realized volatility for expected move pruning.
- Sends Discord alerts using `ALERT_WEBHOOK_URL` (if set).
- Prints stop-loss alerts in plain language with clear percentage reasons.

## 6) What to watch while running

- Alerts in terminal: `EDGE DETECTED`
- Scan log: `artifacts/logs/mispricing_runner_scan_log.csv`
- Top current contract snapshot: `artifacts/latest/mispricing_top_contract.json`

## 7) Trading flow (manual execution)

1. Start scanner and wait for `EDGE DETECTED` alerts.
2. Confirm alert still valid on Kalshi UI (price/liquidity can move).
3. Place trade manually on Kalshi.
4. Log your fills separately and track realized EV vs model EV.

## Reset only this new system

```bash
rm -rf artifacts/*
mkdir -p artifacts/{models,logs,state,latest}
```
