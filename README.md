# BTC Model Monorepo

This repo contains multiple BTC-related models and tools:

- `BTCNEW/` BTC Target Alert + Futures Signals
- `STRUCTURAL_BTC_ENGINE/` Structural Mispricing Engine

## BTCNEW (Target Alert + Futures)

Primary file: `BTCNEW/btc_target_alert.py`

Supports:
- Kalshi-style target probability: `P(BTC final price > strike at expiry)`
- Futures directional signals: expected return + LONG/SHORT/NO_TRADE over short horizons

### Setup

```bash
cd BTCNEW
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Futures mode (directional signal)

```bash
python btc_target_alert.py --mode futures --timeframe 30 --leverage 10
```

See `BTCNEW/README.md` for full flags, features, and outputs.

## Structural BTC Engine

Primary file: `STRUCTURAL_BTC_ENGINE/btc_mispricing_engine.py`

### Setup

```bash
cd STRUCTURAL_BTC_ENGINE
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Required env values:
- `COINALYZE_API_KEY`
- `COINALYZE_SYMBOL`
- `FRED_API_KEY`

### Quick Start (run)

```bash
cd STRUCTURAL_BTC_ENGINE
source ../.venv/bin/activate
python btc_mispricing_engine.py run \
  --model-path artifacts/models \
  --model-cycle-seconds 60 \
  --market-refresh-seconds 60 \
  --require-live-factors
```

See `STRUCTURAL_BTC_ENGINE/README.md` for full usage and training commands.
