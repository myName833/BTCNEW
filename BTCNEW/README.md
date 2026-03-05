# BTC Target Alert CLI

`btc_target_alert.py` is a numeric-only Python CLI tool that computes BTC target-hit probabilities for both directions (`above` and `below`), evaluates edge vs market, logs runs, and sends Discord alerts.

## Requirements

- Python 3.11+
- Install dependencies:

```bash
cd BTCNEW
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Copy and fill env values:

```bash
cp .env.example .env
```

Main env keys:
- `COINBASE_PRODUCT_ID` (default `BTC-USD`)
- `COINBASE_API_KEY` (optional)
- `COINALYZE_API_KEY` (optional but recommended)
- `FRED_API_KEY` (optional)
- `ALERT_WEBHOOK_URL` (optional)
- `CALIBRATOR_ARTIFACT_PATH` (optional override for a trained calibrator artifact)

## CLI (numeric-only)

```bash
python btc_target_alert.py \
  --target 67500 \
  --timeframe 53 \
  --bankroll 1000 \
  --edge-threshold 0.03 \
  --plot
```

Arguments:
- `--target` float (USD)
- `--timeframe` int (minutes)
- `--bankroll` float (optional)
- `--stake` float (optional)
- `--edge-threshold` float (default `0.03`)
- `--plot` optional chart output

Interactive mode:

```bash
python btc_target_alert.py --interactive
```

## Function call

```python
from btc_target_alert import run_btc_target_alert

result = run_btc_target_alert(
    target_price=67500.0,
    timeframe_minutes=53,
    bankroll=1000.0,
    edge_threshold=0.03,
    plot=False,
)
```

## Outputs

- Terminal summary with both above/below probabilities and edges.
- Includes raw model probability and calibrated model probability.
- Discord embed alert to `ALERT_WEBHOOK_URL`.
- JSON artifact: `artifacts/latest/latest_alert.json`
- CSV query log: `artifacts/logs/query_log.csv`
- Optional chart PNG: `artifacts/charts/`
- Auto-resolution of past rows + historical accuracy and Brier score.

## Notes

- Missing APIs degrade gracefully: warnings are logged and neutral factors are used.
- If spot/ticker endpoint fails, script attempts candle-close fallback.
- Calibrator loading uses structural artifact format (`{\"method\": ..., \"model\": ...}`), with fallback to no calibration if no artifact is found.
