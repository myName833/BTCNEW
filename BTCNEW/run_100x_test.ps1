param(
    [int]$Iterations = 120,
    [int]$SleepSeconds = 60,
    [int]$TimeframeMinutes = 30,
    [double]$Leverage = 100,
    [double]$MaintenanceMarginRate = 0.005,
    [double]$TakerFeeBps = 5.0,
    [double]$BacktestMinConfidence = 0.60,
    [double]$BacktestMinSignalStrength = 0.40
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $scriptDir ".venv\Scripts\python.exe"
$app = Join-Path $scriptDir "btc_target_alert.py"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python. Create it first: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt"
}

if (-not (Test-Path $app)) {
    throw "Could not find btc_target_alert.py at $app"
}

Write-Host "Running 100x futures signal collection..."
Write-Host "Iterations=$Iterations SleepSeconds=$SleepSeconds Timeframe=$TimeframeMinutes Leverage=$Leverage"

for ($i = 1; $i -le $Iterations; $i++) {
    Write-Host "[$i/$Iterations] futures signal run..."
    & $python $app `
        --mode futures `
        --timeframe $TimeframeMinutes `
        --leverage $Leverage `
        --contract-check `
        --maintenance-margin-rate $MaintenanceMarginRate `
        --taker-fee-bps $TakerFeeBps

    if ($i -lt $Iterations -and $SleepSeconds -gt 0) {
        Start-Sleep -Seconds $SleepSeconds
    }
}

Write-Host ""
Write-Host "Running futures backtest..."
& $python $app `
    --mode futures_backtest `
    --backtest-min-confidence $BacktestMinConfidence `
    --backtest-min-signal-strength $BacktestMinSignalStrength

$report = Join-Path $scriptDir "artifacts\latest\futures_backtest_report.json"
if (Test-Path $report) {
    Write-Host "Backtest report saved to: $report"
} else {
    Write-Host "Backtest report not found yet. Check terminal output for details."
}
