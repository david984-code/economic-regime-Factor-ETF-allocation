# Terminal Verification Commands

Copy and paste these commands in order to verify your system works.

---

## Step 1: Navigate to Project

```powershell
cd C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main
```

**Expected output:** _(none, just changes directory)_

---

## Step 2: Verify Dependencies

```powershell
# Check Python version
python --version
```

**Expected output:** `Python 3.11.9` (or similar)

```powershell
# Check polars is installed
python -c "import polars as pl; print(f'Polars version: {pl.__version__}')"
```

**Expected output:** `Polars version: 1.38.1`

```powershell
# Check pyarrow is installed
python -c "import pyarrow as pa; print(f'PyArrow version: {pa.__version__}')"
```

**Expected output:** `PyArrow version: 23.0.1`

```powershell
# Check all key imports
python -c "import pandas, numpy, scipy, yfinance, fredapi, openpyxl, polars, pyarrow; print('All dependencies OK')"
```

**Expected output:** `All dependencies OK`

---

## Step 3: Test Database Module

```powershell
python -c "from src.database import Database; db = Database(); print(f'Database created at: {db.db_path}'); db.close()"
```

**Expected output:** `Database created at: C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\outputs\allocations.db`

---

## Step 4: Run Individual Modules

### Test Economic Regime Classification

```powershell
python -m src.economic_regime
```

**Expected output (last few lines):**
```
[CURRENT] Latest Classified Month:
Date: 2026-03-31
Regime: Unknown
Risk-on score (0..1): nan

[SUCCESS] Saved regime labels to database
[SUCCESS] Saved CSV backup: ...\outputs\regime_labels_expanded.csv
```

**Duration:** ~7 seconds

---

### Test Portfolio Optimizer

```powershell
python -m src.optimizer
```

**Expected output (last few lines):**
```
[SUCCESS] Optimized: Recovery
[SUCCESS] Optimized: Overheating
[SUCCESS] Optimized: Contraction
[SUCCESS] Optimized: Stagflation
[SUCCESS] Optimized: Unknown
[SUCCESS] Saved optimal allocations to database
[SUCCESS] Saved CSV backup: ...\outputs\optimal_allocations.csv
```

**Duration:** ~3 seconds

---

### Test Backtest (Polars Version)

```powershell
python -m src.backtest_polars
```

**Expected output (last section):**
```
[PERFORMANCE] Portfolio Performance Based on Dynamic Regime Allocations:
CAGR: 13.85%
Volatility: 12.74%
Sharpe: 0.74
Max Drawdown: -22.83%

[BENCHMARK] Equal-Weight Benchmark (no regime timing):
CAGR: 12.30%
Volatility: 14.68%
Sharpe: 0.56
Max Drawdown: -31.69%

[SUCCESS] Saved results to database: ...\outputs\allocations.db
```

**Duration:** ~5 seconds

---

## Step 5: Test Complete Pipeline

```powershell
python run_daily_update.py
```

**Expected output:**
```
2026-03-07 XX:XX:XX - INFO - ================================================================================
2026-03-07 XX:XX:XX - INFO - STARTING DAILY UPDATE - 2026-03-07 XX:XX:XX
2026-03-07 XX:XX:XX - INFO - ================================================================================

2026-03-07 XX:XX:XX - INFO - [STEP] Fetch economic data & classify regimes
2026-03-07 XX:XX:XX - INFO - Starting: src.economic_regime
2026-03-07 XX:XX:XX - INFO - [OK] Completed: src.economic_regime

2026-03-07 XX:XX:XX - INFO - [STEP] Optimize portfolio allocations
2026-03-07 XX:XX:XX - INFO - Starting: src.optimizer
2026-03-07 XX:XX:XX - INFO - [OK] Completed: src.optimizer

2026-03-07 XX:XX:XX - INFO - [STEP] Run backtest & update database
2026-03-07 XX:XX:XX - INFO - Starting: src.backtest_polars
2026-03-07 XX:XX:XX - INFO - [OK] Completed: src.backtest_polars

2026-03-07 XX:XX:XX - INFO - SUMMARY
2026-03-07 XX:XX:XX - INFO - src.economic_regime: SUCCESS
2026-03-07 XX:XX:XX - INFO - src.optimizer: SUCCESS
2026-03-07 XX:XX:XX - INFO - src.backtest_polars: SUCCESS

2026-03-07 XX:XX:XX - INFO - [SUCCESS] All steps completed successfully in 15.7s
2026-03-07 XX:XX:XX - INFO - [SUCCESS] Database updated: outputs/allocations.db
```

**Duration:** ~15-20 seconds  
**Exit code:** 0 (success)

---

## Step 6: Verify Database Contents

```powershell
# Check regime labels
python -c "from src.database import Database; db = Database(); regimes = db.load_regime_labels(); print(f'Regime data rows: {len(regimes)}'); print(f'Latest regime: {regimes.iloc[-1][\"regime\"]}'); db.close()"
```

**Expected output:**
```
Regime data rows: 195
Latest regime: Unknown
```

```powershell
# Check optimal allocations
python -c "from src.database import Database; db = Database(); allocs = db.load_optimal_allocations(); print(f'Regimes optimized: {list(allocs.keys())}'); db.close()"
```

**Expected output:**
```
Regimes optimized: ['Stagflation', 'Contraction', 'Recovery', 'Overheating', 'Unknown']
```

```powershell
# Check latest backtest results
python -c "from src.database import Database; db = Database(); results = db.get_latest_backtest_results(); print(f'Latest run: {results[\"run_date\"]}'); print(f'Portfolio Sharpe: {results[\"portfolio\"][\"Sharpe\"]:.2f}'); print(f'Portfolio CAGR: {results[\"portfolio\"][\"CAGR\"]:.2%}'); db.close()"
```

**Expected output:**
```
Latest run: 2026-03-07 14:56:04
Portfolio Sharpe: 0.74
Portfolio CAGR: 13.85%
```

---

## Step 7: Verify Files Created

```powershell
# Check database file exists
Test-Path outputs\allocations.db
```

**Expected output:** `True`

```powershell
# Check log file was created
dir logs\daily_update_*.log
```

**Expected output:** List of log files with today's date

```powershell
# Check CSV backups exist
dir outputs\*.csv
```

**Expected output:**
```
regime_labels_expanded.csv
optimal_allocations.csv
current_factor_weights.csv
```

---

## Step 8: Run Quality Checks

### Linting

```powershell
python -m ruff check src tests
```

**Expected output:** `All checks passed!`

---

### Type Checking

```powershell
python -m mypy src tests
```

**Expected output:** `Success: no issues found in 9 source files`

---

### Unit Tests

```powershell
python -m pytest tests -v
```

**Expected output (last line):** `16 passed in X.XXs`

---

## Step 9: View Latest Log

```powershell
# View last 30 lines of today's log
Get-Content logs\daily_update_*.log | Select-Object -Last 30
```

**Expected output:** Shows summary with all SUCCESS

---

## Step 10: Test Task Scheduler (If Set Up)

```powershell
# List portfolio tasks
Get-ScheduledTask | Where-Object {$_.TaskName -like "*Portfolio*"} | Format-Table TaskName, State
```

**Expected output:**
```
TaskName                         State
--------                         -----
Portfolio Update - Pre-Market    Ready
Portfolio Update - Post-Market   Ready
```

```powershell
# Get last run info
Get-ScheduledTaskInfo -TaskName "Portfolio Update - Pre-Market" | Select-Object LastRunTime, LastTaskResult
```

**Expected output:**
```
LastRunTime           LastTaskResult
-----------           --------------
3/7/2026 8:30:00 AM   0
```
(0 means success)

```powershell
# Run task manually (will take ~20 seconds)
Start-ScheduledTask -TaskName "Portfolio Update - Pre-Market"

# Wait a moment, then check status
Get-ScheduledTask -TaskName "Portfolio Update - Pre-Market" | Select-Object State
```

**Expected:** State should be "Running" then "Ready"

---

## Troubleshooting Commands

### If something fails, check:

```powershell
# View full error log
Get-Content logs\daily_update_*.log

# Check Python environment
python -c "import sys; print(sys.executable); print(sys.path)"

# Check working directory
Get-Location

# Verify FRED API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key present:', 'Yes' if os.getenv('FRED_API_KEY') else 'No')"
```

---

## Quick Health Check (Run Anytime)

```powershell
# One-liner to check system health
python -c "from src.database import Database; db = Database(); r = db.get_latest_backtest_results(); print(f'System Status: OK'); print(f'Last update: {r[\"run_date\"]}'); print(f'Sharpe: {r[\"portfolio\"][\"Sharpe\"]:.2f}'); db.close()"
```

**Expected output:**
```
System Status: OK
Last update: 2026-03-07 14:56:04
Sharpe: 0.74
```

---

## Success Checklist

Run through this checklist. All should be ✅:

- [ ] `python --version` shows 3.11+
- [ ] `import polars` works
- [ ] `import pyarrow` works  
- [ ] Database file exists at `outputs/allocations.db`
- [ ] `python -m src.economic_regime` completes
- [ ] `python -m src.optimizer` completes
- [ ] `python -m src.backtest_polars` completes
- [ ] `python run_daily_update.py` shows all SUCCESS
- [ ] Log files created in `logs/`
- [ ] `python -m ruff check src tests` passes
- [ ] `python -m mypy src tests` passes
- [ ] `python -m pytest tests -v` shows 16 passed
- [ ] Database queries return data
- [ ] Task Scheduler tasks created (if applicable)

---

## If All Checks Pass ✅

**Congratulations!** Your system is production-ready.

Next steps:
1. Set up Task Scheduler (see TASK_SCHEDULER_SETUP.md)
2. Monitor logs for first week
3. Review results daily: `python -c "from src.database import Database; db = Database(); r = db.get_latest_backtest_results(); print(r); db.close()"`

---

## Emergency Rollback

If something breaks and you need to go back to basics:

```powershell
# Use original backtest (pandas version)
python -m src.backtest

# This still works and doesn't require polars
```
