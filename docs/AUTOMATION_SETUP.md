# Portfolio Automation Setup Guide

## Overview

This guide will help you set up **automated daily portfolio updates** that run:
- **8:30 AM ET** - Before market open (get fresh overnight data)
- **4:30 PM ET** - After market close (analyze today's prices)

The system updates economic regimes, optimizes allocations, and runs backtests automatically, storing everything in the SQLite database.

---

## Prerequisites

✅ Python 3.11+ installed  
✅ All dependencies installed (`uv pip install -r requirements.txt` or `pip install -e .`)  
✅ `.env` file with your FRED_API_KEY  
✅ Windows 10/11 (for Task Scheduler)

---

## Quick Setup (5 minutes)

### Step 1: Test Manual Run

First, verify everything works manually:

```bash
# Navigate to project directory
cd C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main

# Test the runner script
python run_daily_update.py
```

You should see:
```
[STEP] Fetch economic data & classify regimes
✓ Completed: src.economic_regime
[STEP] Optimize portfolio allocations
✓ Completed: src.optimizer
[STEP] Run backtest & update database
✓ Completed: src.backtest_polars
✓ All steps completed successfully
```

### Step 2: Create Task Scheduler Jobs

#### Open Task Scheduler
1. Press `Win + R`
2. Type `taskschd.msc`
3. Press Enter

#### Create Morning Update (Before Market Open)

1. Click **"Create Task"** (not "Create Basic Task")

2. **General Tab:**
   - Name: `Portfolio Update - Pre-Market`
   - Description: `Update portfolio data before market opens`
   - ✅ Run whether user is logged on or not
   - ✅ Run with highest privileges
   - Configure for: Windows 10

3. **Triggers Tab:**
   - Click **New...**
   - Begin the task: `On a schedule`
   - Settings: `Daily`
   - Start: `8:30:00 AM`
   - ✅ Enabled
   - Advanced settings:
     - ✅ Stop task if it runs longer than: `30 minutes`
   - Click **OK**

4. **Actions Tab:**
   - Click **New...**
   - Action: `Start a program`
   - Program/script: `python`
   - Add arguments: `run_daily_update.py`
   - Start in: `C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main`
   - Click **OK**

5. **Conditions Tab:**
   - ❌ Start the task only if the computer is on AC power (uncheck if laptop)
   - ✅ Wake the computer to run this task (if you want it to wake from sleep)

6. **Settings Tab:**
   - ✅ Allow task to be run on demand
   - ✅ Run task as soon as possible after a scheduled start is missed
   - If task fails, restart every: `5 minutes`, Attempt to restart up to: `3 times`
   - Stop the task if it runs longer than: `1 hour`

7. Click **OK** and enter your Windows password when prompted

#### Create Evening Update (After Market Close)

Repeat the above steps with these changes:
- Name: `Portfolio Update - Post-Market`
- Trigger time: `4:30:00 PM`

---

## Alternative: PowerShell Setup

If you prefer PowerShell for better error handling:

### Create Task via PowerShell

```powershell
# Morning task (8:30 AM)
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-ExecutionPolicy Bypass -File `"C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\run_daily.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At 8:30AM
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "Portfolio Update - Pre-Market" -Action $action -Trigger $trigger -Principal $principal -Settings $settings

# Evening task (4:30 PM)
$trigger2 = New-ScheduledTaskTrigger -Daily -At 4:30PM
Register-ScheduledTask -TaskName "Portfolio Update - Post-Market" -Action $action -Trigger $trigger2 -Principal $principal -Settings $settings
```

---

## Testing Your Setup

### Test Task Manually

1. Open Task Scheduler
2. Find your task: `Portfolio Update - Pre-Market`
3. Right-click → **Run**
4. Watch the status change to "Running"
5. Check the logs: `logs/daily_update_YYYYMMDD.log`

### Verify Results

After a successful run, check:
```bash
# Check database was updated
python -c "from src.database import Database; db = Database(); results = db.get_latest_backtest_results(); print(f'Latest run: {results[\"run_date\"]}'); db.close()"

# View today's log
type logs\daily_update_20260306.log
```

---

## Production Checklist

Before going live with automated runs:

- [ ] **Test manual run completes successfully**
- [ ] **Verify .env file has valid FRED_API_KEY**
- [ ] **Check database is being updated** (`outputs/allocations.db`)
- [ ] **Test Task Scheduler run manually** (right-click → Run)
- [ ] **Verify logs are being created** (`logs/` directory)
- [ ] **Set up disk space monitoring** (logs accumulate over time)
- [ ] **Consider log rotation** (delete logs older than 30 days)
- [ ] **Test what happens if run fails** (check retry settings)
- [ ] **Document your broker's API** (if connecting live trades)

---

## Monitoring & Maintenance

### View Logs

```bash
# View today's log
type logs\daily_update_20260306.log

# View last 50 lines
Get-Content logs\daily_update_20260306.log -Tail 50

# Search for errors
Select-String -Path logs\daily_update_*.log -Pattern "ERROR" -CaseSensitive
```

### Check Task History

1. Open Task Scheduler
2. Click on your task
3. Click the **History** tab (bottom)
4. Look for `Action completed` events

### Database Health Check

```python
from src.database import Database
import pandas as pd

with Database() as db:
    # Check latest regime data
    regimes = db.load_regime_labels()
    print(f"Latest regime date: {regimes.index[-1]}")
    
    # Check latest backtest
    results = db.get_latest_backtest_results()
    print(f"Latest backtest: {results['run_date']}")
    print(f"Portfolio Sharpe: {results['portfolio']['Sharpe']:.2f}")
```

### Clean Old Logs (Monthly)

```powershell
# Delete logs older than 30 days
$cutoff = (Get-Date).AddDays(-30)
Get-ChildItem -Path "logs" -Filter "daily_update_*.log" | Where-Object { $_.LastWriteTime -lt $cutoff } | Remove-Item
```

---

## Troubleshooting

### Task doesn't run

**Check:**
1. Task Scheduler is running: `Get-Service -Name "Task Scheduler"`
2. Task is enabled in Task Scheduler
3. Computer is on or has "Wake to run" enabled
4. User account has permissions

### Script fails

**Check logs:**
```bash
type logs\daily_update_20260306.log
```

**Common issues:**
- Missing .env file → Solution: Copy `.env.example` to `.env` and add API key
- Module not found → Solution: `pip install -e .`
- Database locked → Solution: Close Excel/other programs accessing files
- Network timeout → Solution: Retry settings in Task Scheduler handle this

### No data updates

**Verify:**
```python
# Check when data was last updated
from src.database import Database
db = Database()
results = db.get_latest_backtest_results()
print(f"Last update: {results['run_date']}")
db.close()
```

---

## Advanced: Connecting to Live Trading

### Broker Integration (Example: Interactive Brokers)

Once you're confident in the automated updates, you can connect to your broker's API:

```python
# Example structure (not included, for reference)
from src.database import Database

def execute_trades():
    """Place orders based on current portfolio weights"""
    with Database() as db:
        # Get latest allocations
        allocations = db.load_optimal_allocations()
        
        # Get current regime
        regimes = db.load_regime_labels()
        current_regime = regimes.iloc[-1]['regime']
        
        # Get target weights for current regime
        target_weights = allocations[current_regime]
        
        # TODO: Connect to broker API
        # TODO: Get current positions
        # TODO: Calculate rebalance trades
        # TODO: Execute orders with limits/stops
```

**⚠️ Important:**
- Start with paper trading
- Implement position size limits
- Add safety checks (max daily trades, position limits)
- Monitor fills and slippage
- Keep audit logs of all trades

---

## Next Steps

1. ✅ Verify manual run works: `python run_daily_update.py`
2. ✅ Create Task Scheduler jobs (8:30 AM & 4:30 PM)
3. ✅ Test tasks run successfully
4. ✅ Monitor for 1 week to ensure stability
5. ✅ Set up log monitoring/alerts
6. 🚀 Consider broker integration for live trading

---

## Support

**Files:**
- `run_daily_update.py` - Main orchestration script
- `run_daily.bat` - Windows batch script
- `run_daily.ps1` - PowerShell script (enhanced)
- `logs/` - Daily execution logs

**Database:**
- Location: `outputs/allocations.db`
- Schema: See `src/database.py`

**Questions?**
- Check logs first: `logs/daily_update_YYYYMMDD.log`
- Review database: `python -c "from src.database import Database; ..."`
- Test components individually: `python -m src.economic_regime`, etc.
