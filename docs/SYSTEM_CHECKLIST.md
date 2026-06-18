# System Readiness Checklist

Run this checklist after initial setup or anytime you want to verify the system is working.

---

## Quick verification (right now)

All tests passed. Run this to confirm at any time:

```powershell
cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
uv run python scripts/test_system_end_to_end.py
```

---

## Manual testing protocol

### Test 1 — IBC starts Gateway automatically

**What to test:** IBC can launch Gateway without your intervention.

**Steps:**
1. Close IB Gateway if it's running (right-click taskbar → Exit)
2. Double-click: `C:\IBC\StartGateway.bat`
3. Wait 10–30 seconds

**Expected:**
- Command window appears with green "IBC starting..." banner
- IB Gateway opens automatically
- Login dialog auto-fills username/password and logs in
- Gateway shows paper account `DUM429899`
- API indicator shows green (port 4002 active)

**If it fails:**
- Check `C:\IBC\Logs\` for error logs
- Verify credentials in `C:\IBC\config.ini` are correct
- Confirm `TradingMode=paper` in both `config.ini` and `StartGateway.bat`
- Check Help → About in Gateway: build version must match `TWS_MAJOR_VRSN=1044` in `StartGateway.bat`

---

### Test 2 — Health check connects to Gateway

**What to test:** Your Python scripts can fetch positions from IBKR.

**Steps:**
1. Ensure Gateway is running and logged in (from Test 1)
2. Open terminal in Cursor:
   ```powershell
   cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
   uv run python scripts/daily_health_check.py
   ```

**Expected:**
```
==============================================================
  PORTFOLIO HEALTH CHECK  2026-03-30 HH:MM:SS UTC
==============================================================
Gateway : CONNECTED  (127.0.0.1:4002)
Account : DUM429899
NAV     : $  1,024,328.01
...
Symbol  Shares    Mkt Value    Curr%     Tgt%    Drift%          Status
--------------------------------------------------------------------
GLD        402  $   167,894   16.39%   15.83%    +0.56%              OK
...
VERDICT : PASS  —  All positions within tau no-trade band.
```

Exit code `0` = PASS.

**If it fails:**
- `ERROR: Cannot connect to IB Gateway` → Gateway not running or API disabled
- `ConnectionRefused` → Check Gateway settings: Enable API checkbox, port 4002
- `Login failed` → Credentials in `config.ini` are wrong

---

### Test 3 — Dry-run executes without errors

**What to test:** The paper trading script can preview orders.

**Steps:**
```powershell
uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
```

**Expected:**
```
Dry-run PASS (execution safety checks passed).
Report dir: C:\...\reports\paper_trading\2026-03-30
Proposed orders: 0
Turnover (one-way): 0.0000
```

Since your positions are currently at target, it should show **0 orders**. This is correct.

**If it fails:**
- Check the error message — most likely a missing file or validation failure
- Regenerate prices: `uv run python scripts/current_allocation.py`

---

### Test 4 — Trends report generates clean output

**What to test:** Historical analysis works.

**Steps:**
```powershell
uv run python scripts/show_portfolio_trends.py
```

**Expected:**
```
======================================================================
  PORTFOLIO TRENDS & DIAGNOSTICS
======================================================================

----------------------------------------------------------------------
  LIVE SUBMISSIONS  (1 total)
----------------------------------------------------------------------
  Run:    20260326_201322  (date: 2026-03-26)
  Orders: 10   Filled: 0   PreSubmitted/Queued: 10

----------------------------------------------------------------------
  WHAT IS WORKING
----------------------------------------------------------------------
  [OK]  IB Gateway connectivity
  [OK]  Target weight validation
  [OK]  All 10 positions filled from March 26 live run
  ...
```

**If it fails:**
- No reports found → run `daily_health_check.py` first to generate data

---

### Test 5 — Performance analysis runs (this may take 2–3 minutes)

**What to test:** Extended metrics and bear market diagnostics.

**Steps:**
```powershell
uv run python scripts/performance_analysis.py
```

**Expected:**
```
======================================================================
  STRATEGY PERFORMANCE ANALYSIS
======================================================================
  Focus: risk-adjusted returns + bear market protection

----------------------------------------------------------------------
  1. CORE METRICS  (full history)
----------------------------------------------------------------------
  Name                            CAGR      Vol   Sharpe  Sortino   Calmar    MaxDD   Ulcer
  ------------------------------------------------------------------------------
  Strategy (vs SPY)              +8.63%   9.40%   +0.459   +0.623   +0.394  -21.92%   +4.231
  S&P 500                       +13.41%  17.16%   +0.563   +0.782   ...

----------------------------------------------------------------------
  2. BEAR MARKET PROTECTION vs SPY  (core objective)
----------------------------------------------------------------------
  WHAT THE STRATEGY CAPTURES OF MARKET MOVES:
    Downside capture ratio  : 65.2%   (target < 70%)
    -> GOOD: absorbs less than 70% of SPY's losses
    
    Upside capture ratio    : 71.8%   (target > 60%)
    -> Capture ratio (up/down): 1.10x  (asymmetric in your favor)
    
  ALPHA ON BENCHMARK-DOWN DAYS:
    Bear market alpha (ann.) : +2.3%   (positive = outperforms when market falls)
    Bear market hit rate     : 58.7%   (% of down-market days strategy > SPY)
    ...
```

This will take 2–3 minutes on first run (fetches 16+ years of benchmark data).

---

### Test 6 — IBC autostart on Windows login

**What to test:** Gateway starts automatically when you log into Windows.

**Steps:**
1. Close IB Gateway completely
2. Restart Windows (or log out and log back in)
3. Wait 30 seconds after seeing your desktop

**Expected:**
- IB Gateway window opens automatically
- Auto-logs into paper account `DUM429899`
- API light goes green

**If it fails:**
- Check Windows Startup folder: should have `IBC_Gateway_Autostart.bat`
- Verify path: `C:\Users\dns81\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`
- Check Task Manager → Startup tab: IBC_Gateway_Autostart should be listed

---

### Test 7 — Scheduled health checks run automatically

**What to test:** Task Scheduler runs your scripts at the right times.

**Steps:**
1. Wait until 9:30 AM on a weekday (Mon–Fri)
2. After 9:31 AM, check:
   ```powershell
   Get-Content "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\logs\health\*.log" -Tail 50
   ```

**Expected:**
```
2026-03-31 09:30:24 [INFO] DAILY HEALTH CHECK  2026-03-31 13:30 UTC
2026-03-31 09:30:24 [INFO] Connected to IB Gateway successfully
2026-03-31 09:30:25 [INFO] VERDICT: PASS — All positions within tau
```

**Alternative test (run the task manually right now):**
```powershell
schtasks /Run /TN "IBKR Health Check - Morning"
```
Then check the log file immediately.

---

## What to do once all tests pass

### Immediate (this week)
1. **Test the full workflow manually once:**
   - Close Gateway, restart Windows
   - Confirm Gateway auto-starts via IBC
   - Run `daily_health_check.py` manually to verify connection
   - Run `show_portfolio_trends.py` to see your historical data

2. **Wait for Monday 9:30 AM:**
   - Morning health check runs automatically
   - Check log: `logs/health/2026-03-31_health.log`
   - Should see PASS verdict (no drift → no rebalance needed)

### This month (April 2026)
3. **Run performance analysis:**
   ```powershell
   uv run python scripts/performance_analysis.py
   ```
   This tells you what is working vs not working. Focus on these metrics:
   - **Downside capture:** should be < 70% (if higher, you need better bear market protection)
   - **Bear market alpha:** should be > 0% annualised (positive = strategy protects in down markets)
   - **Sortino vs Sharpe:** Sortino should be higher (asymmetric returns in your favor)

4. **Read the auto-diagnosis section:**
   The script will tell you explicitly:
   - "[OK] Downside capture 65%: strong bear market protection" → keep as-is
   - "[X] Bear market alpha -1.2%: strategy underperforms SPY in down markets" → run an experiment to fix it

5. **If metrics are bad, run ONE experiment:**
   The script gives you a queue. Example:
   ```powershell
   # If downside capture too high:
   uv run python scripts/run_capped_rf_experiment.py
   
   # Then analyze results:
   uv run python scripts/analyze_walk_forward.py
   ```
   Only accept the experiment if Sharpe improves by ≥ 0.02 in full walk-forward.

### Ongoing (every month)
6. **First Monday of each month:**
   - Check if rebalance needed: `uv run python scripts/daily_health_check.py`
   - If exit code = 1 (WARN), run live submission during market hours (9:30 AM – 4:00 PM ET)
   - Confirm fills the next morning with a dry-run

7. **Last Friday of each month:**
   - Run `performance_analysis.py`
   - Update your metrics tracking spreadsheet
   - If any metric is red, queue one experiment for next month

---

## The only thing left to test manually

**IBC autostart on login:**
1. Restart Windows right now
2. Wait 30 seconds after desktop appears
3. IB Gateway should open and log in automatically

If that works, you're **100% automated** from here forward.

---

## Quick reference card (print this)

| Command | When | Purpose |
|---------|------|---------|
| `uv run python scripts/test_system_end_to_end.py` | After setup or if broken | Validates all 13 components |
| `uv run python scripts/daily_health_check.py` | Manual anytime | Check positions vs targets |
| `uv run python scripts/show_portfolio_trends.py` | Manual anytime | NAV history + diagnostics |
| `uv run python scripts/performance_analysis.py` | Monthly | Risk-adjusted metrics + bear alpha |
| `uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv` | Before rebalance | Preview orders |
| `uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv` | Monthly rebalance (RTH only) | Live submission |
| `uv run python scripts/current_allocation.py` | Before rebalance | Refresh prices CSV |
| `double-click C:\IBC\StartGateway.bat` | If Gateway not running | Manual start |

**Automated tasks (no action needed):**
- 9:30 AM Mon–Fri: morning health check
- 3:55 PM Mon–Fri: afternoon health check
- 4:15 PM Fridays: weekly trends report
- On Windows login: IBC starts Gateway automatically

**Emergency:**
- Gateway hung? → Kill `ibgateway.exe` in Task Manager, restart via IBC
- Health check failing? → Check `logs/health/*.log` for error details
- Duplicate-run error? → Delete `reports/paper_trading/YYYY-MM-DD/last_paper_submission.json` (only if first run failed)
