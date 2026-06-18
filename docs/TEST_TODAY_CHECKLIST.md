# Test Today Checklist: Verify Automated Trading System

Complete this checklist RIGHT NOW to confirm your system is fully automated and working correctly.

**Estimated time:** 20 minutes

---

## Pre-flight check

Current status (as of your last validation):
- ✓ IBC installed and configured with your credentials
- ✓ All Task Scheduler jobs registered
- ✓ Git security layers active
- ✓ Target weights valid
- ✗ Gateway not currently running → start it first (see Test 1 below)

---

## Test 1 — Start Gateway via IBC and verify auto-login

**Purpose:** Confirm IBC can launch Gateway and log you in automatically.

**Steps:**
1. Open Command Prompt or PowerShell
2. Run:
   ```cmd
   C:\IBC\StartGateway.bat
   ```

**Expected behavior:**
- Command window appears with green IBC banner
- IB Gateway window opens after 10–20 seconds
- Login dialog auto-fills username and logs in
- Gateway main window shows: **Paper Trading** mode, account **DUM429899**
- Bottom-right status: **API** indicator green (or "TWS" green)

**If it fails:**
- Check `C:\IBC\Logs\ibc-YYYYMMDD-HHMMSS.txt` for error messages
- Verify credentials in `C:\IBC\config.ini` are correct
- Confirm paper mode login (not live)

**Duration:** 30 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 2 — Verify API connectivity from Python

**Purpose:** Confirm your scripts can connect to Gateway API.

**Steps:**
1. With Gateway running (from Test 1), open terminal in Cursor:
   ```powershell
   cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
   uv run python -c "from src.execution.ibkr_adapter import IBKRPaperAdapter; adapter = IBKRPaperAdapter(); adapter.connect(); print(f'Connected to {adapter.account}'); positions = adapter.get_positions(); print(f'Positions: {len(positions)}'); adapter.disconnect()"
   ```

**Expected output:**
```
Connected to DUM429899
Positions: 10
```

**If it fails:**
- Gateway API disabled → Gateway settings: Configure → Settings → API → Enable ActiveX and Socket Clients
- Port wrong → verify port 4002 in Gateway settings
- Firewall blocking → allow `ibgateway.exe` in Windows Firewall

**Duration:** 10 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 3 — Run full health check manually

**Purpose:** Verify the health check script reports positions vs targets correctly.

**Steps:**
```powershell
cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
uv run python scripts/daily_health_check.py
```

**Expected output:**
```
==============================================================
  PORTFOLIO HEALTH CHECK  2026-03-30 HH:MM:SS UTC
==============================================================
Gateway : CONNECTED  (127.0.0.1:4002)
Account : DUM429899
NAV     : $  1,024,328.01
Tau     : 1.5%  (no-trade band)

----------------------------------------------------------------------
Symbol  Shares    Mkt Value    Curr%     Tgt%    Drift%          Status
----------------------------------------------------------------------
GLD        402  $   167,894   16.39%   15.83%    +0.56%              OK
IEF        846  $   169,200   16.52%   15.83%    +0.69%              OK
SPY        152  $   161,520   15.77%   15.83%    -0.06%              OK
...
----------------------------------------------------------------------
Max abs drift: 0.69%  (threshold: 1.5%)

VERDICT : PASS  —  All positions within tau no-trade band.
==============================================================
```

**Exit code:**
- `0` = PASS (no drift, no action needed)
- `1` = WARN (drift > tau, rebalance candidate)
- `2` = ERROR (Gateway down or connection failed)

**Check the report file was created:**
```powershell
Get-ChildItem "reports\health\2026-03-30" -Recurse
```
Should see `health_HHMM.json`.

**Duration:** 10 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 4 — Check historical trends report

**Purpose:** Verify the trends script can read past reports and show diagnostics.

**Steps:**
```powershell
uv run python scripts/show_portfolio_trends.py
```

**Expected output:**
```
======================================================================
  PORTFOLIO TRENDS & DIAGNOSTICS
======================================================================

----------------------------------------------------------------------
  LIVE SUBMISSIONS  (1 total)
----------------------------------------------------------------------
  Run:    20260326_201322  (date: 2026-03-26)
  Orders: 10   Filled: 0   PreSubmitted/Queued: 10
  Turnover: 100.0%  (full portfolio initial fill)

----------------------------------------------------------------------
  NAV HISTORY  (5 health snapshots)
----------------------------------------------------------------------
  First: 2026-03-26  $1,023,000.00
  Last:  2026-03-30  $1,024,328.01
  Change: +$1,328.01  (+0.13%)

----------------------------------------------------------------------
  WHAT IS WORKING
----------------------------------------------------------------------
  [OK]  IB Gateway connectivity (last check: today)
  [OK]  Target weight validation (sum ≈ 1.0, all safety checks pass)
  [OK]  All 10 positions filled from March 26 live run
  ...

----------------------------------------------------------------------
  ISSUES & DIAGNOSES  (0 errors, 0 warnings, 1 info)
----------------------------------------------------------------------
  [INFO] After-hours submission
    10 orders submitted 2026-03-26 20:13 (after market close)
    -> Expected: orders queue as PreSubmitted, fill next morning
    -> Action: confirm fills next trading day with dry-run
```

**Duration:** 5 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 5 — Manually trigger one scheduled task

**Purpose:** Verify Task Scheduler can run your scripts correctly.

**Steps:**
```powershell
# Force-run the morning health check task right now:
schtasks /Run /TN "IBKR Health Check - Morning"

# Wait 10 seconds for it to complete
Start-Sleep -Seconds 10

# Check the log file was written:
Get-Content "logs\health\*.log" -Tail 20
```

**Expected:**
- Task Scheduler says "Task Running"
- Log file shows new entry with timestamp
- Verdict line: `VERDICT: PASS` or `VERDICT: WARN`

**If it fails:**
- Check Task Scheduler GUI: Task Scheduler Library → find "IBKR Health Check - Morning" → History tab
- Check if script path in task is correct
- Run the PowerShell setup script again if needed

**Duration:** 15 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 6 — Verify git credential protection

**Purpose:** Confirm the pre-commit hook blocks credential leaks.

**Steps:**
1. Create a test file with a fake secret:
   ```powershell
   cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
   "my_ibkr_account = 'DU123456'" | Out-File -FilePath "test_leak.txt" -Encoding utf8
   ```

2. Try to stage and commit it:
   ```powershell
   git add test_leak.txt
   git commit -m "test commit"
   ```

**Expected:**
```
PRE-COMMIT BLOCKED: Sensitive credential patterns detected

File: test_leak.txt
Match: DU123456 (line 1)
Pattern: IBKR account ID (DU* or DF*)

Recommendation:
  1. Remove the credential from the file
  2. Store it in C:\IBC\config.ini or .env instead
  3. Verify .gitignore covers this file pattern

Override (use with caution):
  git commit --no-verify
```

**Clean up:**
```powershell
git reset HEAD test_leak.txt
Remove-Item test_leak.txt
```

**If the hook doesn't block:** The hook may not be executable or is missing. Check:
```powershell
Test-Path ".git\hooks\pre-commit"
```

**Duration:** 20 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 7 — IBC autostart on Windows login (requires restart)

**Purpose:** Verify Gateway starts automatically when you log into Windows.

**Steps:**
1. Close IB Gateway completely (right-click taskbar → Exit)
2. Restart Windows
3. After seeing your desktop, wait 30 seconds

**Expected:**
- IB Gateway opens automatically without you doing anything
- Auto-logs into paper account
- API green light appears

**If it fails:**
- Check Startup folder: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\`
- Verify `IBC_Gateway_Autostart.bat` exists there
- Check Task Manager → Startup tab: the .bat file should be listed

**Duration:** 2 minutes (including Windows restart)

**Result:** [  ] PASS  [  ] FAIL (test this next, not right now)

---

## Test 8 — Full end-to-end validation script

**Purpose:** Run all 13 automated tests in sequence.

**Steps:**
```powershell
cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
uv run python scripts/test_system_end_to_end.py
```

**Expected:**
```
======================================================================
  TEST SUMMARY
======================================================================

  Total tests: 13
  Passed: 13
  Failed: 0

======================================================================
  ALL TESTS PASSED
  Your system is fully operational and ready for automated trading.
```

**If any test fails:** The script tells you exactly what to fix.

**Duration:** 15 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Test 9 — Run performance analysis (first baseline)

**Purpose:** Establish baseline metrics to track improvement over time.

**Steps:**
```powershell
uv run python scripts/performance_analysis.py
```

**Expected:** (takes 2–3 minutes on first run)
- Core metrics table (CAGR, Sharpe, Sortino, Calmar, MaxDD)
- Bear market protection section (downside capture, bear alpha, up/down capture ratio)
- Historical bear market episodes (2008, 2020, 2022, etc.)
- Regime breakdown (Expansion/Recession/Stagflation/Contraction)
- Auto-diagnosis: what is working / what to investigate / what is not working
- Hypothesis queue: specific experiment scripts to run if metrics are bad

**What to look for:**
- Downside capture < 70%? (good protection)
- Bear market alpha > 0%? (outperforms in down markets)
- Sortino > Sharpe? (asymmetric returns)

**Duration:** 2–3 minutes (first run only; subsequent runs are instant if data cached)

**Result:** [  ] PASS  [  ] FAIL

---

## Test 10 — Manual dry-run (preview rebalance orders)

**Purpose:** Verify the paper trading script can generate orders (even if 0 needed today).

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

Since your positions are at target (drift < tau), this should show **0 orders**. This is correct — no rebalance needed today.

**Duration:** 5 seconds

**Result:** [  ] PASS  [  ] FAIL

---

## Summary: What you just tested

| Test | Component | Status |
|------|-----------|--------|
| 1 | IBC launches Gateway and auto-logs in | [  ] |
| 2 | Python API connection to Gateway works | [  ] |
| 3 | Health check fetches positions correctly | [  ] |
| 4 | Trends report generates diagnostics | [  ] |
| 5 | Task Scheduler can run health check | [  ] |
| 6 | Git pre-commit hook blocks credentials | [  ] |
| 7 | IBC autostart on Windows login | [  ] |
| 8 | Full end-to-end validation (13 tests) | [  ] |
| 9 | Performance analysis computes metrics | [  ] |
| 10 | Dry-run executes and previews orders | [  ] |

---

## What to do based on results

### If all tests PASS:
**You're done.** Your system is fully automated. The only remaining step:
1. Test #7 (IBC autostart) — restart Windows once to confirm Gateway opens automatically
2. Wait until Monday 9:30 AM — morning health check runs automatically (check log afterward)
3. Monthly: run `performance_analysis.py` and review metrics

### If Test 1 FAILS (IBC won't start Gateway):
- Check credentials in `C:\IBC\config.ini`
- Check Gateway version: `TWS_MAJOR_VRSN` in `StartGateway.bat` must match installed Gateway build
- Review IBC logs: `C:\IBC\Logs\`

### If Test 3 FAILS (health check can't connect):
- Gateway API disabled → Gateway: Configure → Settings → API → Enable ActiveX and Socket Clients
- Port wrong → verify 4002 in Gateway settings matches `config/paper_trading.yaml`
- Credentials wrong → check Gateway shows paper account `DUM429899`

### If Test 5 FAILS (Task Scheduler can't run task):
- Re-run setup: `powershell -ExecutionPolicy Bypass -File scripts\setup_windows_scheduler.ps1`
- Check Task Scheduler GUI: verify tasks exist and "Run whether user is logged on or not" is NOT checked (should run only when logged in)

### If Test 6 FAILS (pre-commit hook doesn't block):
- Hook may not be executable
- Re-create: copy content from `docs/CONTINUOUS_IMPROVEMENT.md` section on git hooks
- Test again with a different fake credential pattern

---

## After all tests pass: The weekly routine

**No daily action required.** Health checks run automatically at 9:30 AM and 3:55 PM Mon–Fri.

**Every Friday (5 minutes):**
```powershell
# View weekly trends (automated at 4:15 PM, or run manually)
uv run python scripts/show_portfolio_trends.py
```

Check the "ISSUES & DIAGNOSES" section. If any WARN/ERROR items appear, follow the fix recommendations.

**First Monday of each month (30 minutes):**
1. Check if rebalance needed:
   ```powershell
   uv run python scripts/daily_health_check.py
   ```
   Exit code `1` = WARN → drift > tau → rebalance needed

2. If rebalancing (only if exit code = 1):
   ```powershell
   uv run python scripts/current_allocation.py  # refresh prices
   uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv  # preview
   uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv  # live submit (9:30 AM - 4:00 PM ET only)
   ```

**Last Friday of each month (30 minutes):**
```powershell
# Run performance analysis
uv run python scripts/performance_analysis.py
```

Review the "BEAR MARKET PROTECTION" section. If any metric is red (downside capture > 70%, bear alpha < 0%, Sortino < Sharpe), the script gives you specific experiments to run.

---

## Emergency contacts (if something breaks)

### Gateway hung or frozen
1. Task Manager → kill `ibgateway.exe`
2. Restart via IBC: `C:\IBC\StartGateway.bat`

### Health check keeps returning ERROR
1. Check Gateway is running: `tasklist | findstr gateway`
2. Check port: `netstat -ano | findstr "4002"`
3. Test connection: `uv run python -c "from ib_insync import IB; ib=IB(); ib.connect('127.0.0.1',4002,999,5); print(ib.managedAccounts())"`

### Scheduled task not running
1. Open Task Scheduler GUI (search "Task Scheduler" in Start menu)
2. Find "IBKR Health Check - Morning"
3. Right-click → Run
4. Check History tab for errors
5. If "Run only when user is logged on" is unchecked, check it (tasks won't have access to your session otherwise)

### Pre-commit hook blocks a safe file
Only if you're CERTAIN the file has no secrets:
```powershell
git commit --no-verify -m "your message"
```
Then immediately inspect what was committed: `git show HEAD`

---

## Quick command reference

Copy-paste these as needed:

```powershell
# === Startup ===
C:\IBC\StartGateway.bat                                                   # Start Gateway via IBC

# === Daily checks ===
cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
uv run python scripts/daily_health_check.py                              # Manual health check
uv run python scripts/show_portfolio_trends.py                           # Historical trends

# === Monthly rebalance ===
uv run python scripts/current_allocation.py                              # Refresh prices
uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv

# === Performance ===
uv run python scripts/performance_analysis.py                            # Full metrics + bear alpha
uv run python scripts/performance_analysis.py --since 2020-01-01         # Post-2020 only
uv run python scripts/performance_analysis.py --bench QQQ                # Use QQQ as benchmark

# === System validation ===
uv run python scripts/test_system_end_to_end.py                          # All 13 tests
uv run python scripts/test_system_end_to_end.py --skip-gateway           # Skip Gateway tests

# === Task Scheduler ===
schtasks /Run /TN "IBKR Health Check - Morning"                          # Trigger morning check now
schtasks /Query /TN "IBKR Health Check - Morning" /V                     # View task details
Get-Content "logs\health\*.log" -Tail 50                                  # Read health check logs

# === Troubleshooting ===
netstat -ano | findstr "4002"                                            # Check if port 4002 listening
tasklist | findstr -i gateway                                            # Check if Gateway running
Get-ChildItem C:\IBC\Logs\ | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content  # Latest IBC log
```

---

## Start here RIGHT NOW

1. **Test 1:** Start Gateway using `C:\IBC\StartGateway.bat`
2. **Test 3:** Run `daily_health_check.py` (should show PASS verdict)
3. **Test 8:** Run full validation `test_system_end_to_end.py` (should pass all 13 tests)
4. **Test 9:** Run `performance_analysis.py` (get baseline metrics)

After that, restart Windows (Test 7) to confirm IBC autostart works, and you're fully operational.

---

**Checklist template (fill in as you go):**

```
[  ] Test 1 — IBC starts Gateway and auto-logs in
[  ] Test 2 — Python API connects to Gateway
[  ] Test 3 — Health check reports positions vs targets
[  ] Test 4 — Trends report shows historical data
[  ] Test 5 — Task Scheduler runs health check manually
[  ] Test 6 — Pre-commit hook blocks fake credential
[  ] Test 7 — IBC autostart on Windows login (after restart)
[  ] Test 8 — Full end-to-end validation (13 tests)
[  ] Test 9 — Performance analysis (baseline metrics)
[  ] Test 10 — Dry-run previews orders correctly
```

Once all 10 boxes are checked, you're 100% automated.
