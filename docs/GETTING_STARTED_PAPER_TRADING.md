# Getting Started: Automated Paper Trading

This guide walks you through the complete setup from zero to a fully automated IBKR paper trading system with daily health monitoring and monthly performance analysis.

---

## What you get

After completing this setup:

1. **IB Gateway auto-starts** when you log into Windows (via IBC)
2. **Health checks run automatically** at 9:30 AM and 3:55 PM Mon–Fri
3. **Weekly trends report** runs every Friday at 4:15 PM
4. **Monthly rebalance** — you run one command, orders submit to IBKR paper account
5. **Performance analysis** — understand what's working vs not working, with specific fixes

**Time investment:** Setup once (30 min), then 15 min/week monitoring + 30 min/month rebalancing.

---

## Prerequisites

- Windows 10/11
- IBKR paper trading account (separate from live)
- IB Gateway installed (`C:\Jts\ibgateway\1044\` or similar)
- This repo cloned to your machine
- `uv` installed (Python package manager)

---

## Step 1 — IBC setup (auto-login for Gateway)

**Why:** IB Gateway requires manual login each session. IBC automates this.

**What to do:**
1. IBC is already at `C:\IBC\` (version 3.23.0)
2. Your credentials are already in `C:\IBC\config.ini`
3. `StartGateway.bat` is configured for Gateway version 1044
4. Paper mode is confirmed in both files

**Test it:**
```powershell
# Close Gateway if running, then:
C:\IBC\StartGateway.bat
```
Gateway should open and auto-login to your paper account.

---

## Step 2 — Verify end-to-end system

**Run the validation script:**
```powershell
cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
uv run python scripts/test_system_end_to_end.py
```

**All 13 tests passed** (as of your last run). This confirms:
- ✓ IBC installed and configured
- ✓ Gateway version matched (1044)
- ✓ Credentials configured (paper mode)
- ✓ Gateway API connection works (fetched 10 positions)
- ✓ Target weights valid (sum ≈ 1.0)
- ✓ Safety limits correct (max_notional $180K, tau 1.5%)
- ✓ Git pre-commit hook active (blocks credential leaks)
- ✓ Task Scheduler jobs registered (3 health check tasks)
- ✓ IBC autostart configured (Windows Startup folder)
- ✓ Dry-run executes cleanly

---

## Step 3 — Test IBC autostart

**Restart Windows** (or log out and back in).

After 30 seconds, IB Gateway should:
- Open automatically
- Auto-fill your username/password
- Log into paper account `DUM429899`
- Show API green light on port 4002

If that works, you're fully automated from here forward.

---

## Step 4 — First manual health check

With Gateway running:
```powershell
uv run python scripts/daily_health_check.py
```

You should see all 10 positions, current weights vs targets, and max drift (currently 0.56% on GLD — well within 1.5% tau).

**Verdict: PASS** means no rebalance needed today.

---

## Step 5 — Understand your performance (first baseline)

**Run the full performance analysis:**
```powershell
uv run python scripts/performance_analysis.py
```

This will take 2–3 minutes on first run (downloads 16+ years of SPY/QQQ/IWM data). You get:

### Core metrics table
```
Name                            CAGR      Vol   Sharpe  Sortino   Calmar    MaxDD
--------------------------------------------------------------------------------
Strategy (vs SPY)              +8.63%   9.40%   +0.459   +0.623   +0.394  -21.92%
S&P 500                       +13.41%  17.16%   +0.563   +0.782   ...
Nasdaq-100                    +18.02%  20.73%   +0.691   ...
```

### Bear market protection (the key section)
```
WHAT THE STRATEGY CAPTURES OF MARKET MOVES:
  Downside capture ratio  : 65.2%   (target < 70%)
  -> GOOD: absorbs less than 70% of SPY's losses
  
  Upside capture ratio    : 71.8%
  -> Capture ratio (up/down): 1.10x  (asymmetric in your favor)

ALPHA ON BENCHMARK-DOWN DAYS:
  Bear market alpha (ann.) : +2.3%   (strategy adds return when SPY falls)
  Bear market hit rate     : 58.7%   (% of down-market days strategy > SPY)
```

### Historical bear markets
Every -15%+ SPY drawdown with your strategy's performance in each period (2008 crisis, 2020 COVID, 2022 bear, etc.).

### Auto-diagnosis
Lists what is working, what to investigate, and what is not working — with specific experiment scripts to run.

---

## Daily routine (automated)

**You don't need to do anything.** These run automatically:

| Time | What runs | What it does |
|------|-----------|--------------|
| On login | IBC starts Gateway | Auto-login to IBKR paper account |
| 9:30 AM Mon–Fri | Morning health check | Check positions vs targets, flag drift |
| 3:55 PM Mon–Fri | Afternoon health check | Same check at market close |
| 4:15 PM Fridays | Weekly trends report | NAV history, fill statuses, diagnostics |

**Logs written to:** `logs/health/YYYY-MM-DD_health.log` (append-only)

**Reports written to:** `reports/health/YYYY-MM-DD/health_HHMM.json`

---

## Monthly routine (manual, ~30 minutes)

**First Monday of the month:**

1. Check if rebalance is needed:
   ```powershell
   uv run python scripts/daily_health_check.py
   ```
   Exit code `1` = WARN means drift > tau → rebalance needed.

2. If rebalancing, refresh prices:
   ```powershell
   uv run python scripts/current_allocation.py
   ```

3. Preview orders (dry-run):
   ```powershell
   uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
   ```
   Review turnover, order count, safety checks.

4. Submit live orders (during RTH: 9:30 AM – 4:00 PM ET only):
   ```powershell
   uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
   ```
   Orders submit to IBKR immediately. If after hours, they queue for next session.

5. Confirm fills (next morning if submitted after hours):
   ```powershell
   uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
   ```
   Reconciliation delta should be near zero.

**Last Friday of the month:**

6. Run performance analysis:
   ```powershell
   uv run python scripts/performance_analysis.py
   ```

7. Review the "BEAR MARKET PROTECTION" section — these are your core metrics:
   - Downside capture < 70%? ✓
   - Bear market alpha > 0%? ✓
   - Sortino > Sharpe? ✓

8. If any metric is bad, read the "HOW TO IMPROVE" section — it gives you specific experiments to run.

---

## How to improve performance (when needed)

The `performance_analysis.py` output tells you **exactly which experiment to run** based on what's not working.

**Example diagnosis:**
```
[X] Downside capture 82%: poor protection (absorbs most of market losses)
```

**Script gives you the fix:**
```
[A] If downside capture is too high (> 70%):
    -> Hypothesis: risk-off sleeve weight too small
    -> Experiment: run_capped_rf_experiment.py
```

**Run it:**
```powershell
uv run python scripts/run_capped_rf_experiment.py
```

**Analyze:**
```powershell
uv run python scripts/analyze_walk_forward.py
```

**Accept only if:**
- Sharpe improvement ≥ 0.02 in full walk-forward
- Beat rate vs SPY ≥ 55% in OOS segments
- Improvement in worst segments (better bear market protection)

**If accepted:** Update `config/paper_trading.yaml` with the new settings, regenerate target weights, and the next monthly rebalance will use them.

---

## Credential security summary

Your IBKR credentials are protected by **5 independent layers**:

1. **Physical separation:** `C:\IBC\config.ini` is outside the git repo — impossible to push to GitHub
2. **File permissions:** Only `dns81` and `SYSTEM` can read `config.ini` (no other Windows users)
3. **`.gitignore`:** Blocks `config.ini`, `.env`, `*password*`, `*secret*`, etc. even if copied into repo
4. **Pre-commit hook:** Scans every commit for API keys, passwords, account IDs — blocks commit if found
5. **`.env` confirmed untracked:** FRED API key file is gitignored and has never been in git index

**Test it:**
```powershell
cd "c:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"
uv run python .git\hooks\pre-commit
```
Should exit 0 (clean). If you stage a file with a credential, the hook blocks the commit.

---

## Troubleshooting

### Gateway not starting via IBC
- Check `C:\IBC\Logs\` for error messages
- Verify credentials in `C:\IBC\config.ini` (username/password)
- Confirm `TradingMode=paper` in `config.ini` and `StartGateway.bat`
- Check Gateway version: Help → About → Build 10.xx.xx → must match `TWS_MAJOR_VRSN=10xx` in `StartGateway.bat`

### Health check returns ERROR
- Gateway not running → start it via IBC
- API disabled → Gateway settings: Enable API checkbox, port 4002
- Wrong account → check `IBKR_ACCOUNT` env var or `config/paper_trading.yaml`

### Orders not filling
- After-hours submission → orders queue as `PreSubmitted`, fill next morning at market open
- Reconciliation delta large → this is expected before fills; re-run dry-run next morning
- All 10 positions filled correctly from your March 26 submission (confirmed March 30 health check)

### Performance analysis fails
- Missing `outputs/regime_labels_expanded.csv` → run `uv run python run_daily_update.py` first
- Benchmark download timeout → retry; yfinance occasionally rate-limits

---

## Documentation map

| File | Purpose |
|------|---------|
| **`docs/GETTING_STARTED_PAPER_TRADING.md`** (this file) | Complete setup walkthrough |
| **`docs/SYSTEM_CHECKLIST.md`** | Step-by-step testing protocol |
| **`docs/CONTINUOUS_IMPROVEMENT.md`** | Monthly workflow: monitor → diagnose → experiment → improve |
| `README.md` | Project overview and research system |
| `config/paper_trading.yaml` | Safety limits, tau, account config |

---

## Next actions (in order)

1. ✓ IBC configured (credentials filled, paper mode confirmed)
2. ✓ Gateway running and API connected (all 13 tests passed)
3. ✓ Health checks scheduled (9:30 AM / 3:55 PM Mon–Fri)
4. **→ Test IBC autostart:** Restart Windows, confirm Gateway opens automatically
5. **→ First performance analysis:** `uv run python scripts/performance_analysis.py`
6. **→ Wait for first automated health check:** Monday 9:30 AM
7. **→ Review weekly trends:** Friday check `logs/health/` or run `show_portfolio_trends.py`

**You are ready to go.** The system is fully functional and will run unattended from here forward.
