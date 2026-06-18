# Continuous Improvement Workflow

This document describes the systematic process for monitoring what is working vs not working in your strategy, and how to make data-driven improvements.

---

## Overview: The feedback loop

```
Daily       → Health checks (automated)
              ├─ Positions vs targets
              ├─ Drift monitoring
              └─ Gateway connectivity

Weekly      → Trends analysis
              ├─ NAV time series
              ├─ Order fill history
              └─ Reconciliation quality

Monthly     → Performance analysis
              ├─ Risk-adjusted metrics
              ├─ Bear market protection
              ├─ Regime-level breakdown
              └─ Actionable experiment queue

Quarterly   → Full walk-forward validation
              └─ Compare experiment results vs baseline
```

---

## Daily: Automated health checks

**What runs:** `scripts/daily_health_check.py` at 9:30 AM and 3:55 PM ET (Mon–Fri, via Task Scheduler)

**What it checks:**
- IB Gateway connectivity (port 4002)
- All 10 positions fetched from IBKR
- Current weight vs target weight for each symbol
- Max drift across all positions
- Whether any symbol exceeds the 1.5% tau no-trade band

**Output:**
- `reports/health/YYYY-MM-DD/health_HHMM.json` (structured)
- `logs/health/YYYY-MM-DD_health.log` (append-only audit trail)

**Exit codes:**
- `0` = PASS: all positions within tau, no action needed
- `1` = WARN: one or more positions drifted > tau → rebalance candidate
- `2` = ERROR: Gateway down or API connection failed

**When to act:**
- Exit code `1` (WARN) → plan a live rebalance submission on the next monthly rebalance date
- Exit code `2` (ERROR) → check Gateway is running and logged in

---

## Weekly: Trend analysis (every Friday 4:15 PM)

**What runs:** `scripts/show_portfolio_trends.py`

**What it shows:**
1. **Live submission history** — every run, order counts, fill statuses, turnover
2. **NAV time series** — estimated P&L since first health check
3. **What is working** — 8 items that are confirmed functional (connectivity, safety, fills, duplicate-run protection, etc.)
4. **Issues and diagnoses** — detected problems with severity (INFO / WARN / ERROR) and fixes
5. **Recommendations** — actionable next steps (reconciliation timing, price CSV refresh, IBC setup, etc.)

**When to act:**
- Review the "ISSUES & DIAGNOSES" section for any WARN/ERROR items
- If you see "After-hours submission" → confirm fills the next morning with a dry-run
- If you see repeated "Gateway was down" → verify IBC autostart is working

**Manual run:**
```powershell
uv run python scripts/show_portfolio_trends.py
uv run python scripts/show_portfolio_trends.py --since 2026-03-01  # filter by date
uv run python scripts/show_portfolio_trends.py --json > trends.json  # machine-readable
```

---

## Monthly: Performance analysis and experiment design

**What runs:** `scripts/performance_analysis.py` (manual, after each month-end)

**What it calculates:**
1. **Core metrics** — CAGR, Vol, Sharpe, Sortino, Calmar, MaxDD, Ulcer Index (vs SPY/QQQ/IWM/AGG/GLD)
2. **Bear market protection** (the most important section):
   - **Downside capture ratio** — what % of SPY's losses does the strategy absorb?
   - **Upside capture ratio** — what % of SPY's rallies does the strategy capture?
   - **Bear market alpha** — annualised excess return when SPY is negative
   - **Bear hit rate** — % of down-market days where strategy > SPY
   - **Beta** — systematic exposure to SPY (target 0.5–0.7)
   - **Alpha (Jensen's)** — risk-adjusted excess return after controlling for beta
   - **Information ratio** — active return / tracking error (skill measure)
3. **Historical bear market episodes** — strategy vs SPY in every -15%+ drawdown (2008, 2020, etc.)
4. **Regime-level breakdown** — CAGR/Sharpe/Alpha in each macro regime (Expansion/Contraction/Stagflation/Recession)
5. **Auto-diagnosis** — what is working / what to investigate / what is not working
6. **Experiment queue** — specific scripts to run to fix detected issues

**When to run:**
```powershell
# Full analysis (all history)
uv run python scripts/performance_analysis.py

# Post-2020 only (recent regime shifts)
uv run python scripts/performance_analysis.py --since 2020-01-01

# Use QQQ as primary benchmark (higher-vol reference)
uv run python scripts/performance_analysis.py --bench QQQ

# Save metrics to CSV for Excel charts
uv run python scripts/performance_analysis.py --save-csv outputs/monthly_perf_report.csv
```

**What good looks like:**
| Metric | Target | Why this matters |
|--------|--------|------------------|
| **Downside capture** | < 70% | Strategy absorbs less than 70 cents of every dollar SPY loses |
| **Upside capture** | > 60% | Still participates in market rallies |
| **Bear market alpha** | > 0% ann | Positive excess return specifically when SPY is down (core objective) |
| **Sortino** | > Sharpe | Reward/downside-risk ratio better than generic risk (asymmetric) |
| **Calmar** | > 0.4 | CAGR at least 40% of max drawdown (fast recovery) |
| **Beta** | 0.5–0.7 | Lower market sensitivity than SPY (diversification benefit) |
| **Information ratio** | > 0.5 | Strong evidence of skill after controlling for active risk |

---

## When metrics are bad: How to fix

The `performance_analysis.py` script gives you a prioritised experiment queue at the end. Here is the general decision tree:

### Problem 1: Downside capture > 70% (poor bear market protection)

**Root cause:** Risk-off sleeve (bonds/gold) weight too low, or tau band too wide so equity exposure stays on during early drawdowns.

**Experiments to run:**
```powershell
# Hypothesis: increase risk-off allocation cap
uv run python scripts/run_capped_rf_experiment.py

# Hypothesis: faster rebalance response (tighter tau)
uv run python scripts/run_tau_020_experiment.py

# Hypothesis: add TIP to risk-off sleeve (inflation protection)
uv run python scripts/run_tip_addition_experiment.py
```

**After each experiment:** compare vs baseline with `scripts/analyze_walk_forward.py`

---

### Problem 2: Bear market alpha < 0% (strategy underperforms in down markets)

**Root cause:** Regime signal too slow (24M lookback misses early recession signals).

**Experiments to run:**
```powershell
# Hypothesis: faster regime detection
uv run python scripts/run_12m_lookback_experiment.py

# Hypothesis: early warning trend filter
uv run python scripts/run_200dma_experiment.py

# Hypothesis: yield curve inversion as early warning
uv run python scripts/run_inversion_flag_experiment.py
```

---

### Problem 3: Sortino < Sharpe (losses worse than gains)

**Root cause:** Equal-weight construction treats high-vol and low-vol assets the same.

**Experiments to run:**
```powershell
# Hypothesis: inverse-vol weighting reduces tail losses
uv run python scripts/run_invvol_ablation_experiment.py

# Hypothesis: explicit vol targeting at 8%
uv run python scripts/run_vol_target_experiment.py
```

---

### Problem 4: Calmar < 0.4 (slow recovery from drawdowns)

**Root cause:** Max drawdown too large; need faster de-risking.

**Experiments to run:**
```powershell
# Hypothesis: smoother regime labels reduce whipsaw
uv run python scripts/run_regime_smoothing_experiment.py

# Hypothesis: breadth as de-risk trigger
uv run python scripts/run_breadth_flag_experiment.py
```

---

### Problem 5: Upside capture < 50% (missing rallies)

**Root cause:** Strategy too defensive in risk-on regimes.

**Experiments to run:**
```powershell
# Hypothesis: increase risk-on allocation ceiling
uv run python scripts/run_sigmoid_scale_experiment.py

# Hypothesis: expand risk-on universe
uv run python scripts/run_ro_expansion_experiment.py
```

---

## Quarterly: Walk-forward validation and experiment comparison

**When:** After running any experiment (fast-mode screen → full WF if promising).

**Scripts:**
```powershell
# 1. Run experiment (example: 12M lookback)
uv run python scripts/run_12m_lookback_experiment.py

# 2. Analyze robustness (segments, beat rates, regime breakdown)
uv run python scripts/analyze_walk_forward.py

# 3. If Sharpe improvement > 0.02, run full validation
uv run python scripts/run_12m_lookback_full_validation.py
```

**What to check in `analyze_walk_forward.py` output:**
- **Segment beat rates** — % of OOS segments where experiment beats SPY
- **Worst segments** — does the experiment protect better in bear markets?
- **Regime breakdown** — is improvement concentrated in one regime, or robust across all?
- **Warning signs** — >20% of segments with negative Sharpe = unstable

**Accept criteria:**
- Sharpe improvement ≥ 0.02 vs baseline (after full WF)
- Beat rate vs SPY ≥ 55% in OOS segments
- No worse than baseline in worst 5 segments
- Improvement not just in one regime (must be robust)

---

## The full monthly checklist

Run this sequence at the start of each month:

1. **Check if rebalance is needed:**
   ```powershell
   uv run python scripts/daily_health_check.py
   ```
   If exit code = 1 (WARN), positions have drifted → proceed to step 2.

2. **Refresh prices CSV:**
   ```powershell
   uv run python scripts/current_allocation.py
   ```
   Generates fresh `paper_trading_validation_prices.csv`.

3. **Dry-run to preview orders:**
   ```powershell
   uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
   ```
   Review turnover, order count, safety checks.

4. **Live submission (RTH only):**
   ```powershell
   uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
   ```
   Run during 9:30 AM – 4:00 PM ET. Orders submit immediately.

5. **Confirm fills next day (if submitted after hours):**
   ```powershell
   uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
   ```
   Reconciliation delta should be near zero if all fills completed.

6. **Performance analysis (month-end):**
   ```powershell
   uv run python scripts/performance_analysis.py
   ```
   Review bear market alpha, downside capture, Sortino. If any metric is below target, queue an experiment.

7. **Run one experiment per month (if needed):**
   Pick from the experiment queue in the performance analysis output. Run fast-mode first, then full WF if promising. Never run more than one experiment per month (stay disciplined).

---

## Key files and what they do

| File | Purpose | When to run |
|------|---------|-------------|
| `scripts/test_system_end_to_end.py` | Validate IBC, Gateway, health checks, git security | After setup, or if something breaks |
| `scripts/daily_health_check.py` | Check positions vs targets, flag drift | Automated at 9:30 AM / 3:55 PM Mon–Fri |
| `scripts/show_portfolio_trends.py` | NAV history, order fills, diagnostics | Automated Fridays 4:15 PM, or manual anytime |
| `scripts/performance_analysis.py` | Risk-adjusted metrics, bear alpha, regime breakdown | Manual, monthly or after big market moves |
| `scripts/run_paper_trading.py` | Dry-run or live paper submission | Monthly rebalance (when health check shows drift > tau) |
| `scripts/current_allocation.py` | Generate fresh prices CSV | Before each live submission |
| `scripts/analyze_walk_forward.py` | WF robustness, segment beat rates | After any experiment to compare vs baseline |
| `scripts/benchmark_comparison_chart.py` | Strategy vs SPY/QQQ/IWM over 1Y/5Y/10Y/15Y/20Y | Manual, for presentations or quarterly reviews |

---

## What metrics to watch each week

Print this table and put it next to your desk:

| Metric | Current | Target | Traffic light |
|--------|---------|--------|---------------|
| **Downside capture** | ? | < 70% | 🟢 < 60%  🟡 60–80%  🔴 > 80% |
| **Bear market alpha** | ? | > 0% ann | 🟢 > +3%  🟡 0–3%  🔴 < 0% |
| **Sortino** | ? | > Sharpe | 🟢 > 1.5× Sharpe  🟡 > Sharpe  🔴 < Sharpe |
| **Calmar** | ? | > 0.4 | 🟢 > 0.6  🟡 0.4–0.6  🔴 < 0.4 |
| **Beta** | ? | 0.5–0.7 | 🟢 0.4–0.7  🟡 0.7–0.9  🔴 > 0.9 |
| **Information Ratio** | ? | > 0.5 | 🟢 > 0.7  🟡 0.4–0.7  🔴 < 0.4 |

Run `scripts/performance_analysis.py` monthly to update this table.

---

## Experiment workflow (when you need to improve a metric)

### Step 1 — Identify the problem metric
From `performance_analysis.py` output, find the "NOT WORKING" section. Example:
```
[X] Downside capture 82%: strategy absorbs most of market losses (poor protection)
```

### Step 2 — Pick one experiment from the queue
The script gives you a prioritised list. Example:
```
[A] If downside capture is too high (> 70%):
    -> Hypothesis: risk-off sleeve weight too small
    -> Experiment: run_capped_rf_experiment.py
```

### Step 3 — Run fast-mode screen
```powershell
uv run python scripts/run_capped_rf_experiment.py
```
This runs on recent data only (fast). Look for the **Sharpe improvement** at the end.

### Step 4 — Decision gate
- If Sharpe improvement **< 0.02**: reject, move to next experiment
- If Sharpe improvement **≥ 0.02**: proceed to full WF validation

### Step 5 — Full walk-forward validation (if passed gate)
```powershell
uv run python scripts/run_capped_rf_full_validation.py  # if exists
# Or re-run with --no-fast-mode flag
```

### Step 6 — Robustness analysis
```powershell
uv run python scripts/analyze_walk_forward.py
```
Check:
- Beat rate vs SPY in OOS segments
- Performance in worst 5 segments (did protection improve?)
- Regime breakdown (is improvement robust or one-regime only?)

### Step 7 — Accept or reject
**Accept if:**
- Full WF Sharpe improvement ≥ 0.02 vs baseline
- Beat rate vs SPY ≥ 55%
- Not worse than baseline in worst 5 segments
- Improvement visible in ≥ 2 regimes (not concentrated)

**If accepted:** Update baseline config in `config/paper_trading.yaml` with the new knob settings. Document the change in a log file. Re-run `current_allocation.py` to regenerate target weights with the new config.

**If rejected:** Revert, document why in a research log, move to the next hypothesis.

---

## Research log template (maintain in `docs/research_log.md`)

```markdown
## 2026-04-15 — Experiment: 12M lookback (faster regime detection)

**Hypothesis:** 24M lookback is too slow to detect recessions early; 12M should improve bear market protection.

**Config change:** `market_lookback_months: 24 → 12`

**Fast-mode results:**
- Sharpe: 0.48 → 0.51 (+0.03) ✓ passed gate
- Downside capture: 82% → 74%
- Bear alpha: -1.2% → +1.8% ann

**Full WF results:**
- Sharpe: 0.47 → 0.49 (+0.02) ✓ accept threshold
- Beat rate vs SPY: 58% OOS segments ✓
- Worst segment Sharpe: -0.12 → -0.08 (improvement ✓)
- Regime breakdown: improvement in Stagflation (+0.08 Sharpe) and Recession (+0.11 Sharpe) ✓

**Decision:** ACCEPT. Update baseline to 12M lookback.

**Action taken:**
- Updated `config/paper_trading.yaml`: `market_lookback_months: 12`
- Regenerated target weights: `uv run python scripts/current_allocation.py`
- Next rebalance will use new weights.
```

---

## When to NOT run an experiment

Discipline is key. Do not run experiments if:
- Current metrics are all green (nothing broken → nothing to fix)
- Less than 3 months since last accepted change (let it settle)
- No clear hypothesis (random exploration wastes time)
- Fast-mode Sharpe improvement < 0.02 (noise, not signal)

**The goal is NOT to constantly tweak the strategy.** The goal is to systematically detect and fix underperformance when it appears in the data.

---

## Summary: Your weekly routine

| Day | Time | Action | Command |
|-----|------|--------|---------|
| Mon–Fri | 9:30 AM | (Automated) Morning health check | Task Scheduler runs it |
| Mon–Fri | 3:55 PM | (Automated) Afternoon health check | Task Scheduler runs it |
| Friday | 4:15 PM | (Automated) Weekly trends report | Task Scheduler runs it |
| Friday | 4:30 PM | **[Manual]** Review trends output | Check logs/health/ or just wait for the log file to be written |
| Monthly | First Mon | **[Manual]** Check if rebalance needed | `uv run python scripts/daily_health_check.py` (exit code 1 = yes) |
| Monthly | First Mon | **[Manual]** Live rebalance if needed | `uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv` |
| Month-end | Any day | **[Manual]** Performance analysis | `uv run python scripts/performance_analysis.py` |
| As needed | Any day | **[Manual]** Run one experiment | Pick from performance analysis queue |

**Time investment:** 15 minutes/week (Friday review) + 30 minutes/month (rebalance + performance analysis).

---

## Emergency: What to do if the system breaks

### Gateway won't start
1. Check Task Manager for `ibgateway.exe` (kill if hung)
2. Check `C:\IBC\Logs\` for IBC error logs
3. Manually start Gateway outside IBC to verify credentials
4. Common fix: TWS_MAJOR_VRSN in StartGateway.bat is wrong

### Health check returns ERROR
1. Verify Gateway is running: `netstat -ano | findstr ":4002"`
2. Test connection manually: `uv run python scripts/test_ibkr_connection.py`
3. If that works, check Task Scheduler logs: `logs/health/`

### Pre-commit hook blocks a legitimate file
Only if you are **certain** the file has no secrets:
```powershell
git commit --no-verify -m "your message"
```
Then immediately check what was committed: `git diff HEAD~1 HEAD`

### Duplicate-run protection blocking a submission
Only occurs if you try to submit twice in the same calendar day. The marker file is:
```
reports/paper_trading/YYYY-MM-DD/last_paper_submission.json
```
Delete it only if you are certain the first submission failed and needs to be retried.

---

## Quick reference: All commands you need

```powershell
# --- Daily operations ---
uv run python scripts/daily_health_check.py
uv run python scripts/show_portfolio_trends.py

# --- Monthly rebalance ---
uv run python scripts/current_allocation.py                    # refresh prices
uv run python scripts/run_paper_trading.py --dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv
uv run python scripts/run_paper_trading.py --no-dry-run --prices outputs/rebalance/paper_trading_validation_prices.csv

# --- Performance analysis ---
uv run python scripts/performance_analysis.py
uv run python scripts/performance_analysis.py --since 2020-01-01
uv run python scripts/benchmark_comparison_chart.py

# --- Experiments (pick one) ---
uv run python scripts/run_12m_lookback_experiment.py
uv run python scripts/run_capped_rf_experiment.py
uv run python scripts/run_tau_020_experiment.py
uv run python scripts/run_invvol_ablation_experiment.py
uv run python scripts/analyze_walk_forward.py

# --- System validation ---
uv run python scripts/test_system_end_to_end.py
```

---

**End of continuous improvement workflow.**
