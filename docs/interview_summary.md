# Building a Macro Regime-Based ETF Allocation System

**David Shaltakoff** | UF Finance/Accounting + Computer Science | AG3 Capital Equity Research

---

## What I Built

A fully automated systematic allocation model that classifies the economic regime from FRED macro data and dynamically allocates across 10 ETFs (7 risk-on equity factors + 3 risk-off: bonds, gold). The system runs daily, rebalances monthly, and currently paper-trades a $1M portfolio on Interactive Brokers.

**Pipeline:** FRED macro ingestion → regime classification (GDP/CPI z-scores) → ML regime forecast (Gradient Boosting) → Sortino optimization per regime → inverse-vol scaling → execution via IBKR API.

**Tooling I built on top:** automated parameter sensitivity scanner (192-combination grid search through walk-forward validation), on-demand status dashboard, daily performance reporting, and auto-rebalance with pre-trade safety checks.

---

## The Signal: What Works and What Doesn't

The core signal maps macro indicators to a continuous risk-on score: `risk_on = sigmoid(macro_score * 0.25)`, where macro_score combines z-scores of GDP growth, CPI, M2, velocity, and the yield curve.

**What works:** The signal correctly identifies bear markets. In 2018, 2022, and 2026 YTD, the model reduced equity exposure before or during the drawdown. Bear alpha is 130% annualized with an 81% hit rate on benchmark-down days.

**What doesn't work:** The signal never makes a decisive risk-on call during bull markets. I ran a year-by-year analysis and found that in every bull year from 2017 to 2025, the average risk_on score stayed between 0.37 and 0.53. The model missed every rally.

**Why:** The z-scores use an expanding window normalization. In a secular bull market where GDP growth is persistently positive, the expanding mean absorbs the growth signal, and the z-score converges to zero. `sigmoid(0) = 0.5` — permanent neutrality. This is a fundamental limitation of expanding z-score normalization on trending macro data. A fixed-window or percentile-based normalization would preserve more signal variance, but I tested this (via the hybrid market signal) and the improvement was marginal (+0.001 Sharpe) because the market signal has the same structural issue — positive momentum is "normal" in a secular bull, so it also z-scores to near zero.

**The takeaway:** Macro regime models are structurally better at identifying risk-off environments than risk-on. This isn't a bug in my implementation — it's a property of how macro indicators behave in trending economies. The correct use of this model is as a defensive overlay, not a standalone alpha generator.

---

## Three Bugs That Changed Everything

### 1. Six Months of Flying Blind (Sharpe impact: unmeasurable)

The regime classifier was returning "Unknown" for the last 6 months because FRED GDP data is published quarterly with a 2-month lag. When GDP stopped at October 2025, the monthly resampling produced NaN, which cascaded through z-scores to NaN regime labels. The model was allocating based on fallback weights with no regime signal.

**Fix:** Forward-fill quarterly series (GDP, velocity) across the monthly date range before computing z-scores. This is standard practice — GDP doesn't "disappear" between releases. Also made the macro score robust to partial NaN by filling missing z-scores with 0 (neutral) instead of letting one NaN poison the entire score.

**Lesson:** Data pipeline bugs are invisible in backtests because historical data is complete. They only appear in live systems where data has publication lag. Every data source needs an explicit staleness check.

### 2. The Optimizer Loved Gold (Sharpe: 0.675 → 0.897)

The Sortino optimizer allocated 51% to gold during Recovery — the regime where you'd expect maximum equity exposure. During Stagflation, it held 82% in small-cap equity (IJR + MTUM). The allocations were inverted.

**Why it happened:** The Sortino ratio penalizes downside deviation only. Gold had strong risk-adjusted returns during periods that happened to be classified as Recovery in the training data, so the optimizer loaded up on it. This is textbook overfitting to in-sample correlations — the optimizer found the historically best Sortino portfolio, not the economically sensible one.

**Fix:** Added regime-specific constraints: Recovery must hold ≥55% equity, GLD capped at 20%. Stagflation must hold ≤40% equity, ≥40% bonds/gold, ≥20% cash. These constraints encode economic priors that the optimizer can't derive from data alone.

**Result:** Max drawdown improved from -34.7% to -18.6% (the old Stagflation allocation was 82% small-cap equity during COVID). Sharpe went from 0.675 to 0.897. The constraints didn't just fix the allocation — they improved the model, because they prevented the optimizer from fitting to noise.

**Lesson:** Unconstrained optimizers will always overfit. The role of constraints isn't to limit the optimizer — it's to inject domain knowledge that the data can't provide. An optimizer that puts 50% in gold during Recovery has found a statistical artifact, not an insight.

### 3. Dead Code That Wasn't Dead (Sharpe: 0 impact, but matters)

The 200-day SMA crash gate was supposed to cap risk_on at 0.3 when SPY was below its 200-day moving average. When I ran the sensitivity scanner, all three trend filter variants (none, 200dma, 10mma) produced identical results. The crash gate had zero effect.

**Why:** The trend filter was only applied inside the hybrid signal code path, which requires `use_hybrid_signal=True`. In the baseline model (pure macro signal), the code was unreachable. The feature was "implemented" but never executed.

**Lesson:** Backtests don't test code paths that aren't triggered. A feature that appears to "not help" might actually be dead code. Always verify a parameter change actually changes the output before concluding it has no effect.

---

## Results

### Final Model (after fixes, with regime smoothing window=2)

| Metric | Strategy | SPY | 60/40 |
|---|---|---|---|
| **Sharpe** | **0.97** | 0.81 | 0.89 |
| **Sortino** | **1.30** | 0.99 | 1.12 |
| **Max Drawdown** | **-18.6%** | -33.7% | -21.0% |
| CAGR | 7.0% | 13.2% | 8.9% |
| Volatility | 7.3% | 17.1% | 10.1% |

The strategy beats both benchmarks on every risk-adjusted metric. It underperforms on raw CAGR because it runs at half the volatility. Vol-targeted at 15% (2.1x leverage), the CAGR matches SPY at 13.1% while maintaining the 0.97 Sharpe.

**50/50 core-satellite blend (SPY + Strategy):** Sharpe 0.93, CAGR 10.3%, Max DD -22.0% — beats 60/40 on every metric including CAGR.

**YTD 2026:** Strategy +0.7%, SPY -4.4%, 60/40 -2.6%. The model is working in real-time during a drawdown.

### What I Tested and Rejected

| Experiment | Result | Why |
|---|---|---|
| 200-day SMA crash gate | -0.011 Sharpe | Caps upside during V-recoveries more than it protects on the downside |
| Sigmoid scale (0.15 to 0.50) | Zero effect | Expanding z-scores cluster near zero, so sigmoid scale doesn't differentiate |
| Tolerance band 0.01 vs 0.015 | +0.008 Sharpe | Below the +0.02 significance threshold — noise |
| Pure market momentum signal | +0.001 Sharpe | Same expanding z-score problem as macro signal |
| Hybrid macro + market | +0.000 Sharpe | Two near-zero signals blended = one near-zero signal |

These rejections are as informative as the acceptances. The parameter sensitivity surface is flat — the model's behavior is driven by the regime classification and allocation constraints, not the signal mapping. This tells me where future alpha is (and isn't).

---

## What I'd Do Differently

1. **Start with the benchmark comparison, not the backtest.** I spent weeks optimizing Sharpe before realizing the model couldn't beat 60/40. The first question should always be: does this beat the simplest alternative?

2. **Constrain the optimizer from day one.** Letting an unconstrained Sortino optimizer choose allocations cost me the worst single bug (-34.7% drawdown from 82% small-cap in Stagflation). Economic priors belong in the optimization, not as an afterthought.

3. **Build the sensitivity scanner first.** Manual one-off experiments are slow and don't reveal interaction effects. The automated 192-combination grid search found in 5 minutes what would have taken weeks of manual testing — including the discovery that sigmoid scale has zero effect.

---

## Architecture

```
run_daily_update.py → src/pipeline.py
  ├── Regime Classification (FRED → z-scores → regime label)
  ├── ML Regime Forecast (Gradient Boosting, walk-forward CV)
  ├── Sortino Optimization (per-regime, with constraints)
  ├── Backtest (vol-scaled, tolerance-filtered)
  ├── Auto-Rebalance (1st trading day of month → IBKR paper)
  └── Daily Report (JSON + optional SMS)
```

**Stack:** Python, pandas, Polars, scipy, scikit-learn, yfinance, fredapi, ib_insync, SQLite. Package management via uv. Code quality enforced by ruff + mypy strict.

**Walk-forward validation:** 124 expanding-window segments, 60-month minimum training, 12-month test periods, 2013-2026 out-of-sample. No lookahead bias — all signals use only data available at time t.
