# Economic Regime Factor ETF Allocation

A macro-regime-aware ETF allocation system: classify the macro environment from FRED data, optimize a multi-asset portfolio under regime-specific constraints, rebalance monthly via the Interactive Brokers API.

## Out-of-sample performance

Walk-forward OOS, monthly refit, Sep 2015 → May 2026 (10.7 years, 129 months, ~116 refit segments).

|                              | **Strategy** | 60/40 (SPY/IEF) | SPY     | IEF     |
| ---------------------------- | -----------: | --------------: | ------: | ------: |
| CAGR                         |   **10.93%** |           9.73% |  15.25% |   1.16% |
| Annualized volatility        |    **9.98%** |          10.50% |  17.82% |   6.55% |
| Sharpe (excess, rf = 4.5%)   |    **0.649** |           0.518 |   0.639 |  -0.463 |
| Sortino (excess)             |    **0.620** |           0.493 |   0.604 |  -0.460 |
| Max drawdown                 |  **-18.27%** |         -21.28% | -33.72% | -23.92% |
| Calmar (CAGR / \|MaxDD\|)    |    **0.598** |           0.457 |   0.452 |   0.048 |
| Monthly CVaR (worst 5%)      |   **-5.21%** |          -5.46% |  -8.91% |     n/a |
| Beats benchmark in down mo.  |    **76.3%** |               — |       — |       — |

![Equity curve and drawdown](docs/figures/equity_drawdown.png)

**Read this before quoting any number above:** the only statistically robust outperformance claim vs 60/40 is the **76.3% down-month hit rate** (cluster-aware 95% CI [62%, 90%], well above the 50% null). The Sharpe delta (+0.13) is not significant (p = 0.30), the max-drawdown delta (-3pp) is not significant either (p = 0.108, CI crosses zero), and the +1.2pp CAGR edge is **inside the unmodeled transaction-cost band of 1.2–3.0pp/yr** — net of costs it could be zero or negative. A regression of strategy-minus-60/40 excess on GLD returns shows the entire excess is explained by the gold tilt (R² = 0.32, β_GLD = 0.19, residual alpha = −1.73pp/yr); a fairer benchmark including gold would close most of the gap. SPY's Sharpe (0.639) is statistically indistinguishable from the strategy's (0.649). The honest pitch is **lower volatility and a defensive overlay**, not Sharpe outperformance. Full audit: [docs/bootstrap_reconciliation.md](docs/bootstrap_reconciliation.md). The caveats section below names every known limitation explicitly.

## How it works

1. **Regime classification** (`src/models/regime_classifier.py`). FRED macro indicators (GDP, inflation, money supply, velocity, yields, PMI, claims, HY spreads) → standardized z-scores → continuous `risk_on` score ∈ [0, 1] and discrete regime label (Recovery / Overheating / Stagflation / Contraction).
2. **Per-regime optimization** (`src/allocation/optimizer.py`). Sortino objective over a 10-ETF universe (SPY, GLD, MTUM, VLUE, USMV, QUAL, IJR, VIG, IEF, TLT + cash) with regime-specific cash floors (5–30%) and minimum per-asset weights.
3. **Walk-forward evaluation** (`src/evaluation/walk_forward.py`). Expanding training window (≥60 months), monthly refit, ~116 OOS segments. Each OOS segment contributes its first novel month to a stitched non-overlapping return series. This is the source of every performance number quoted above.
4. **Live execution** (`src/execution/`). `auto_rebalance.py::run_auto_rebalance()` self-gates to the first trading day of the month, generates fresh target weights from the regime classifier + allocations, runs dry-run + safety checks, then submits paper orders to IBKR via the API. Scheduled via Windows Task Scheduler.

> **Note on previous ML overlay.** An earlier version blended a GradientBoosting next-month `risk_on` forecast into live weights. The layer never entered the walk-forward backtest, so it was an unvalidated degree of freedom on top of validated logic. Removed 2026-06-07. Git history preserves the model code if it is ever revisited; the published OOS numbers above are unaffected.

## Project layout

```text
src/
  pipeline.py                daily data → labels → allocations → backtest
  config.py                  tickers, dates, regime constraints, RF
  status.py                  CLI summary of latest run
  allocation/                weight blending, vol scaling, regime overrides, optimizer
  backtest/                  vectorized backtest engine
  data/                      market data + FRED macro ingestion
  evaluation/                walk-forward harness, benchmarks, metrics
  execution/                 IBKR adapter, monthly rebalance, safety, reconciliation
  features/                  macro feature engineering (z-scores, etc.)
  models/                    regime_classifier.py
  research/                  bootstrap significance, sensitivity sweeps
  utils/                     database, ticker universe, caching helpers
tests/                       pytest suite (73 tests)
docs/                        methodology + statistical-audit memos
scripts/                     run_walk_forward.py, analyze_walk_forward.py
config/                      paper_trading.yaml (IBKR connection, safety limits)
```

## Quick start

```bash
# Install
uv sync

# Set FRED API key
echo 'FRED_API_KEY=your_key_here' > .env

# Daily pipeline (regime classification + allocations)
python run_daily_update.py

# Walk-forward backtest
python scripts/run_walk_forward.py

# Tests
pytest
```

See [docs/QUICK_START.md](docs/QUICK_START.md) for more.

## Universe and data sources

**ETF universe (10 tickers + cash):** SPY, GLD, MTUM, VLUE, USMV, QUAL, IJR, VIG (risk-on sleeve); IEF, TLT (risk-off sleeve); GLD also acts as a real-asset / inflation hedge. Cash earns the configured risk-free rate. Daily prices via Yahoo Finance (`yfinance`).

**Macro features (FRED series IDs):**

| Indicator | FRED ID | Frequency |
|---|---|---|
| Real GDP | `GDPC1` / `GDP` | Quarterly |
| CPI (all items) | `CPIAUCSL` | Monthly |
| M2 money stock | `M2SL` | Monthly |
| M2 velocity | `M2V` | Quarterly |
| 10-year Treasury yield | `DGS10` | Daily |
| 3-month Treasury yield | `DGS3MO` | Daily |
| Industrial production | `INDPRO` | Monthly |
| ISM Manufacturing PMI | `NAPM` | Monthly |
| Initial jobless claims | `ICSA` | Weekly |
| HY credit OAS (BofA) | `BAMLH0A0HYM2` | Daily |

## Methodology and known caveats

This section names the sources of bias the strategy is and is not protected against. Read this before quoting any number from the table.

- **Walk-forward, monthly refit, expanding window.** All OOS numbers above are from `evaluation.walk_forward.collect_walk_forward_oos_returns`: train on data ≥60 months from `START_DATE` through month *T*, generate weights for month *T+1*, advance, repeat (~116 segments). The first novel month of each segment is stitched into the OOS return series. The default `backtest.engine.run_backtest()` is **in-sample** — its docstring warns; do not quote its numbers.
- **Publication lag (partially handled).** `regime_classifier.py` builds two pipelines: a "legacy" naive merge and a "publication-lag-aware" merge (`resample_to_monthly`). The latter aligns each indicator at its release-date timestamp, not the reference-period timestamp — so for a January 31 weight decision, the model only sees indicators that had actually been released by January 31. This handles the publication-timing axis. **The legacy pipeline is retained for comparison only; the production code path uses the asof-aware version.**
- **Vintage / revised data (not yet handled — known lookahead source).** FRED series are downloaded via `fredapi`'s `get_series` without `realtime_start`/`realtime_end` parameters, which returns the *latest revised value*. Initial GDP and CPI releases are often revised meaningfully (10–30bps on inflation; sometimes much more on GDP first-vs-third revisions). The "data available on date *T*" used in training is therefore revised data, not the print as released. **This is a real lookahead source.** Switching to ALFRED vintage releases is a known scope item; expect a small but non-zero hit to backtest Sharpe.
- **Transaction costs (not modeled — known drag source).** The walk-forward backtest assumes frictionless rebalancing. Live rebalance turnover averages 20–30% one-way per month per the dry-run preview. At a realistic 5–10 bps round-trip per dollar traded on a tight-spread ETF universe, this is 1.2–3.0% annualized drag. The published Sharpe / CAGR overstates by roughly this amount. The strategy still beats 60/40 on max drawdown and down-month hit rate (which are largely cost-insensitive), but the Sharpe edge specifically is within the unmodeled cost band.
- **Statistical significance — what survives, what does not.**
    - **Sharpe delta vs 60/40 = +0.131.** Paired centered block-bootstrap p = 0.30, 95% CI [-0.149, +0.360]. **Not significant at 5%.**
    - **Max drawdown delta = +3.0pp** (strategy -18.27% vs 60/40 -21.28%). Paired block-bootstrap (10k iter, 6-month blocks) gives 95% CI [-1.27pp, +6.62pp] and two-sided p = 0.108. **Also not significant at 5%** — directionally favorable but inside the noise band of one 10.7-year path. Single-path max-drawdown is the highest-variance statistic available; quoting "3pp reduction" without dispersion was overstating.
    - **Down-month hit rate vs 60/40 = 76.3%** (29 of 38 60/40 down months). **This one survives both naive and cluster-aware tests.** Naive binomial (assumes IID) gives p ≈ 0.0008; cluster-aware block bootstrap (resamples 6-month blocks to preserve drawdown-episode clustering) gives 95% CI on the hit rate of [62%, 90%] — entire interval is above the 50% null. **This is the only statistically robust outperformance claim.**
- **The gold-attribution test (single-asset concentration risk).** OLS regression of daily (strategy − 60/40) excess on daily GLD return gives R² = 0.32, β_GLD = 0.19, p(β) ≈ 0. The strategy's annualized excess over 60/40 is +0.93pp/yr; the gold-explained portion (β × mean(GLD)) is +2.66pp/yr; the **residual alpha after stripping gold is −1.73pp/yr** (intercept p = 0.21). **Translation: the strategy's outperformance vs 60/40 is fully explained by gold exposure; the regime classifier + Sortino optimizer, holding gold constant, would have *underperformed* 60/40.** A fairer benchmark that includes gold (e.g., 50% SPY / 30% IEF / 20% GLD) would substantially close the gap — possibly close it entirely. Reported because the reviewer is right that this is the question to ask, not because it flatters the strategy.
- **Costs erase the return edge.** CAGR edge over 60/40 is +1.2pp/yr (10.93% vs 9.73%). Modeled transaction-cost drag is 1.2–3.0pp/yr. **Post-cost, the CAGR edge is zero or negative.** The only post-cost-defensible claims are vol reduction (~half SPY's) and the cluster-significant down-month hit rate.
- **Why not just hold SPY?** Fair question. SPY's Sharpe (0.639) is statistically indistinguishable from the strategy's (0.649), and SPY's CAGR is 4.3pp/yr higher (15.25% vs 10.93%). **This strategy does not beat SPY on risk-adjusted return; the pitch is lower vol (10% vs 18%) and lower max drawdown (-18% vs -34%), nothing else.** Use case is for an investor whose loss-aversion makes that trade worth ~4pp/yr of foregone return — not for someone optimizing total wealth.
- **Live track record clock effectively resets 2026-06-07.** Live paper trading began 2026-05-01 under a hybrid system that blended an unvalidated GradientBoosting forecast into live weights. That overlay was removed 2026-06-07 (commit `c836972` + `75e973e`); the live system now matches the validated walk-forward path. Live OOS data *for the system whose numbers appear in this README* effectively starts 2026-06-07. Expect ~12 months of clean data before live-vs-backtest reconciliation has any signal.
- **Test coverage.** 73 tests pass. Direct known-answer coverage of: `regime_classifier`, `bootstrap_significance`, `allocation/optimizer` (Sortino math, cash floor/ceiling, weight normalization, long-only constraint, winning-asset selection), and `backtest/engine` helpers (`_blend_alloc` math, `_equal_weight_alloc` normalization, `_smooth_regime_labels` window correctness). These are the modules where signal-alignment and weight-normalization bugs would silently inflate edge.

Full methodology references:

- **Walk-forward design and OOS construction:** [docs/WALK_FORWARD_EVALUATION.md](docs/WALK_FORWARD_EVALUATION.md)
- **Statistical significance audit (centered block bootstrap):** [docs/bootstrap_reconciliation.md](docs/bootstrap_reconciliation.md)

## FRED API key

Set `FRED_API_KEY` in the environment or in a local `.env` file (gitignored). See `.env.example`.

## License

MIT — see [LICENSE](LICENSE).
