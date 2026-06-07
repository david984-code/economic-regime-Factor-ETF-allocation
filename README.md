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

Strategy outperforms 60/40 in 76% of months where 60/40 declines (`avg +0.52pp` in those months) and roughly matches in up months (50/50, `-0.09pp`). Delta-Sharpe vs 60/40 is `+0.131` with paired block-bootstrap two-sided `p = 0.30` (centered, 6mo blocks, 10,000 iterations) — not statistically significant at 5%. The 3pp drawdown reduction vs 60/40 (and 15pp vs SPY) is the cleaner defensible claim. Full statistical audit: [docs/bootstrap_reconciliation.md](docs/bootstrap_reconciliation.md).

## How it works

1. **Regime classification** (`src/models/regime_classifier.py`). FRED macro indicators (GDP, inflation, money supply, velocity, yields, PMI, claims, HY spreads) → standardized z-scores → continuous `risk_on` score ∈ [0, 1] and discrete regime label (Recovery / Overheating / Stagflation / Contraction).
2. **Per-regime optimization** (`src/allocation/optimizer.py`). Sortino objective over a 10-ETF universe (SPY, GLD, MTUM, VLUE, USMV, QUAL, IJR, VIG, IEF, TLT + cash) with regime-specific cash floors (5–30%) and minimum per-asset weights.
3. **Walk-forward evaluation** (`src/evaluation/walk_forward.py`). Expanding training window (≥60 months), monthly refit, ~116 OOS segments. Each OOS segment contributes its first novel month to a stitched non-overlapping return series. This is the source of every performance number quoted above.
4. **Live execution** (`src/execution/`). `auto_rebalance.py::run_auto_rebalance()` self-gates to the first trading day of the month, generates fresh target weights from the regime classifier + allocations, runs dry-run + safety checks, then submits paper orders to IBKR via the API. Scheduled via Windows Task Scheduler.

> **Note on the ML forecast module** (`src/models/regime_forecast.py`). A GradientBoosting next-month `risk_on` forecast model exists as experimental code but is **NOT in the live trading path or the published OOS numbers** as of 2026-06-07. It was previously blended into live weights, was never validated in the walk-forward backtest, and has been disconnected from the production pipeline. Treat as research/unvalidated.

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
  models/                    regime_classifier.py (used), regime_forecast.py (experimental)
  research/                  bootstrap significance, sensitivity sweeps
  utils/                     database, ticker universe, caching helpers
tests/                       pytest suite (53 tests)
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

## Methodology

- **Walk-forward design and OOS construction:** [docs/WALK_FORWARD_EVALUATION.md](docs/WALK_FORWARD_EVALUATION.md)
- **Statistical significance audit (centered block bootstrap):** [docs/bootstrap_reconciliation.md](docs/bootstrap_reconciliation.md)

## FRED API key

Set `FRED_API_KEY` in the environment or in a local `.env` file (gitignored). See `.env.example`.

## License

MIT — see [LICENSE](LICENSE).
