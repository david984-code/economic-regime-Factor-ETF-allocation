# Economic Regime Factor ETF Allocation

## Overview
This project implements a macro-driven, volatility-aware asset allocation framework designed to dynamically allocate across equity, factor, defensive, and cash sleeves. The model ingests a set of macroeconomic indicators (e.g., GDP growth, inflation, money supply, velocity, and interest-rate levels), transforms them into standardized z-scores, and aggregates them into a continuous `risk_on` score ranging from 0 (fully risk-off) to 1 (fully risk-on). Rather than relying solely on discrete economic regimes, this continuous signal allows the portfolio to scale exposure gradually as macro conditions improve or deteriorate.

At each monthly rebalance, the model blends between predefined risk-on and risk-off base allocations using the current `risk_on` score, then applies inverse-volatility scaling so that higher-volatility assets (e.g., momentum ETFs) do not dominate portfolio risk relative to lower-volatility sleeves (e.g., minimum-volatility or value factors). Daily returns are generated using forward-filled monthly signals, and portfolio performance is evaluated against an equal-weight benchmark. In addition to historical backtests, the script outputs current recommended portfolio weights as of the latest trading day.

---

## Project Structure
```text
src/
  economic_regime.py        # Macro regime classification (refactored as class)
  optimizer.py              # Portfolio optimization per regime
  backtest.py               # Historical backtest + performance metrics
  format_allocations.py     # Excel report formatting

tests/
  test_economic_regime.py   # Unit tests (16 tests covering core logic)

outputs/                    # Generated at runtime (not in git)
  regime_labels_expanded.csv
  optimal_allocations.csv
  optimal_allocations_formatted.xlsx
  current_factor_weights.csv

pyproject.toml              # Dependencies, build config, tool settings (uv, ruff, mypy)
.env                        # API keys (local only, not in git)
