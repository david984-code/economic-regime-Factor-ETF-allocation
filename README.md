# Economic Regime Factor ETF Allocation

## Overview

Macro-driven, volatility-aware asset allocation framework for factor ETFs. The model ingests macroeconomic indicators (GDP growth, inflation, money supply, velocity, interest rates), transforms them into z-scores, and produces a continuous `risk_on` score (0 = fully defensive, 1 = fully risk-on). At each monthly rebalance, the portfolio blends between risk-on and risk-off base allocations using that score, then applies inverse-volatility scaling so high-volatility assets don't dominate portfolio risk.

## Quick Start

```bash
# Clone and install
git clone https://github.com/david984-code/economic-regime-Factor-ETF-allocation.git
cd economic-regime-Factor-ETF-allocation
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -e ".[dev]"

# Set your FRED API key
cp .env.example .env
# Edit .env with your key from https://fred.stlouisfed.org/docs/api/api_key.html

# Run the daily pipeline
python run_daily_update.py

# Run with custom tickers
python -m src.pipeline --tickers SPY,GLD,TLT

# Dry run (no trades, no file writes)
python -m src.pipeline --dry-run

# Load alternative config
python -m src.pipeline --config config/paper_trading.yaml
```

## Project Structure

```text
src/
  config.py                 # Central configuration (tickers, paths, constraints)
  pipeline.py               # Daily pipeline orchestration
  models/
    regime_classifier.py    # Rule-based economic regime classification
    regime_forecast.py      # ML regime forecast (next month)
  allocation/
    optimizer.py            # Sortino optimization per regime
    vol_scaling.py          # Inverse-volatility weight scaling
  backtest/
    engine.py               # Vectorized backtest engine
    metrics.py              # Performance metrics (CAGR, Sharpe, MaxDD, etc.)
  data/
    fred_ingestion.py       # FRED API data with local caching
    market_ingestion.py     # yfinance market data
    pipeline_data.py        # Shared data layer (fetch once, reuse)
  features/
    macro_features.py       # Macro z-scores, regime score
    market_features.py      # Momentum, seasonality
    transforms.py           # Rolling z-score, sigmoid, month-end alignment
  evaluation/
    walk_forward.py         # Walk-forward OOS evaluation
    benchmarks.py           # Benchmark return series
  execution/
    auto_rebalance.py       # IBKR paper trading auto-rebalance
  utils/
    fred_key.py             # FRED API key resolution
    ticker_universe.py      # Ticker list resolution (CLI > env > config)
    cache.py                # File-based caching
    retry.py                # Exponential backoff retry
    database.py             # SQLite persistence
    logging_config.py       # Centralized logging

scripts/                    # Experiment runners and diagnostics
tests/                      # Unit and integration tests
config/                     # YAML configs (paper trading, etc.)
outputs/                    # Generated at runtime (gitignored)

pyproject.toml              # Dependencies, ruff, mypy config
.env                        # API keys (local only, gitignored)
```

## Development

```bash
# Lint and format
ruff check . --fix
ruff format .

# Type check
mypy src/

# Run tests
pytest tests/ -v
```

<!-- TODO: Future: migrate results/ output to S3 or SQLite to keep repo clean -->
