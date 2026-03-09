# P0 Data Caching Architecture

## Overview

Market data (yfinance) and FRED series are now cached to eliminate repeated API calls across the pipeline. Prices are fetched **once per run** and reused by regime forecast, optimizer, and backtest.

## Architecture

```
Pipeline Start
     │
     ├── Step 1: Regime Classification (FRED only)
     │   └── fetch_fred_core, fetch_fred_optional (with in-memory cache)
     │
     ├── [FETCH MARKET DATA ONCE]
     │   └── PipelineData.get_prices() → yfinance (1 call)
     │
     ├── Step 2: Regime Forecast
     │   └── pipeline_data.get_momentum_features("SPY")  ← cache hit
     │
     ├── Step 3: Portfolio Optimization
     │   └── pipeline_data.get_monthly_returns()        ← cache hit
     │
     └── Step 4: Backtest
         └── pipeline_data.get_prices()                  ← cache hit
```

## Components

### 1. PipelineData (`src/data/pipeline_data.py`)

Centralized market data cache for a single pipeline run.

| Method | Returns | Used By |
|--------|---------|---------|
| `get_prices()` | Daily prices (all tickers) | Backtest |
| `get_monthly_returns()` | Month-end returns | Optimizer |
| `get_momentum_features(ticker)` | 1/3/6m momentum | Regime forecast |
| `set_prices(prices)` | — | Testing / injection |

On first `get_prices()` call, fetches via `fetch_prices()`. Subsequent calls return cached data.

### 2. FRED Cache (`src/data/fred_ingestion.py`)

In-memory cache keyed by `(series_id, end_date)`. Each FRED series is fetched once per run; repeated requests (e.g. same process, multiple runs) hit cache.

- `_get_fred_series_cached()` — internal helper used by `fetch_fred_core` and `fetch_fred_optional`
- `get_fred_cache_stats()` — returns `(hits, misses)`
- `clear_fred_cache()` — reset cache (testing)

### 3. Pipeline Integration (`src/pipeline.py`)

1. Run regime classification (FRED only).
2. Create `PipelineData`, call `get_prices()` once.
3. Pass `pipeline_data` to regime forecast, optimizer, backtest.
4. Each step uses the appropriate getter; no re-fetch.

## Backward Compatibility

All downstream functions accept `pipeline_data=None`:

- `regime_forecast.main(pipeline_data=None)` — if None, calls `build_momentum_features()` (fetches)
- `run_optimizer(pipeline_data=None)` — if None, calls `fetch_monthly_returns()` (fetches)
- `run_backtest(pipeline_data=None)` — if None, calls `fetch_prices()` (fetches)

Standalone execution (e.g. `python -m src.allocation.optimizer`) continues to work without changes.

## Logging

| Log | Meaning |
|-----|---------|
| `[DATA] Fetched prices: N tickers, M rows in X.XXs` | Initial yfinance fetch |
| `[DATA] Reusing cached prices` | Cache hit (DEBUG) |
| `[DATA] Regime forecast using shared pipeline_data` | Step reusing cache |
| `[FRED] Cache HIT/MISS: series_id` | FRED cache (DEBUG) |
| `[FRED] Cache stats: N hits, M misses` | After regime classification |
| `SUMMARY: Market data fetched once in X.XXs (3 steps reused cache)` | Pipeline summary |

## Validation

Run `scripts/validate_cache_parity.py` to confirm:

- Same tickers and date index
- Same monthly returns (manual vs PipelineData)
- Same momentum features
- Regime labels, allocations, backtest results present

## Files Changed

| File | Change |
|------|--------|
| `src/data/pipeline_data.py` | **New** — PipelineData class |
| `src/data/fred_ingestion.py` | FRED cache, `_get_fred_series_cached`, stats |
| `src/data/__init__.py` | Export PipelineData |
| `src/models/regime_forecast.py` | `main(pipeline_data=None)` |
| `src/allocation/optimizer.py` | `run_optimizer(pipeline_data=None)` |
| `src/backtest/engine.py` | `run_backtest(pipeline_data=None)` |
| `src/pipeline.py` | Fetch once, pass pipeline_data to steps |
| `scripts/validate_cache_parity.py` | **New** — validation script |
| `docs/P0_CACHE_ARCHITECTURE.md` | **New** — this doc |

## Assumptions & Risks

1. **Single process** — Cache is in-memory. Each pipeline run in a new process starts with an empty cache.
2. **Same date range** — All steps use `START_DATE` and `get_end_date()`. No step overrides.
3. **yfinance consistency** — Same request can return slightly different data (e.g. timing). Validation uses a single fetch for both paths.
4. **FRED cache** — First run: all misses. Same process calling FRED again (e.g. manual re-run): hits. Cross-process: always misses.
