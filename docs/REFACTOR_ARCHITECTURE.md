# Quantitative Research Architecture Refactor

## Current State Analysis

### Existing Modules
| File | Responsibilities | Issues |
|------|------------------|--------|
| `economic_regime.py` | FRED fetch, resample, z-scores, macro score, regime classification, save | Data + features + model mixed; print() instead of logging |
| `regime_forecast.py` | Load CSV, market fetch, seasonality, ML model, save | Duplicates yfinance logic; no shared feature utils |
| `optimizer.py` | Load regimes, fetch prices, Sortino optimization, save | Duplicates TICKERS, yfinance; subprocess for format_allocations |
| `backtest_polars.py` | Fetch prices, load DB, vol scaling, backtest loop, metrics | Duplicates TICKERS; module-level execution; no error handling |
| `backtest.py` | Legacy pandas backtest | Duplicate of backtest_polars; different ticker list |
| `database.py` | SQLite CRUD | Good; move to utils |
| `format_allocations.py` | Excel export | Good; move to allocation |

### Duplicated Code
1. **TICKERS, START_DATE, END_DATE** – 4 files
2. **yfinance price extraction** (MultiIndex handling) – 4 files
3. **ROOT_DIR, OUTPUTS_DIR** – 5 files
4. **Vol scaling logic** – backtest.py (pandas) vs backtest_polars.py (polars)
5. **to_month_end, rolling_z_score, sigmoid** – in economic_regime only but reusable

---

## Target Architecture

```
src/
├── __init__.py
├── config.py              # TICKERS, paths, constants (single source of truth)
├── data/
│   ├── __init__.py
│   ├── fred_ingestion.py  # FRED API fetch (core + optional series)
│   └── market_ingestion.py# yfinance fetch (prices, returns)
├── features/
│   ├── __init__.py
│   ├── transforms.py      # to_month_end, rolling_z_score, sigmoid
│   ├── macro_features.py  # build_dataframe, add_z_scores, macro_score
│   └── market_features.py # momentum, seasonality
├── models/
│   ├── __init__.py
│   ├── regime_classifier.py   # Rule-based GDP/inflation classification
│   └── regime_forecast.py     # ML GradientBoosting predictor
├── allocation/
│   ├── __init__.py
│   ├── optimizer.py       # Sortino optimization per regime
│   ├── vol_scaling.py     # Inverse-vol weight scaling (Polars)
│   └── format_allocations.py # Excel export
├── backtest/
│   ├── __init__.py
│   ├── engine.py          # Backtest loop, transaction costs
│   └── metrics.py         # compute_metrics (CAGR, Sharpe, drawdown)
└── utils/
    ├── __init__.py
    ├── database.py        # SQLite (unchanged, add logging)
    ├── logging_config.py  # Centralized logging setup
    └── retry.py           # Exponential backoff for file writes
```

---

## Structural Changes (Detailed)

### 1. `src/config.py` (NEW)
- **TICKERS**, **START_DATE**, **END_DATE**, **VOL_LOOKBACK**, **COST_BPS**, **REGIME_CASH**, **REGIME_MIN_ASSETS**
- **get_outputs_dir()**, **get_project_root()**
- Single source of truth; all modules import from here

### 2. `src/data/fred_ingestion.py` (EXTRACT from economic_regime)
- `FredIngestion` class or `fetch_fred_core()`, `fetch_fred_optional()`
- Returns raw Series; no feature engineering
- Logging instead of print; error handling for API failures

### 3. `src/data/market_ingestion.py` (NEW – consolidate)
- `fetch_prices(tickers, start, end) -> pd.DataFrame` – handles MultiIndex
- `fetch_monthly_returns(tickers, start, end) -> pd.DataFrame`
- Used by optimizer, backtest, regime_forecast

### 4. `src/features/transforms.py` (EXTRACT from economic_regime)
- `to_month_end(series) -> pd.Series`
- `rolling_z_score(series, window, min_periods) -> pd.Series`
- `sigmoid(series) -> pd.Series`
- Pure functions; fully typed; testable

### 5. `src/features/macro_features.py` (EXTRACT from economic_regime)
- `resample_to_monthly(...)` 
- `build_macro_dataframe(...)` – raw + mom + z-scores
- `calculate_macro_score(df) -> pd.DataFrame`
- Uses transforms; no I/O

### 6. `src/features/market_features.py` (EXTRACT from regime_forecast)
- `build_momentum_features(prices, ticker="SPY") -> pd.DataFrame`
- `build_seasonality_features(regime_df) -> pd.Series`
- Reusable for other models

### 7. `src/models/regime_classifier.py` (EXTRACT from economic_regime)
- `classify_regime(row) -> str` (static)
- `classify_regimes(df) -> pd.DataFrame`
- `RegimeClassifier.run()` – orchestrates data → features → classify → save
- Depends on data, features, utils

### 8. `src/models/regime_forecast.py` (REFACTOR)
- Uses `features/market_features`, `data/market_ingestion`
- `build_feature_matrix()`, `evaluate_forecast()`, `predict_next_month()`
- No data fetching of regime CSV – receives DataFrame from pipeline

### 9. `src/allocation/optimizer.py` (REFACTOR)
- Uses `data/market_ingestion`, `utils/database`, `config`
- `negative_sortino()`, `get_constraints()` – keep logic
- Call `allocation.format_allocations` directly (no subprocess)

### 10. `src/allocation/vol_scaling.py` (EXTRACT from backtest_polars)
- `vol_scaled_weights(raw_w, trailing_rets, risky_assets, ...) -> dict`
- Polars implementation; can add pandas fallback

### 11. `src/backtest/engine.py` (REFACTOR from backtest_polars)
- `run_backtest(regime_df, allocations, prices, config) -> tuple[pd.Series, dict]`
- Returns (portfolio_returns, metrics); no module-level execution
- Uses allocation.vol_scaling, backtest.metrics

### 12. `src/backtest/metrics.py` (EXTRACT)
- `compute_metrics(rets, rf_daily) -> dict[str, float]`
- Pure function; testable

### 13. `src/utils/database.py` (MOVE, add logging)
- Add logging on save/load
- Context manager usage encouraged

### 14. `src/utils/logging_config.py` (NEW)
- `setup_logging(level, log_file?)` – used by pipeline and modules

### 15. `src/utils/retry.py` (EXTRACT from economic_regime)
- `retry_on_permission_error(func, max_attempts=4)` – for CSV writes

### 16. Pipeline Entry Points
- `run_daily_update.py` – calls pipeline **in-process** (no subprocess)
- `src/pipeline.py` (NEW) – `run_daily_pipeline() -> int` – orchestration with logging/error handling

---

## Migration Strategy

1. Create new structure; add `__init__.py` with re-exports
2. Implement `config.py`, `utils/` first
3. Extract `data/`, `features/` (no behavior change)
4. Extract `models/`, `allocation/`, `backtest/`
5. Create `pipeline.py`; refactor `run_daily_update.py` to use it
6. Update tests to import from new paths
7. Remove `economic_regime.py`, `optimizer.py`, etc. (or keep thin wrappers for `python -m src.models.regime_classifier`)
8. Delete `backtest.py` (legacy)

---

## Backward Compatibility

- Add `src/economic_regime.py` as thin wrapper: `from src.models.regime_classifier import main; main()`
- Same for `optimizer`, `backtest_polars`, `regime_forecast`
- Ensures `python -m src.economic_regime` still works for Task Scheduler

---

## Test Updates

- `tests/test_economic_regime.py` → `tests/test_regime_classifier.py` (import from models)
- `tests/test_features.py` (NEW) – transforms, macro_features
- `tests/test_data.py` (NEW) – market_ingestion with mocked yfinance
