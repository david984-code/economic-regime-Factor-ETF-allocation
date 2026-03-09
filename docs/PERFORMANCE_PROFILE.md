# Performance Profile & Optimization Recommendations

This document profiles the codebase for performance bottlenecks and recommends optimizations with estimated impact.

---

## 1. Repeated API Calls (High Impact)

### 1.1 yfinance `fetch_prices` — Called 3× per pipeline run

| Caller | Tickers | Purpose |
|--------|---------|---------|
| `regime_forecast.main()` → `build_momentum_features()` | `[SPY]` | SPY 1/3/6m momentum |
| `optimizer.run_optimizer()` → `fetch_monthly_returns()` | All TICKERS | Monthly returns |
| `backtest.run_backtest()` | All TICKERS | Daily prices |

**Problem:** Each step fetches overlapping date ranges (START_DATE → today) independently. Same data is downloaded 3 times.

**Estimate:** ~3–8 seconds per yfinance call × 3 ≈ **9–24 seconds** wasted per run.

**Recommendation:**
- Add a **data cache layer** (e.g., `@functools.lru_cache` with date-based invalidation, or SQLite/Parquet cache keyed by `(tickers, start, end)`).
- Or **pipeline-level data sharing**: fetch prices once at pipeline start, pass `prices` into regime_forecast, optimizer, and backtest.

**Estimated improvement:** **60–80%** reduction in market data fetch time (save ~6–20 seconds per run).

---

### 1.2 FRED API — 10 sequential calls per run

**Core (6):** GDP, CPI, DGS10, DGS3MO, M2SL, M2V  
**Optional (4):** NAPM/INDPRO, INDPRO, ICSA, BAMLH0A0HYM2

**Problem:** All calls are sequential. FRED rate limits are generous, but network latency adds up.

**Estimate:** ~0.5–1.5 s per call × 10 ≈ **5–15 seconds** total.

**Recommendation:**
- **Cache FRED results** by date (e.g., cache key `(series_id, end_date)`). Data changes infrequently (GDP quarterly, CPI monthly).
- **Parallel fetch** for optional series using `concurrent.futures.ThreadPoolExecutor` (FRED is I/O bound).

**Estimated improvement:** Cache hit → **~90%** reduction (1–2 s vs 5–15 s). Parallel optional → **~30%** on cache miss.

---

## 2. Slow Pandas Operations

### 2.1 `df.apply(classify_regime, axis=1)` — Row-wise Python loop

**Location:** `src/models/regime_classifier.py` (lines 82, 118)

**Problem:** `apply(axis=1)` invokes a Python function per row. For ~200 monthly rows, this is ~200 Python call overheads plus non-vectorized logic.

**Current logic:**
```python
def classify_regime(row):
    gdp_z, infl_z = row["gdp_z"], row["infl_z"]
    if gdp_z > 0 and infl_z <= 0: return "Recovery"
    if gdp_z > 0 and infl_z > 0: return "Overheating"
    ...
```

**Recommendation:** **Vectorize with `np.select` or `pd.cut`**:

```python
gdp_z = df["gdp_z"].values
infl_z = df["infl_z"].values
conditions = [
    (gdp_z > 0) & (infl_z <= 0),
    (gdp_z > 0) & (infl_z > 0),
    (gdp_z <= 0) & (infl_z > 0),
]
choices = ["Recovery", "Overheating", "Stagflation"]
df["regime"] = np.select(conditions, choices, default="Contraction")
# Handle NaN separately
df.loc[pd.isna(gdp_z) | pd.isna(infl_z), "regime"] = "Unknown"
```

**Estimated improvement:** **5–20×** faster for this step (milliseconds vs tens of ms). Small absolute gain but cleaner and more scalable.

---

### 2.2 `for date in returns.index` — Daily backtest loop

**Location:** `src/backtest/engine.py` (line 113)

**Problem:** Loops over ~3,000+ trading days. Per iteration:
- `regime_df.loc[date, "regime"]`
- `returns.loc[:date].tail(VOL_LOOKBACK)` — O(n) slice each time
- `pl.from_pandas(trailing_pd)` — pandas → polars conversion on every rebalance
- `vol_scaled_weights(...)` — polars std computation
- `sum(returns.loc[date, a] * weight for a in ASSETS)` — dict iteration

**Estimate:** ~3,000 iterations × (loc + possible polars conversion) ≈ **1–3 seconds** in the loop.

**Recommendation:**
- **Precompute rebalance dates** (month boundaries) and only compute vol scaling at those points.
- **Vectorize portfolio returns**: build a matrix of weights (dates × assets) and use `(returns * weights).sum(axis=1)` instead of a Python loop.
- **Keep returns in Polars** end-to-end; avoid repeated `to_pandas()` / `from_pandas()` at each date.
- **Precompute trailing volatility** per asset per date (or per month) in one pass.

**Estimated improvement:** **3–10×** faster backtest loop (from ~1–3 s to ~0.1–0.5 s).

---

### 2.3 `merged[merged["regime"] == regime]` — Repeated boolean mask

**Location:** `src/allocation/optimizer.py` (line 114)

**Problem:** For each of 5 regimes, creates a full boolean mask over the merged DataFrame. O(n × 5) scans.

**Recommendation:** Use `groupby`:

```python
for regime, subset in merged.groupby("regime"):
    subset_risky = subset[risky_assets].fillna(0)
    ...
```

**Estimated improvement:** **~20–40%** faster optimizer (avoids repeated full scans). Modest but cleaner.

---

## 3. Regime Forecast — Row-by-row feature building

**Location:** `src/models/regime_forecast.py` (lines 51–76)

**Problem:**
```python
for i in range(len(regime_df) - 1):
    row = regime_df.iloc[i]
    next_row = regime_df.iloc[i + 1]
    ...
    if period_ts in market_df.index:
        mkt = market_df.loc[period_ts]
    ...
```

- `iloc[i]` and `iloc[i+1]` — repeated indexing.
- `market_df.loc[period_ts]` — repeated lookup.
- Building `X_list`, `y_list` in Python loops.

**Recommendation:** **Vectorize**:
- Use `shift(-1)` for next-row values.
- Use `merge` or `reindex` to align `market_df` with regime dates in one shot.
- Build `X` as a DataFrame from vectorized columns instead of list-of-lists.

**Estimated improvement:** **5–15×** faster feature matrix construction (from ~50–100 ms to ~5–10 ms).

---

## 4. Inefficient Joins / Reindexing

### 4.1 `regime_df.reindex(prices.index, method="ffill")`

**Location:** `src/backtest/engine.py` (line 74)

**Problem:** Expands monthly regime labels to daily index. For 3k+ days, creates a large DataFrame. `method="ffill"` is efficient, but the result is stored for the whole backtest.

**Recommendation:** Keep regime as a monthly Series and use `date.to_period("M")` for lookup. Or use `merge_asof` for efficient alignment. Current approach is acceptable; minor gains possible.

---

### 4.2 Optimizer merge

**Location:** `src/allocation/optimizer.py` (line 107)

```python
merged = pd.merge(returns, regimes, on="Period", how="inner")
```

**Assessment:** Single merge, reasonable. No major issue.

---

## 5. Polars / Pandas Conversion Overhead

**Locations:**
- `backtest/engine.py`: `prices_pd` → `prices_pl` → `returns_pl` → `returns` (pandas)
- `vol_scaling.py`: receives `trailing_rets_pl`, uses Polars for std

**Problem:** Backtest converts to Polars for return computation, then back to pandas for the date loop. Each rebalance converts a 63-row slice to Polars again.

**Recommendation:**
- **Stay in Polars** for the full backtest: compute returns, build weight matrix, vectorize portfolio returns.
- Or **stay in pandas** and use `returns.rolling(VOL_LOOKBACK).std()` for trailing vol — avoid Polars conversion in the hot loop.

**Estimated improvement:** **~30–50%** faster backtest by reducing conversion overhead.

---

## 6. Memory Usage

**Observations:**
- Multiple copies of price data (regime_forecast, optimizer, backtest each load).
- `regime_df.reindex(prices.index)` duplicates regime labels to daily frequency.

**Recommendation:** Shared data cache reduces memory (one copy of prices). Reindex is acceptable for ~3k rows.

---

## 7. Summary: Priority & Estimated Impact

| Priority | Optimization | Est. time saved | Effort |
|----------|--------------|-----------------|--------|
| **P0** | Cache / share `fetch_prices` across pipeline | 6–20 s | Medium |
| **P0** | Cache FRED API results by date | 4–13 s (cache hit) | Low |
| **P1** | Vectorize backtest loop (precompute weights, vectorize returns) | 1–2.5 s | High |
| **P1** | Vectorize `classify_regime` (np.select) | &lt;0.1 s | Low |
| **P2** | Vectorize regime forecast `build_feature_matrix` | &lt;0.1 s | Medium |
| **P2** | Optimizer `groupby` instead of repeated mask | &lt;0.1 s | Low |
| **P2** | Reduce Polars ↔ pandas conversions in backtest | 0.3–0.5 s | Medium |
| **P3** | Parallel FRED optional fetch | 1–3 s | Low |

**Total pipeline runtime (observed):** ~25 s  
**Estimated after P0+P1:** ~10–15 s (**~40–50%** faster)  
**Estimated after all:** ~8–12 s (**~50–65%** faster)

---

## 8. Conversion to Polars — Where It Helps

| Component | Current | Polars benefit |
|-----------|---------|----------------|
| Returns computation | pandas `pct_change` | Polars native — similar speed |
| Rolling z-score | pandas `rolling` | Polars `rolling_std` — **2–5×** faster on large data |
| Backtest loop | pandas row iteration | Polars expression-based — **5–10×** if fully vectorized |
| Regime merge | pandas `merge` | Polars `join` — similar |
| Feature matrix | pandas + loops | Polars `select` + `shift` — **3–5×** if vectorized |

**Recommendation:** Polars is most beneficial for:
1. **Backtest engine** — vectorize the entire returns × weights computation.
2. **Macro features** — `add_z_scores` rolling computations (if data grows).
3. **Regime forecast** — building feature matrix with `shift`, `join`.

Pandas is fine for optimizer (small data) and regime classifier (monthly, ~200 rows).
