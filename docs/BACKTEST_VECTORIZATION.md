# Backtest Vectorization

## Re-Profile Results (After P0 + P1)

| Step | Time |
|------|------|
| data_fetch | 0.93s |
| regime_classification | 7.62s |
| regime_forecast | 1.67s |
| optimizer | 0.53s |
| backtest | **0.19s** (was 1.83s) |
| **Total** | **11.0s** (was ~14.5s) |

**Bottleneck:** Regime classification (7.62s, ~69%) is the largest, but it is **I/O-bound** (FRED API calls). The backtest was the main **compute bottleneck** and is now optimized.

---

## How the Original Backtest Loop Worked

1. **Setup:** Load prices → convert to Polars → compute returns → convert back to pandas.

2. **Per-date loop (~3,177 iterations):**
   - `regime_df.loc[date, "regime"]` — pandas loc
   - If month changed (rebalance, ~170 times):
     - `returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)` — O(n) pandas slice per rebalance
     - `pl.from_pandas(trailing_pd)` — pandas → Polars conversion
     - `vol_scaled_weights(..., trailing_pl, ...)` — Polars std
   - `daily_ret = sum(returns.loc[date, a] * weight for a in ASSETS)` — dict iteration
   - If rebalanced: turnover cost, update prev_weights

3. **Slowdown sources:**
   - ~3k loop iterations
   - ~170 rebalances × (pandas slice + Polars conversion + vol_scaled_weights)
   - Every iteration: loc lookups

---

## What Caused the Slowdown (Plain English)

- **Per-date loop:** ~3,177 days, each with a Python loop and loc access.
- **Repeated conversions:** ~170 times per run: `pandas → Polars` for each rebalance.
- **Repeated volatility:** ~170 times: `returns.loc[:date].tail(63)` and `vol_scaled_weights` instead of one rolling std.
- **Non-vectorized returns:** Daily returns computed one day at a time with `sum()` over a dict.

---

## Vectorized Implementation

1. **Precompute rolling volatility:** `returns[TICKERS].rolling(VOL_LOOKBACK).std()` once.
2. **No Polars in loop:** Use `vol_scaled_weights_from_std` with precomputed std.
3. **Precompute month changes:** `month_changed` for each date.
4. **Build weight matrix:** dates × assets, one loop over dates (no per-date Polars).
5. **Vectorized returns:** `(returns * weights).sum(axis=1)` instead of per-date sum.

---

## Files Changed

| File | Change |
|------|--------|
| `src/backtest/engine.py` | `_compute_returns_and_setup`, `_run_backtest_loop`, `_run_backtest_vectorized`, `run_backtest` uses vectorized |
| `src/allocation/vol_scaling.py` | `vol_scaled_weights_from_std` (pandas-based, no Polars) |
| `src/pipeline.py` | Per-step timing |
| `scripts/validate_backtest_parity.py` | **New** — parity validation |
| `docs/BACKTEST_VECTORIZATION.md` | **New** — this doc |

---

## Old vs New Timing

| Implementation | Time | Speedup |
|----------------|------|---------|
| Loop (per-date) | 1.41s | — |
| Vectorized | 0.13s | **11.2x** |

---

## Parity Results

```
OK: Portfolio returns match
OK: Cumulative returns match
OK: Drawdown match
OK: All metrics match
PASS: Vectorized backtest matches loop implementation.
```

---

## Assumptions & Risks

1. **Returns computation:** Uses pandas `pct_change()` instead of Polars; results match within tolerance.
2. **Rolling std:** `min_periods=1` for first rows; early dates may differ slightly from original.
3. **`_run_backtest_loop`:** Kept for parity checks; can be removed after validation.

---

## Remaining Bottlenecks

- **Regime classification (7.62s):** I/O-bound (FRED). Options: caching, parallel optional series.
- **Regime forecast (1.67s):** ML training. Options: fewer CV splits, smaller models.
- **Data fetch (0.93s):** yfinance. Already cached.
