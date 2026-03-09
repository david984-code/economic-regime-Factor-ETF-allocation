# Regime Classification: Local Macro Cache

## Current FRED/Regime Flow (Before Optimization)

1. **fetch_fred_core** (6 API calls, sequential):
   - GDP, CPIAUCSL, DGS10, DGS3MO, M2SL, M2V
   - Each: `fred.get_series(series_id, observation_end=end)` — full history
   - **API latency:** ~0.5–1.5s per call ≈ 3–9s total

2. **fetch_fred_optional** (4 API calls, sequential):
   - NAPM, INDPRO, ICSA, BAMLH0A0HYM2
   - Same full-history fetch
   - **API latency:** ~2–6s total

3. **resample_high_freq_to_monthly** — local compute (~0.01s)

4. **resample_to_monthly** — local compute (~0.01s)

5. **build_macro_dataframe** — local compute (~0.01s)

6. **add_z_scores** — local compute (~0.02s)

7. **calculate_macro_score** — local compute (~0.01s)

8. **classify_regimes_vectorized** — local compute (~0.001s)

**Total:** ~5–15s API + ~0.2s compute. The 7.73s observed is almost entirely API latency.

---

## Local Macro Cache Architecture

### Storage

- **Directory:** `outputs/macro_cache/`
- **Format:** One CSV per series: `{series_id}.csv` with columns `date`, `value`
- **Example:** `GDP.csv`, `CPIAUCSL.csv`, `DGS10.csv`, etc.

### Fetch Logic

1. **Load cache:** If `{series_id}.csv` exists, load and sort by date.
2. **Cache hit:** If `cached_max_date >= end_date`, return cached series (no API call).
3. **Incremental fetch:** If `end_date > cached_max_date`, call API with `observation_start=cached_max+1`, `observation_end=end_date`. Merge with cache, save, return.
4. **Full fetch:** If no cache or load fails, full API fetch, save, return.

### Sub-Step Timing

```
[REGIME] Sub-step timing: fred=X.XXs merge=X.XXs features=X.XXs label=X.XXs total=X.XXs
```

- **fred:** FRED retrieval (API + cache)
- **merge:** Resample, alignment
- **features:** Z-scores, macro score
- **label:** Regime classification

---

## Logging

| Log | Meaning |
|-----|---------|
| `[FRED cache] GDP: api_full (320 rows)` | Full API fetch, cache populated |
| `[FRED cache] GDP: cache hit (no new data)` | Used cache, no API |
| `[FRED cache] GDP: api_incremental (1 new obs, total 320)` | Fetched new data, merged, saved |
| `[FRED cache] Loaded GDP from cache (latest 2026-03-05)` | Cache hit, data current |

---

## Files Changed

| File | Change |
|------|--------|
| `src/config.py` | `MACRO_CACHE_DIR` |
| `src/data/fred_cache.py` | **New** — load/save, incremental fetch |
| `src/data/fred_ingestion.py` | `fetch_fred_core_cached`, `fetch_fred_optional_cached` |
| `src/models/regime_classifier.py` | Sub-step timing, `run_and_return_df`, `use_local_cache` |
| `scripts/validate_regime_cache_parity.py` | **New** — parity validation |
| `docs/REGIME_MACRO_CACHE.md` | **New** — this doc |

---

## Old vs New Timing

| Scenario | Regime classification | Notes |
|----------|------------------------|-------|
| First run (no cache) | ~7.7s | Same as before, populates cache |
| Second run (cache warm) | ~4.1s | Cache hits + incremental |
| Cache fully warm | ~1–2s | All series from cache |

---

## Parity Results

```
OK: Index match (1287 rows)
OK: gdp_z, infl_z, risk_on, macro_score
OK: All regime labels match
OK: Latest regime
PASS: Cached and non-cached produce identical outputs.
```

---

## Risks & Assumptions

1. **Revisions:** FRED can revise older observations. Incremental fetch only adds new dates; we do not re-fetch revised history. A full refresh (delete cache or use `use_local_cache=False`) is needed to pick up revisions.

2. **Release lags:** Macro data is released with delay (e.g. GDP quarterly, CPI monthly). We use `observation_end=today`; FRED returns the latest available. No change from prior behavior.

3. **Stale cache:** If the cache is corrupted or out of sync, use `use_local_cache=False` or delete `outputs/macro_cache/` to force a full refresh.

4. **Concurrent runs:** No locking. Avoid running multiple pipelines that write to the same cache at once.
