# Fast Mode Guide for Experiment Iteration

## Overview

Fast mode reduces experiment runtime by **7.4x per configuration** (from ~100s to ~13s) by:
1. Using recent data only (last 8 years)
2. Limiting walk-forward segments (e.g., 20 most recent)
3. Skipping CSV and SQLite persistence
4. Caching intermediate computations

**Typical experiment suite speedup (4 configurations):**
- Full mode: **6.6 minutes**
- Fast mode: **0.9 minutes**
- Time saved: **5.7 minutes per experiment**

## When to Use Fast Mode

### Use Fast Mode For:
- Initial feature development and testing
- Rapid iteration on signal variants
- Quick validation of code changes
- Debugging experiment logic
- Parameter sensitivity checks

### Use Full Mode For:
- Final validation before accepting results
- Publishing results
- Cross-regime validation (needs full history)
- SQLite record keeping
- Comparing against historical benchmarks

## CLI Flags

### Fast Mode Flags

```bash
--fast-mode              # Enable all fast features (recent data, skip persist)
--start-date YYYY-MM-DD  # Override start date (fast mode defaults to recent 8 years)
--end-date YYYY-MM-DD    # Override end date
--max-segments N         # Limit to N most recent segments (e.g., 20)
--no-persist             # Skip CSV and SQLite writes
--use-cache              # Use cached allocations (experimental)
--show-timing            # Display detailed timing breakdown
```

### Fast Mode Behavior

When `--fast-mode` is enabled:
- Start date defaults to 8 years ago (unless `--start-date` is provided)
- Segments are limited to 20 most recent (unless `--max-segments` is provided)
- Persistence is skipped (CSV + SQLite)
- Caching is enabled

## Usage Examples

### Quick Test (single run, ~13 seconds)

```bash
python scripts/run_walk_forward.py \
    --fast-mode \
    --hybrid-signal \
    --hybrid-macro-weight 0.0 \
    --market-lookback-months 24 \
    --use-momentum \
    --portfolio-construction equal_weight \
    --show-timing
```

### Fast Experiment Suite (multiple configs, ~1 minute)

See `scripts/run_asset_momentum_experiment_fast.py` for an example of running multiple configurations in fast mode:

```bash
python scripts/run_asset_momentum_experiment_fast.py
```

This runs 4 configurations in ~1.3 minutes (vs 6.6 minutes in full mode).

### Custom Fast Mode (granular control)

```bash
python scripts/run_walk_forward.py \
    --start-date 2015-01-01 \
    --max-segments 30 \
    --no-persist \
    --use-cache \
    --show-timing \
    --hybrid-signal \
    --hybrid-macro-weight 0.0 \
    --market-lookback-months 24 \
    --use-momentum \
    --portfolio-construction equal_weight
```

### Full Validation Mode (~100 seconds)

```bash
python scripts/run_walk_forward.py \
    --hybrid-signal \
    --hybrid-macro-weight 0.0 \
    --market-lookback-months 24 \
    --use-momentum \
    --portfolio-construction equal_weight \
    --show-timing
```

## Runtime Breakdown

Based on profiling with 24M momentum + equal_weight:

| Component | Time (Full) | % of Total | Notes |
|-----------|-------------|------------|-------|
| Segment backtest | 72.2s | 73.7% | Vectorized implementation (already optimized) |
| Segment optimization | 18.9s | 19.3% | Bypassed when using equal_weight |
| Data prep (per segment) | 2.5s | 2.5% | Monthly resampling and filtering |
| Price fetching | 2.4s | 2.4% | Yahoo Finance API |
| Metrics computation | 1.8s | 1.9% | CAGR, Sharpe, MaxDD, etc. |
| SQLite persist | 0.1s | 0.1% | Database writes |
| CSV writes | 0.03s | 0.0% | File I/O |
| **Total** | **98.0s** | **100%** | |

**Key Insights:**
- Backtest execution dominates runtime (74%)
- Reducing segments has linear impact on runtime
- Equal weight bypasses expensive optimization
- Persistence overhead is negligible (~0.1%)

## Trade-offs

### Fast Mode Advantages:
- **7.4x faster** per configuration
- **5-6x faster** for typical 4-config experiments
- Enables rapid iteration
- Same code paths (no logic changes)

### Fast Mode Limitations:
- **Recent data only** (may miss 2008 crisis, early 2010s)
- **Fewer segments** (may overfit to recent market conditions)
- **No SQLite record** (can't query results later)
- **No CSV archive** (results lost after script completes)

### Recommended Workflow:

1. **Develop** with `--fast-mode` (iterate quickly)
2. **Validate** with full mode (confirm robustness)
3. **Accept** only full-mode results (avoid recency bias)

## Implementation Notes

### Timing Infrastructure

New utilities in `src/utils/timing.py`:
- `Timer`: Context manager for timing code blocks
- `TimingReport`: Collect and aggregate timing statistics

### Caching Infrastructure

New utilities in `src/utils/cache.py`:
- File-based cache using pickle
- Caches expensive allocations by (train_start, train_end, portfolio_method)
- Cache directory: `outputs/.cache/`
- Clear cache: `rm -rf outputs/.cache/`

### Modified Files

**Core Framework:**
- `src/evaluation/walk_forward.py`: Added timing, caching, fast-mode parameters
- `scripts/run_walk_forward.py`: Added CLI flags for fast mode

**New Files:**
- `src/utils/timing.py`: Timing utilities
- `src/utils/cache.py`: Caching utilities
- `scripts/profile_experiment_runtime.py`: Profiling script
- `scripts/run_asset_momentum_experiment_fast.py`: Fast experiment example

## Performance Benchmarks

Measured on current baseline (24M momentum, equal_weight):

| Mode | Runtime | Segments | Speedup |
|------|---------|----------|---------|
| Full | 98.4s | 116 | 1.0x (baseline) |
| Fast | 13.3s | 20 | **7.4x faster** |

**4-config experiment suite:**
- Full: 6.6 minutes
- Fast: 0.9 minutes (measured: 1.3 minutes with overhead)
- Speedup: ~5-7x

## Next Steps

When developing new experiments:

1. Create a fast-mode version of your experiment script
2. Use `--fast-mode` for initial development
3. Run full mode for final validation
4. Report both fast and full results when presenting findings

**Example Pattern:**

```python
# scripts/run_my_experiment_fast.py
def _run_experiment_fast(config):
    return run_walk_forward_evaluation(
        ...,
        fast_mode=True,
        max_segments=20,
        skip_persist=True,
        use_cache=True,
        show_timing=False,
    )
```

This pattern is demonstrated in `run_asset_momentum_experiment_fast.py`.
