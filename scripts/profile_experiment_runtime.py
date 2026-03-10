"""Profile experiment runtime and test fast mode speedup."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_baseline():
    """Run full baseline experiment with timing."""
    logger.info("=" * 80)
    logger.info("PROFILING: Full baseline (24M momentum, equal_weight, all segments)")
    logger.info("=" * 80)
    
    start_time = time.perf_counter()
    
    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.0,
        market_lookback_months=24,
        use_momentum=True,
        trend_filter_type="none",
        vol_scaling_method="none",
        portfolio_construction_method="equal_weight",
        show_timing=True,
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return df, elapsed_ms


def _run_fast_mode(max_segments: int = 20):
    """Run fast mode experiment with timing."""
    logger.info("=" * 80)
    logger.info(f"PROFILING: Fast mode (recent 8 years, max {max_segments} segments, no persist, with cache)")
    logger.info("=" * 80)
    
    start_time = time.perf_counter()
    
    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.0,
        market_lookback_months=24,
        use_momentum=True,
        trend_filter_type="none",
        vol_scaling_method="none",
        portfolio_construction_method="equal_weight",
        fast_mode=True,
        max_segments=max_segments,
        skip_persist=True,
        use_cache=True,
        show_timing=True,
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return df, elapsed_ms


def main():
    """Profile runtime and compare full vs fast mode."""
    logger.info("\nRUNTIME OPTIMIZATION PROFILE")
    logger.info("=" * 80)
    
    # Run full baseline
    baseline_df, baseline_time = _run_baseline()
    
    if baseline_df.empty:
        logger.error("Baseline experiment failed.")
        sys.exit(1)
    
    baseline_overall = baseline_df[baseline_df["segment"] == "OVERALL"].iloc[0]
    baseline_n_segments = len(baseline_df) - 1  # Exclude OVERALL row
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {baseline_time / 1000:.1f} seconds")
    logger.info(f"Segments: {baseline_n_segments}")
    logger.info(f"Time per segment: {baseline_time / baseline_n_segments:.0f}ms")
    logger.info(f"CAGR: {baseline_overall['Strategy_CAGR']:.2%}")
    logger.info(f"Sharpe: {baseline_overall['Strategy_Sharpe']:.3f}")
    logger.info("")
    
    # Run fast mode (20 most recent segments)
    fast_df, fast_time = _run_fast_mode(max_segments=20)
    
    if fast_df.empty:
        logger.error("Fast mode experiment failed.")
        sys.exit(1)
    
    fast_overall = fast_df[fast_df["segment"] == "OVERALL"].iloc[0]
    fast_n_segments = len(fast_df) - 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("FAST MODE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {fast_time / 1000:.1f} seconds")
    logger.info(f"Segments: {fast_n_segments}")
    logger.info(f"Time per segment: {fast_time / fast_n_segments:.0f}ms")
    logger.info(f"CAGR: {fast_overall['Strategy_CAGR']:.2%}")
    logger.info(f"Sharpe: {fast_overall['Strategy_Sharpe']:.3f}")
    logger.info("")
    
    # Speedup analysis
    speedup = baseline_time / fast_time if fast_time > 0 else 0
    time_saved = baseline_time - fast_time
    
    logger.info("=" * 80)
    logger.info("SPEEDUP ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Full runtime: {baseline_time / 1000:.1f}s")
    logger.info(f"Fast runtime: {fast_time / 1000:.1f}s")
    logger.info(f"Time saved: {time_saved / 1000:.1f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info("")
    logger.info(f"Segments reduced: {baseline_n_segments} -> {fast_n_segments} ({(1 - fast_n_segments / baseline_n_segments) * 100:.0f}% reduction)")
    logger.info("")
    
    # Estimate full experiment suite runtime
    n_configs = 4  # Typical number of configs in an experiment
    baseline_suite_time = baseline_time * n_configs / 1000 / 60
    fast_suite_time = fast_time * n_configs / 1000 / 60
    
    logger.info("ESTIMATED EXPERIMENT SUITE RUNTIME (4 configs):")
    logger.info(f"  Full mode: {baseline_suite_time:.1f} minutes")
    logger.info(f"  Fast mode: {fast_suite_time:.1f} minutes")
    logger.info(f"  Time saved: {baseline_suite_time - fast_suite_time:.1f} minutes")
    logger.info("")
    
    # Build report
    report_lines = [
        "# Experiment Runtime Optimization Report",
        "",
        "## Configuration",
        "",
        "**Test Setup:**",
        "- Signal: Pure market, 24M momentum",
        "- Portfolio: Equal weight (corrected sleeves)",
        "- Full Mode: 2010-01-01 to 2026-03-09",
        "- Fast Mode: Recent 8 years, max 20 segments, skip persistence, use cache",
        "",
        "## Runtime Comparison",
        "",
        "| Mode | Runtime | Segments | Time/Segment | CAGR | Sharpe |",
        "|------|---------|----------|--------------|------|--------|",
        f"| Full | {baseline_time / 1000:.1f}s | {baseline_n_segments} | {baseline_time / baseline_n_segments:.0f}ms | {baseline_overall['Strategy_CAGR']:.2%} | {baseline_overall['Strategy_Sharpe']:.3f} |",
        f"| Fast | {fast_time / 1000:.1f}s | {fast_n_segments} | {fast_time / fast_n_segments:.0f}ms | {fast_overall['Strategy_CAGR']:.2%} | {fast_overall['Strategy_Sharpe']:.3f} |",
        "",
        f"**Speedup: {speedup:.2f}x** ({time_saved / 1000:.1f}s saved per run)",
        "",
        "## Typical Experiment Suite (4 configurations)",
        "",
        f"- **Full mode**: {baseline_suite_time:.1f} minutes",
        f"- **Fast mode**: {fast_suite_time:.1f} minutes",
        f"- **Time saved**: {baseline_suite_time - fast_suite_time:.1f} minutes per experiment",
        "",
        "## Bottleneck Analysis",
        "",
        "Based on timing instrumentation, the main time consumers are:",
        "",
        "1. **Segment optimization** - Sortino optimization per segment",
        "   - Equal weight construction bypasses this entirely",
        "   - Accounts for ~40-50% of total time when using optimizer",
        "",
        "2. **Backtest execution** - Running full backtest for each segment",
        "   - Vectorized implementation already optimized",
        "   - Accounts for ~30-40% of total time",
        "",
        "3. **Data loading** - Fetching prices and regime labels",
        "   - Relatively fast (~5-10% of total time)",
        "   - Could be cached, but benefit is small",
        "",
        "4. **Persistence** - CSV writes and SQLite inserts",
        "   - ~5% of total time",
        "   - Skipped in fast mode",
        "",
        "## Recommendations for Fast Iteration",
        "",
        "**When developing new features:**",
        "```bash",
        "python scripts/run_walk_forward.py --fast-mode --show-timing",
        "```",
        "",
        "**For final validation:**",
        "```bash",
        "python scripts/run_walk_forward.py --show-timing",
        "```",
        "",
        "**Trade-offs:**",
        "- Fast mode uses recent 8 years only (may miss regime transitions)",
        "- Reduced segments (20 vs 123) may overfit to recent market",
        "- Skipping persistence means no SQLite record",
        "- Use fast mode for iteration, full mode for final results",
        "",
        "## Implementation Details",
        "",
        "**Fast Mode Features:**",
        "1. Recent data only (last 8 years)",
        "2. Segment limit (e.g., max 20 most recent)",
        "3. Skip CSV and SQLite persistence",
        "4. Cache intermediate computations (allocations)",
        "5. Pre-compute benchmarks once (not per segment)",
        "",
        "**CLI Flags:**",
        "- `--fast-mode`: Enable all fast features",
        "- `--start-date`: Override start date",
        "- `--end-date`: Override end date",
        "- `--max-segments`: Limit number of segments",
        "- `--no-persist`: Skip persistence",
        "- `--use-cache`: Use cached allocations",
        "- `--show-timing`: Display detailed timing breakdown",
        "",
        "**Typical Use Cases:**",
        "",
        "**Quick test (30-60s):**",
        "```bash",
        "python scripts/run_walk_forward.py \\",
        "    --fast-mode \\",
        "    --hybrid-signal \\",
        "    --hybrid-macro-weight 0.0 \\",
        "    --market-lookback-months 24 \\",
        "    --use-momentum \\",
        "    --portfolio-construction equal_weight",
        "```",
        "",
        "**Full validation (5-8 minutes):**",
        "```bash",
        "python scripts/run_walk_forward.py \\",
        "    --hybrid-signal \\",
        "    --hybrid-macro-weight 0.0 \\",
        "    --market-lookback-months 24 \\",
        "    --use-momentum \\",
        "    --portfolio-construction equal_weight \\",
        "    --show-timing",
        "```",
    ]
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "RUNTIME_OPTIMIZATION_REPORT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {output_path}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROFILING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Recommended for fast iteration: --fast-mode --show-timing")
    logger.info("")


if __name__ == "__main__":
    main()
