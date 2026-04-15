"""Test asset-specific momentum vs market-level momentum (FAST MODE)."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment_fast(portfolio_method: str) -> pd.DataFrame:
    """Run 24M momentum experiment with specified asset selection method (FAST)."""
    logger.info("=" * 80)
    logger.info("RUNNING: %s", portfolio_method)
    logger.info("=" * 80)
    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.0,  # Pure market
        market_lookback_months=24,  # 24M momentum
        use_momentum=True,  # Momentum, not mean-reversion
        trend_filter_type="none",
        vol_scaling_method="none",
        portfolio_construction_method=portfolio_method,
        fast_mode=True,  # Enable fast mode
        max_segments=20,  # Limit segments
        skip_persist=True,  # Skip persistence
        use_cache=True,  # Use caching
        show_timing=False,  # Suppress detailed timing per run
    )
    return df


def main():
    """Run asset-specific momentum experiment (FAST MODE)."""
    logger.info("Starting asset-specific momentum experiment (FAST MODE)")
    logger.info("Recent 8 years, max 20 segments, no persistence, with caching")

    overall_start = time.perf_counter()

    # Test configurations
    methods = [
        {"method": "equal_weight", "name": "Baseline (all risk-on assets)"},
        {"method": "asset_momentum_positive", "name": "Positive momentum only"},
        {"method": "asset_momentum_top3", "name": "Top 3 by momentum"},
        {"method": "asset_momentum_top5", "name": "Top 5 by momentum"},
    ]

    results = []

    # Run experiments
    for config in methods:
        df = _run_experiment_fast(config["method"])

        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue

        overall = df[df["segment"] == "OVERALL"].iloc[0]

        result = {
            "method_name": config["name"],
            "method": config["method"],
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
        }
        results.append(result)

    overall_elapsed = (time.perf_counter() - overall_start) / 60

    if not results:
        logger.error("All experiments failed.")
        sys.exit(1)

    results_df = pd.DataFrame(results)

    logger.info("")
    logger.info("=" * 80)
    logger.info("FAST MODE EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {overall_elapsed:.1f} minutes")
    logger.info(f"Configurations tested: {len(methods)}")
    logger.info(f"Time per config: {overall_elapsed / len(methods):.1f} minutes")
    logger.info("")

    # Summary table
    logger.info("RESULTS SUMMARY:")
    logger.info("")
    logger.info("| Method | CAGR | Sharpe | MaxDD | Turnover |")
    logger.info("|--------|------|--------|-------|----------|")

    for _, row in results_df.iterrows():
        logger.info(
            f"| {row['method_name']:>30} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )

    logger.info("")

    # Best performers
    baseline = results_df[results_df["method"] == "equal_weight"].iloc[0]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]

    logger.info("QUICK INSIGHTS:")
    logger.info(f"- Baseline: {baseline['sharpe']:.3f} Sharpe")
    logger.info(
        f"- Best: {best_sharpe['method_name']} ({best_sharpe['sharpe']:.3f} Sharpe, {best_sharpe['sharpe'] - baseline['sharpe']:+.3f} vs baseline)"
    )
    logger.info("")
    logger.info(
        "Note: Fast mode uses recent 8 years only. Run full experiment for final validation."
    )


if __name__ == "__main__":
    main()
