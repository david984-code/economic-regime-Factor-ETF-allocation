"""Test volatility regime signal as orthogonal complement to momentum."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment(
    config_name: str,
    use_vol_regime: bool,
    vol_regime_weight: float,
    fast_mode: bool = True,
) -> pd.DataFrame:
    """Run 24M momentum experiment with optional volatility regime."""
    logger.info("=" * 80)
    logger.info("RUNNING: %s", config_name)
    logger.info("=" * 80)

    if use_vol_regime:
        logger.info(f"Volatility regime enabled: weight={vol_regime_weight:.2f}")
    else:
        logger.info("Volatility regime disabled (momentum only)")

    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.0,  # Pure market (no macro)
        market_lookback_months=24,  # 24M momentum
        use_momentum=True,  # Momentum
        trend_filter_type="none",
        vol_scaling_method="none",
        portfolio_construction_method="equal_weight",
        momentum_12m_weight=0.0,
        use_vol_regime=use_vol_regime,
        vol_regime_weight=vol_regime_weight,
        fast_mode=fast_mode,
        max_segments=20 if fast_mode else None,
        skip_persist=fast_mode,
        use_cache=fast_mode,
        show_timing=False,
    )
    return df


def _difficult_period_metrics(df: pd.DataFrame) -> dict:
    """Extract 2021-2022 period metrics."""
    test_segments = df[df["segment"] != "OVERALL"].copy()

    if test_segments.empty:
        return {}

    difficult_segments = []
    for _, row in test_segments.iterrows():
        test_start = pd.Period(row["test_start"], freq="M").to_timestamp()
        test_end = pd.Period(row["test_end"], freq="M").to_timestamp()

        if (test_start.year >= 2021 and test_start.year <= 2022) or (
            test_end.year >= 2021 and test_end.year <= 2022
        ):
            difficult_segments.append(row)

    if not difficult_segments:
        return {
            "n_difficult": 0,
            "difficult_cagr": np.nan,
            "difficult_sharpe": np.nan,
            "difficult_maxdd": np.nan,
        }

    difficult_df = pd.DataFrame(difficult_segments)

    return {
        "n_difficult": len(difficult_df),
        "difficult_cagr": difficult_df["Strategy_CAGR"].mean(),
        "difficult_sharpe": difficult_df["Strategy_Sharpe"].mean(),
        "difficult_maxdd": difficult_df["Strategy_MaxDD"].mean(),
    }


def main():
    """Run volatility regime experiment."""
    logger.info("Starting volatility regime experiment")
    logger.info("Fixed: 24M momentum + equal-weight sleeves")
    logger.info("Testing: adding volatility regime to risk_on calculation")
    logger.info("")

    # === PHASE 1: Fast mode screening ===
    logger.info("=" * 80)
    logger.info("PHASE 1: FAST MODE SCREENING")
    logger.info("=" * 80)
    logger.info("Recent 8 years, max 20 segments, no persistence")
    logger.info("")

    fast_start = time.perf_counter()

    # Test configurations
    configs = [
        {"name": "Baseline (momentum only)", "use_vol": False, "vol_weight": 0.0},
        {"name": "Momentum + Vol Regime (0.2)", "use_vol": True, "vol_weight": 0.2},
        {"name": "Momentum + Vol Regime (0.3)", "use_vol": True, "vol_weight": 0.3},
        {"name": "Momentum + Vol Regime (0.5)", "use_vol": True, "vol_weight": 0.5},
    ]

    fast_results = []

    for config in configs:
        df = _run_experiment(
            config_name=config["name"],
            use_vol_regime=config["use_vol"],
            vol_regime_weight=config["vol_weight"],
            fast_mode=True,
        )

        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue

        overall = df[df["segment"] == "OVERALL"].iloc[0]
        difficult = _difficult_period_metrics(df)

        result = {
            "config_name": config["name"],
            "use_vol_regime": config["use_vol"],
            "vol_weight": config["vol_weight"],
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
            "difficult_cagr": difficult.get("difficult_cagr", np.nan),
            "difficult_sharpe": difficult.get("difficult_sharpe", np.nan),
            "difficult_maxdd": difficult.get("difficult_maxdd", np.nan),
        }
        fast_results.append(result)

    fast_elapsed = (time.perf_counter() - fast_start) / 60

    if not fast_results:
        logger.error("All fast mode experiments failed.")
        sys.exit(1)

    fast_df = pd.DataFrame(fast_results)

    logger.info("")
    logger.info("=" * 80)
    logger.info("FAST MODE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Runtime: {fast_elapsed:.1f} minutes")
    logger.info("")
    logger.info("| Configuration | CAGR | Sharpe | MaxDD | Turnover |")
    logger.info("|---------------|------|--------|-------|----------|")

    for _, row in fast_df.iterrows():
        logger.info(
            f"| {row['config_name']:>35} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )

    logger.info("")

    # Check if vol regime is promising
    baseline_fast = fast_df[not fast_df["use_vol_regime"]].iloc[0]
    best_vol_fast = fast_df[fast_df["use_vol_regime"]].loc[
        fast_df[fast_df["use_vol_regime"]]["sharpe"].idxmax()
    ]

    sharpe_improvement = best_vol_fast["sharpe"] - baseline_fast["sharpe"]
    cagr_improvement = best_vol_fast["cagr"] - baseline_fast["cagr"]

    logger.info("FAST MODE COMPARISON:")
    logger.info(
        f"  Baseline: {baseline_fast['sharpe']:.3f} Sharpe, {baseline_fast['cagr']:.2%} CAGR"
    )
    logger.info(
        f"  Best Vol Regime: {best_vol_fast['config_name']} - {best_vol_fast['sharpe']:.3f} Sharpe, {best_vol_fast['cagr']:.2%} CAGR"
    )
    logger.info(
        f"  Improvement: {sharpe_improvement:+.3f} Sharpe, {cagr_improvement:+.2%} CAGR"
    )
    logger.info("")

    # Decision: run full validation if promising
    run_full_validation = sharpe_improvement > 0.02 or (
        sharpe_improvement > 0.01 and cagr_improvement > 0.01
    )

    if not run_full_validation:
        logger.info("=" * 80)
        logger.info("DECISION: SKIP FULL VALIDATION")
        logger.info("=" * 80)
        logger.info("Fast mode shows no meaningful improvement.")
        logger.info("Volatility regime does not appear to improve risk_on signal.")
        logger.info("")

        # Generate fast-mode-only report
        _generate_report(fast_df, None, fast_elapsed, 0, run_full_validation=False)
        return

    # === PHASE 2: Full validation ===
    logger.info("=" * 80)
    logger.info("PHASE 2: FULL VALIDATION")
    logger.info("=" * 80)
    logger.info("Volatility regime shows promise in fast mode.")
    logger.info("Running full backtest (all segments, full history)...")
    logger.info("")

    full_start = time.perf_counter()

    # Run full validation on baseline and best vol regime config
    full_configs = [
        {"name": "Baseline (momentum only)", "use_vol": False, "vol_weight": 0.0},
        {
            "name": best_vol_fast["config_name"],
            "use_vol": best_vol_fast["use_vol_regime"],
            "vol_weight": best_vol_fast["vol_weight"],
        },
    ]

    full_results = []

    for config in full_configs:
        df = _run_experiment(
            config_name=config["name"],
            use_vol_regime=config["use_vol"],
            vol_regime_weight=config["vol_weight"],
            fast_mode=False,
        )

        if df.empty:
            logger.error(f"Full validation failed for {config['name']}")
            continue

        overall = df[df["segment"] == "OVERALL"].iloc[0]
        difficult = _difficult_period_metrics(df)

        result = {
            "config_name": config["name"],
            "use_vol_regime": config["use_vol"],
            "vol_weight": config["vol_weight"],
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
            "difficult_cagr": difficult.get("difficult_cagr", np.nan),
            "difficult_sharpe": difficult.get("difficult_sharpe", np.nan),
            "difficult_maxdd": difficult.get("difficult_maxdd", np.nan),
        }
        full_results.append(result)

    full_elapsed = (time.perf_counter() - full_start) / 60

    if not full_results:
        logger.error("All full validation experiments failed.")
        sys.exit(1)

    full_df = pd.DataFrame(full_results)

    logger.info("")
    logger.info("=" * 80)
    logger.info("FULL VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Runtime: {full_elapsed:.1f} minutes")
    logger.info("")
    logger.info("| Configuration | CAGR | Sharpe | MaxDD | Turnover |")
    logger.info("|---------------|------|--------|-------|----------|")

    for _, row in full_df.iterrows():
        logger.info(
            f"| {row['config_name']:>35} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )

    logger.info("")

    # Generate report
    _generate_report(
        fast_df, full_df, fast_elapsed, full_elapsed, run_full_validation=True
    )


def _generate_report(
    fast_df: pd.DataFrame,
    full_df: pd.DataFrame | None,
    fast_elapsed: float,
    full_elapsed: float,
    run_full_validation: bool,
):
    """Generate experiment report."""
    report_lines = [
        "# Volatility Regime Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Fixed Setup:**",
        "- Pure market model (0.0 macro / 1.0 market momentum)",
        "- 24-month momentum signal",
        "- Equal-weight sleeve construction",
        "- Corrected no-lookahead implementation",
        "",
        "**Goal:**",
        "Add an orthogonal volatility regime signal to improve risk_on estimation.",
        "",
        "**Volatility Regime Signal:**",
        "1. Compute 63-day realized volatility of SPY",
        "2. Convert to percentile over trailing 3 years",
        "3. Z-score normalize (expanding window)",
        "4. Invert: high vol → negative score (reduce risk_on)",
        "          low vol → positive score (increase risk_on)",
        "",
        "**Signal Combination:**",
        "```",
        "risk_on = sigmoid(",
        "    momentum_weight * momentum_zscore",
        "    + vol_regime_weight * vol_regime_zscore",
        ")```",
        "",
        "Where weights are normalized to sum to 1.0.",
        "",
        "## Phase 1: Fast Mode Screening",
        "",
        f"**Runtime:** {fast_elapsed:.1f} minutes",
        "",
        "| Configuration | Vol Weight | CAGR | Sharpe | MaxDD | Turnover |",
        "|---------------|------------|------|--------|-------|----------|",
    ]

    for _, row in fast_df.iterrows():
        report_lines.append(
            f"| {row['config_name']:>35} | {row['vol_weight']:>10.2f} | {row['cagr']:>4.2%} | "
            f"{row['sharpe']:>6.3f} | {row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )

    report_lines.append("")

    baseline_fast = fast_df[not fast_df["use_vol_regime"]].iloc[0]
    best_vol_fast = fast_df[fast_df["use_vol_regime"]].loc[
        fast_df[fast_df["use_vol_regime"]]["sharpe"].idxmax()
    ]

    report_lines.extend(
        [
            "### Fast Mode Comparison",
            "",
            f"**Baseline (momentum only):** {baseline_fast['sharpe']:.3f} Sharpe, {baseline_fast['cagr']:.2%} CAGR",
            f"**Best Vol Regime:** {best_vol_fast['config_name']} - {best_vol_fast['sharpe']:.3f} Sharpe, {best_vol_fast['cagr']:.2%} CAGR",
            "",
            f"**Improvement:** {best_vol_fast['sharpe'] - baseline_fast['sharpe']:+.3f} Sharpe, "
            f"{best_vol_fast['cagr'] - baseline_fast['cagr']:+.2%} CAGR",
            "",
        ]
    )

    # Full validation results
    if run_full_validation and full_df is not None:
        baseline_full = full_df[not full_df["use_vol_regime"]].iloc[0]
        vol_regime_full = full_df[full_df["use_vol_regime"]].iloc[0]

        report_lines.extend(
            [
                "## Phase 2: Full Validation",
                "",
                f"**Runtime:** {full_elapsed:.1f} minutes",
                "",
                "| Configuration | CAGR | Sharpe | MaxDD | Vol | Turnover |",
                "|---------------|------|--------|-------|-----|----------|",
            ]
        )

        for _, row in full_df.iterrows():
            report_lines.append(
                f"| {row['config_name']:>35} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
                f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
            )

        report_lines.extend(
            [
                "",
                "### Full Validation Comparison",
                "",
                f"**Baseline:** {baseline_full['sharpe']:.3f} Sharpe, {baseline_full['cagr']:.2%} CAGR",
                f"**Vol Regime:** {vol_regime_full['sharpe']:.3f} Sharpe, {vol_regime_full['cagr']:.2%} CAGR",
                "",
                f"**Improvement:** {vol_regime_full['sharpe'] - baseline_full['sharpe']:+.3f} Sharpe, "
                f"{vol_regime_full['cagr'] - baseline_full['cagr']:+.2%} CAGR",
                "",
                "### Difficult Period Performance (2021-2022)",
                "",
                "| Configuration | CAGR | Sharpe | MaxDD |",
                "|---------------|------|--------|-------|",
            ]
        )

        for _, row in full_df.iterrows():
            if not pd.isna(row["difficult_cagr"]):
                report_lines.append(
                    f"| {row['config_name']:>35} | {row['difficult_cagr']:>4.2%} | "
                    f"{row['difficult_sharpe']:>6.3f} | {row['difficult_maxdd']:>5.2%} |"
                )

        report_lines.extend(
            [
                "",
                "### Analysis",
                "",
            ]
        )

        sharpe_imp_full = vol_regime_full["sharpe"] - baseline_full["sharpe"]
        turnover_diff_full = vol_regime_full["turnover"] - baseline_full["turnover"]
        difficult_sharpe_imp = (
            vol_regime_full["difficult_sharpe"] - baseline_full["difficult_sharpe"]
        )

        # Recommendation based on full validation
        if sharpe_imp_full > 0.03:
            verdict = "**ADOPT VOLATILITY REGIME**"
            explanation = (
                f"Volatility regime materially improves risk_on signal:\n"
                f"- Sharpe: {sharpe_imp_full:+.3f}\n"
                f"- CAGR: {vol_regime_full['cagr'] - baseline_full['cagr']:+.2%}\n"
                f"- Turnover: {turnover_diff_full:+.1%}\n"
                f"- Difficult period Sharpe: {difficult_sharpe_imp:+.3f}\n"
                f"- Optimal weight: {vol_regime_full['vol_weight']:.2f}"
            )
        elif sharpe_imp_full > 0.0:
            verdict = "**MARGINAL IMPROVEMENT: Consider vol regime**"
            explanation = (
                f"Volatility regime improves Sharpe by {sharpe_imp_full:+.3f}.\n"
                "Benefit is modest but may justify added complexity if:\n"
                f"- Difficult periods improve materially ({difficult_sharpe_imp:+.3f})\n"
                f"- Turnover impact is acceptable ({turnover_diff_full:+.1%})\n"
                f"- Implementation cost is low"
            )
        else:
            verdict = (
                "**KEEP BASELINE: Volatility regime does not improve performance**"
            )
            explanation = (
                f"Volatility regime underperforms momentum-only baseline by {sharpe_imp_full:+.3f} Sharpe.\n"
                "The momentum signal alone is sufficient.\n"
                "Volatility regime adds complexity without improving risk-adjusted returns."
            )

        report_lines.append(verdict)
        report_lines.append("")
        report_lines.append(explanation)
        report_lines.append("")

    else:
        # Fast mode only (no full validation)
        report_lines.extend(
            [
                "## Fast Mode Decision",
                "",
            ]
        )

        sharpe_imp_fast = best_vol_fast["sharpe"] - baseline_fast["sharpe"]

        verdict = "**SKIP FULL VALIDATION: No improvement in fast mode**"
        explanation = (
            f"Fast mode screening shows {sharpe_imp_fast:+.3f} Sharpe improvement.\n"
            "This is below the threshold (+0.02) for running full validation.\n"
            "Volatility regime does not appear to improve the risk_on signal."
        )

        report_lines.append(verdict)
        report_lines.append("")
        report_lines.append(explanation)
        report_lines.append("")

        # Analysis of why it didn't work
        report_lines.extend(
            [
                "### Why Volatility Regime Doesn't Help",
                "",
                "Possible reasons:",
                "1. **Momentum already captures volatility information**",
                "   - High vol periods often coincide with negative momentum",
                "   - The two signals are correlated, not orthogonal",
                "",
                "2. **Volatility percentile is noisy**",
                "   - Short-term vol spikes may not persist",
                "   - 3-year percentile may be too backward-looking",
                "",
                "3. **Risk_on blending already smooths exposure**",
                "   - Equal-weight construction with dynamic blending provides natural de-risking",
                "   - Adding vol regime may be redundant",
                "",
            ]
        )

    # Next experiment recommendations
    report_lines.extend(
        [
            "## Next Experiment Recommendations",
            "",
        ]
    )

    if run_full_validation and full_df is not None:
        vol_regime_full = full_df[full_df["use_vol_regime"]].iloc[0]
        baseline_full = full_df[not full_df["use_vol_regime"]].iloc[0]

        if vol_regime_full["sharpe"] > baseline_full["sharpe"] + 0.03:
            report_lines.append(
                "1. **Test alternative vol regime construction** - VIX, implied vol, vol-of-vol"
            )
            report_lines.append(
                "2. **Test vol regime windows** - different lookbacks for realized vol"
            )
            report_lines.append(
                "3. **Test regime-specific vol thresholds** - vary by macro environment"
            )
        else:
            report_lines.append(
                "1. **Test rebalance frequency** - quarterly vs monthly (reduce turnover)"
            )
            report_lines.append(
                "2. **Test alternative orthogonal signals** - credit spreads, yield curve, breadth"
            )
            report_lines.append(
                "3. **Test regime-conditional universes** - vary assets by macro regime"
            )
    else:
        report_lines.append(
            "1. **Test rebalance frequency** - quarterly vs monthly (reduce turnover)"
        )
        report_lines.append(
            "2. **Test alternative orthogonal signals** - credit spreads, yield curve, market breadth"
        )
        report_lines.append(
            "3. **Test regime-conditional universes** - vary assets by macro regime"
        )

    report_lines.append(
        "4. **Test cross-sectional momentum** - rank assets by relative strength"
    )
    report_lines.append(
        "5. **Test alternative signal horizons** - blend multiple momentum lookbacks"
    )

    report_lines.extend(
        [
            "",
            "## Summary",
            "",
            "### Fast Mode Performance",
            "",
            "| Configuration | Vol Weight | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe |",
            "|---------------|------------|------|--------|-------|-----|----------|------------------|",
        ]
    )

    for _, row in fast_df.iterrows():
        report_lines.append(
            f"| {row['config_name']:>35} | {row['vol_weight']:>10.2f} | {row['cagr']:>4.2%} | "
            f"{row['sharpe']:>6.3f} | {row['maxdd']:>5.2%} | {row['vol']:>3.2%} | "
            f"{row['turnover']:>8.1%} | {row['difficult_sharpe']:>16.3f} |"
        )

    if run_full_validation and full_df is not None:
        report_lines.extend(
            [
                "",
                "### Full Validation Performance",
                "",
                "| Configuration | Vol Weight | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe |",
                "|---------------|------------|------|--------|-------|-----|----------|------------------|",
            ]
        )

        for _, row in full_df.iterrows():
            report_lines.append(
                f"| {row['config_name']:>35} | {row['vol_weight']:>10.2f} | {row['cagr']:>4.2%} | "
                f"{row['sharpe']:>6.3f} | {row['maxdd']:>5.2%} | {row['vol']:>3.2%} | "
                f"{row['turnover']:>8.1%} | {row['difficult_sharpe']:>16.3f} |"
            )

    report = "\n".join(report_lines)

    output_path = OUTPUTS_DIR / "VOLATILITY_REGIME_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {output_path}")
    logger.info("")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
