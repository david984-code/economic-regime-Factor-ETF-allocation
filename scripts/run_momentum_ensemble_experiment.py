"""Test momentum ensemble signals with equal-weight construction."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import OUTPUTS_DIR, START_DATE, get_end_date
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment(
    primary_lookback: int,
    momentum_12m_weight: float,
) -> pd.DataFrame:
    """Run equal-weight momentum experiment with specified ensemble."""
    if momentum_12m_weight == 0.0:
        logger.info("=" * 80)
        logger.info("RUNNING: %dM momentum only", primary_lookback)
        logger.info("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info(
            "RUNNING: %.0f%% 12M + %.0f%% %dM ensemble",
            momentum_12m_weight * 100,
            (1 - momentum_12m_weight) * 100,
            primary_lookback,
        )
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
        market_lookback_months=primary_lookback,
        use_momentum=True,  # Momentum
        trend_filter_type="none",
        vol_scaling_method="none",
        portfolio_construction_method="equal_weight",  # Equal weight
        momentum_12m_weight=momentum_12m_weight,
    )
    return df


def _compute_signal_ic(
    prices: pd.DataFrame,
    primary_lookback: int,
    momentum_12m_weight: float,
) -> dict[str, float]:
    """Compute IC for ensemble signal (no lookahead)."""
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_returns_monthly = spy_monthly.pct_change().dropna()
    spy_returns_monthly.index = spy_returns_monthly.index.to_period("M")

    # Compute primary momentum
    primary_list = []
    for i in range(len(spy_monthly)):
        if i < primary_lookback:
            primary_list.append(np.nan)
        else:
            price_now = spy_monthly.iloc[i]
            price_lookback = spy_monthly.iloc[i - primary_lookback]
            momentum = (price_now / price_lookback) - 1
            primary_list.append(momentum)

    primary_signal = pd.Series(primary_list, index=spy_monthly.index)

    # If ensemble, compute 12M as well
    if momentum_12m_weight > 0:
        momentum_12m_list = []
        for i in range(len(spy_monthly)):
            if i < 12:
                momentum_12m_list.append(np.nan)
            else:
                price_now = spy_monthly.iloc[i]
                price_12m_ago = spy_monthly.iloc[i - 12]
                momentum_12m = (price_now / price_12m_ago) - 1
                momentum_12m_list.append(momentum_12m)

        momentum_12m_signal = pd.Series(momentum_12m_list, index=spy_monthly.index)
        blended_signal = (
            1 - momentum_12m_weight
        ) * primary_signal + momentum_12m_weight * momentum_12m_signal
    else:
        blended_signal = primary_signal

    blended_signal.index = blended_signal.index.to_period("M")

    # Expanding window z-score (no lookahead)
    signal_z = blended_signal.copy()
    min_history = max(primary_lookback, 12)
    for i in range(len(blended_signal)):
        trailing = blended_signal.iloc[: i + 1].dropna()
        if len(trailing) >= min_history:
            signal_z.iloc[i] = (
                blended_signal.iloc[i] - trailing.mean()
            ) / trailing.std()
        else:
            signal_z.iloc[i] = 0.0

    # Align indices
    common = spy_returns_monthly.index.intersection(signal_z.index)
    spy_ret = spy_returns_monthly.loc[common]
    signal = signal_z.loc[common]

    # Forward returns
    fwd_1m = (1 + spy_ret).rolling(1).apply(lambda x: x.prod() - 1, raw=True).shift(-1)
    fwd_3m = (1 + spy_ret).rolling(3).apply(lambda x: x.prod() - 1, raw=True).shift(-3)
    fwd_6m = (1 + spy_ret).rolling(6).apply(lambda x: x.prod() - 1, raw=True).shift(-6)

    df = pd.DataFrame(
        {
            "signal": signal,
            "fwd_1m": fwd_1m,
            "fwd_3m": fwd_3m,
            "fwd_6m": fwd_6m,
        }
    ).dropna()

    if len(df) < 10:
        return {
            "ic_1m": np.nan,
            "ic_3m": np.nan,
            "ic_6m": np.nan,
            "ic_avg": np.nan,
        }

    ic_1m, _ = spearmanr(df["signal"], df["fwd_1m"])
    ic_3m, _ = spearmanr(df["signal"], df["fwd_3m"])
    ic_6m, _ = spearmanr(df["signal"], df["fwd_6m"])

    return {
        "ic_1m": ic_1m,
        "ic_3m": ic_3m,
        "ic_6m": ic_6m,
        "ic_avg": np.mean([ic_1m, ic_3m, ic_6m]),
    }


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
    """Run momentum ensemble experiment with equal-weight construction."""
    logger.info("Starting momentum ensemble experiment")
    logger.info("Fixed portfolio construction: Equal Weight")

    # Test configurations
    configs = [
        {"primary": 24, "weight_12m": 0.00, "name": "24M only"},
        {"primary": 12, "weight_12m": 0.00, "name": "12M only"},
        {"primary": 24, "weight_12m": 0.50, "name": "50/50 (12M+24M)"},
        {"primary": 24, "weight_12m": 0.25, "name": "25/75 (12M+24M)"},
        {"primary": 24, "weight_12m": 0.75, "name": "75/25 (12M+24M)"},
    ]

    results = []
    ic_results = []

    # Load price data once for IC calculations
    logger.info("Loading data for IC calculations...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())

    # Run experiments
    for config in configs:
        df = _run_experiment(config["primary"], config["weight_12m"])

        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue

        overall = df[df["segment"] == "OVERALL"].iloc[0]

        # Compute IC
        ic = _compute_signal_ic(prices, config["primary"], config["weight_12m"])

        # Difficult period metrics
        difficult = _difficult_period_metrics(df)

        result = {
            "signal": config["name"],
            "primary_lookback": config["primary"],
            "weight_12m": config["weight_12m"],
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
            "difficult_cagr": difficult.get("difficult_cagr", np.nan),
            "difficult_sharpe": difficult.get("difficult_sharpe", np.nan),
            "difficult_maxdd": difficult.get("difficult_maxdd", np.nan),
        }
        results.append(result)

        ic_result = {
            "signal": config["name"],
            **ic,
        }
        ic_results.append(ic_result)

    if not results:
        logger.error("All experiments failed.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    ic_df = pd.DataFrame(ic_results)

    # Build report
    report_lines = [
        "# Momentum Ensemble Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Fixed Setup:**",
        "- Pure market model (0.0 macro / 1.0 market)",
        "- Equal-weight portfolio construction",
        "- No trend filter",
        "- No volatility scaling",
        "- Corrected no-lookahead implementation",
        "",
        "**Signal Variants Tested:**",
        "",
        "1. **24M only**: 24-month momentum (current baseline)",
        "2. **12M only**: 12-month momentum",
        "3. **50/50 (12M+24M)**: Equal blend of 12M and 24M momentum",
        "4. **25/75 (12M+24M)**: 25% 12M + 75% 24M momentum",
        "5. **75/25 (12M+24M)**: 75% 12M + 25% 24M momentum",
        "",
        "All signals are momentum (+momentum), not mean-reversion.",
        "",
        "## Walk-Forward Performance by Signal",
        "",
        "| Signal | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|--------|------|--------|-------|-----|----------|",
    ]

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['signal']:>18} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )

    report_lines.append("")

    # Best by different metrics
    baseline = results_df[results_df["signal"] == "24M only"].iloc[0]
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_maxdd = results_df.loc[results_df["maxdd"].idxmax()]
    best_turnover = results_df.loc[results_df["turnover"].idxmin()]

    report_lines.extend(
        [
            "### Top Performers",
            "",
            f"- **Best CAGR**: {best_cagr['signal']} ({best_cagr['cagr']:.2%})",
            f"- **Best Sharpe**: {best_sharpe['signal']} ({best_sharpe['sharpe']:.3f})",
            f"- **Best MaxDD**: {best_maxdd['signal']} ({best_maxdd['maxdd']:.2%})",
            f"- **Lowest Turnover**: {best_turnover['signal']} ({best_turnover['turnover']:.1%})",
            "",
            "## Information Coefficient by Signal",
            "",
            "| Signal | 1M IC | 3M IC | 6M IC | Avg IC |",
            "|--------|-------|-------|-------|--------|",
        ]
    )

    for _, row in ic_df.iterrows():
        report_lines.append(
            f"| {row['signal']:>18} | {row['ic_1m']:>5.4f} | {row['ic_3m']:>5.4f} | "
            f"{row['ic_6m']:>5.4f} | {row['ic_avg']:>6.4f} |"
        )

    report_lines.append("")

    best_ic = ic_df.loc[ic_df["ic_avg"].idxmax()]
    report_lines.append(
        f"**Best IC**: {best_ic['signal']} (avg IC = {best_ic['ic_avg']:.4f})"
    )
    report_lines.append("")

    # Difficult period performance
    report_lines.extend(
        [
            "## Difficult Period Performance (2021-2022)",
            "",
            "| Signal | CAGR | Sharpe | MaxDD |",
            "|--------|------|--------|-------|",
        ]
    )

    for _, row in results_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['signal']:>18} | {row['difficult_cagr']:>4.2%} | "
                f"{row['difficult_sharpe']:>6.3f} | {row['difficult_maxdd']:>5.2%} |"
            )

    report_lines.append("")

    # Difficult period analysis
    baseline_difficult_cagr = baseline["difficult_cagr"]
    baseline_difficult_sharpe = baseline["difficult_sharpe"]

    best_difficult_cagr = results_df.loc[results_df["difficult_cagr"].idxmax()]
    best_difficult_sharpe = results_df.loc[results_df["difficult_sharpe"].idxmax()]

    report_lines.extend(
        [
            "### Difficult Period Analysis",
            "",
            f"**Baseline (24M only):** {baseline_difficult_cagr:.2%} CAGR, {baseline_difficult_sharpe:.3f} Sharpe",
            "",
            f"**Best Difficult CAGR**: {best_difficult_cagr['signal']} "
            f"({best_difficult_cagr['difficult_cagr']:.2%}, "
            f"+{best_difficult_cagr['difficult_cagr'] - baseline_difficult_cagr:.2%} vs baseline)",
            "",
            f"**Best Difficult Sharpe**: {best_difficult_sharpe['signal']} "
            f"({best_difficult_sharpe['difficult_sharpe']:.3f}, "
            f"+{best_difficult_sharpe['difficult_sharpe'] - baseline_difficult_sharpe:.3f} vs baseline)",
            "",
            "## Analysis",
            "",
            "### Impact vs 24M Baseline",
            "",
            "| Signal | dCAGR | dSharpe | dMaxDD | dVol | dTurnover |",
            "|--------|-------|---------|--------|------|-----------|",
        ]
    )

    for _, row in results_df.iterrows():
        if row["signal"] == "24M only":
            continue
        delta_cagr = row["cagr"] - baseline["cagr"]
        delta_sharpe = row["sharpe"] - baseline["sharpe"]
        delta_maxdd = row["maxdd"] - baseline["maxdd"]
        delta_vol = row["vol"] - baseline["vol"]
        delta_turnover = row["turnover"] - baseline["turnover"]
        report_lines.append(
            f"| {row['signal']:>18} | {delta_cagr:>5.2%} | {delta_sharpe:>7.3f} | "
            f"{delta_maxdd:>6.2%} | {delta_vol:>4.2%} | {delta_turnover:>9.1%} |"
        )

    report_lines.append("")

    # Ensemble benefit analysis
    report_lines.extend(
        [
            "### Ensemble vs Single-Horizon Signals",
            "",
        ]
    )

    single_signals = results_df[results_df["weight_12m"] == 0.0]
    ensemble_signals = results_df[results_df["weight_12m"] > 0.0]

    if len(ensemble_signals) > 0:
        avg_single_sharpe = single_signals["sharpe"].mean()
        avg_ensemble_sharpe = ensemble_signals["sharpe"].mean()

        avg_single_turnover = single_signals["turnover"].mean()
        avg_ensemble_turnover = ensemble_signals["turnover"].mean()

        report_lines.append("**Single-horizon signals:**")
        report_lines.append(f"- Average Sharpe: {avg_single_sharpe:.3f}")
        report_lines.append(f"- Average Turnover: {avg_single_turnover:.1%}")
        report_lines.append("")
        report_lines.append("**Ensemble signals:**")
        report_lines.append(f"- Average Sharpe: {avg_ensemble_sharpe:.3f}")
        report_lines.append(f"- Average Turnover: {avg_ensemble_turnover:.1%}")
        report_lines.append("")

        if avg_ensemble_sharpe > avg_single_sharpe:
            improvement = avg_ensemble_sharpe - avg_single_sharpe
            report_lines.append(
                f"**Ensemble improves Sharpe by {improvement:+.3f} on average**"
            )
        else:
            degradation = avg_single_sharpe - avg_ensemble_sharpe
            report_lines.append(
                f"**Ensemble reduces Sharpe by {degradation:.3f} on average**"
            )

        report_lines.append("")

    # Responsiveness analysis
    report_lines.extend(
        [
            "### Responsiveness vs Stability",
            "",
        ]
    )

    # Check if shorter horizons add responsiveness without hurting performance
    momentum_12m_only = results_df[results_df["signal"] == "12M only"].iloc[0]

    if momentum_12m_only["sharpe"] > baseline["sharpe"]:
        report_lines.append("**12M outperforms 24M**: Shorter horizon is better")
        report_lines.append(f"- 12M Sharpe: {momentum_12m_only['sharpe']:.3f}")
        report_lines.append(f"- 24M Sharpe: {baseline['sharpe']:.3f}")
        report_lines.append(
            f"- Improvement: {momentum_12m_only['sharpe'] - baseline['sharpe']:+.3f}"
        )
    else:
        report_lines.append("**24M outperforms 12M**: Longer horizon is better")
        report_lines.append(f"- 24M Sharpe: {baseline['sharpe']:.3f}")
        report_lines.append(f"- 12M Sharpe: {momentum_12m_only['sharpe']:.3f}")
        report_lines.append(
            f"- Advantage: {baseline['sharpe'] - momentum_12m_only['sharpe']:+.3f}"
        )

    report_lines.append("")

    # Robustness score
    results_df["robustness_score"] = (
        results_df["sharpe"] / results_df["sharpe"].max() * 0.4
        + (1 - (results_df["maxdd"].abs() / results_df["maxdd"].abs().max())) * 0.2
        + (1 - results_df["turnover"] / results_df["turnover"].max()) * 0.2
        + (results_df["difficult_sharpe"] / results_df["difficult_sharpe"].max()) * 0.2
    )

    best_robust = results_df.loc[results_df["robustness_score"].idxmax()]

    report_lines.extend(
        [
            "### Robustness Score",
            "",
            "Robustness = 0.4*Sharpe + 0.2*(1-|MaxDD|) + 0.2*(1-Turnover) + 0.2*Difficult_Sharpe",
            "",
            "| Signal | Score |",
            "|--------|-------|",
        ]
    )

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['signal']:>18} | {row['robustness_score']:>5.3f} |"
        )

    report_lines.extend(
        [
            "",
            f"**Most Robust**: {best_robust['signal']}",
            "",
            "## Recommendation",
            "",
        ]
    )

    # Final recommendation
    baseline_sharpe = baseline["sharpe"]
    best_signal = results_df.loc[results_df["sharpe"].idxmax()]

    sharpe_improvement = best_signal["sharpe"] - baseline_sharpe
    turnover_diff = best_signal["turnover"] - baseline["turnover"]

    if best_signal["signal"] == "24M only":
        verdict = "**KEEP 24M: Current baseline is optimal**"
        explanation = (
            "The 24M momentum signal provides the best risk-adjusted returns.\n"
            "Neither 12M alone nor ensemble blends improve on the baseline."
        )
    elif sharpe_improvement > 0.02:
        verdict = f"**CHANGE TO {best_signal['signal']}**"
        explanation = (
            f"The {best_signal['signal']} signal materially outperforms 24M:\n"
            f"- Sharpe: {sharpe_improvement:+.3f}\n"
            f"- CAGR: {best_signal['cagr'] - baseline['cagr']:+.2%}\n"
            f"- Turnover: {turnover_diff:+.1%}"
        )
    else:
        verdict = "**KEEP 24M: Marginal differences only**"
        explanation = (
            f"Best alternative ({best_signal['signal']}) improves Sharpe by only {sharpe_improvement:+.3f}.\n"
            "The improvement is not material enough to justify changing the baseline."
        )

    report_lines.append(verdict)
    report_lines.append("")
    report_lines.append(explanation)
    report_lines.append("")

    if best_signal["signal"] == "24M only":
        report_lines.extend(
            [
                "### Why 24M Remains Best",
                "",
            ]
        )

        avg_ensemble_sharpe = (
            ensemble_signals["sharpe"].mean() if len(ensemble_signals) > 0 else 0
        )
        avg_ensemble_turnover = (
            ensemble_signals["turnover"].mean() if len(ensemble_signals) > 0 else 0
        )

        if avg_ensemble_sharpe < baseline_sharpe:
            report_lines.append(
                f"- Ensembles reduce Sharpe on average (avg: {avg_ensemble_sharpe:.3f} vs baseline: {baseline_sharpe:.3f})"
            )
        if avg_ensemble_turnover > baseline["turnover"]:
            report_lines.append(
                f"- Ensembles increase turnover (avg: {avg_ensemble_turnover:.1%} vs baseline: {baseline['turnover']:.1%})"
            )

        report_lines.append("- 24M momentum captures long-term trends effectively")
        report_lines.append(
            "- Adding shorter horizons introduces noise without benefit"
        )
        report_lines.append("")

    report_lines.extend(
        [
            "## Next Experiment Recommendations",
            "",
        ]
    )

    if best_signal["signal"] != "24M only":
        report_lines.append(
            "1. **Fine-tune ensemble weights** - test more granular blends"
        )
        report_lines.append(
            "2. **Test adaptive ensemble** - vary weights based on market conditions"
        )
    else:
        report_lines.append(
            "1. **Expand asset universe** - test sector rotation, international, alternatives"
        )
        report_lines.append(
            "2. **Test asset-specific momentum** - apply momentum to individual assets, not just market"
        )
        report_lines.append(
            "3. **Test regime-conditional signals** - vary signal by macro regime"
        )

    report_lines.append(
        "4. **Test concentration limits** - constrain max position size in equal weight"
    )
    report_lines.append(
        "5. **Test dynamic universe** - vary assets based on momentum signal strength"
    )

    report_lines.extend(
        [
            "",
            "## Summary Statistics",
            "",
            "| Signal | CAGR | Sharpe | Turnover | IC | Difficult Sharpe | Robustness |",
            "|--------|------|--------|----------|-----|------------------|------------|",
        ]
    )

    for i, row in results_df.iterrows():
        ic_row = ic_df.iloc[i]
        report_lines.append(
            f"| {row['signal']:>18} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['turnover']:>8.1%} | {ic_row['ic_avg']:>7.4f} | "
            f"{row['difficult_sharpe']:>16.3f} | {row['robustness_score']:>10.3f} |"
        )

    report = "\n".join(report_lines)

    output_path = OUTPUTS_DIR / "MOMENTUM_ENSEMBLE_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)

    logger.info("\nExperiment complete. Report generated.")


if __name__ == "__main__":
    main()
