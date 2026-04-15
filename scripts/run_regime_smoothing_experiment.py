"""Run baseline vs regime_smoothing experiment and compare results."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment(name: str, **kwargs) -> pd.DataFrame:
    """Run experiment and return results."""
    logger.info("=" * 60)
    logger.info("RUNNING: %s", name)
    logger.info("=" * 60)
    df = run_walk_forward_evaluation(
        min_train_months=60, test_months=12, expanding=True, **kwargs
    )
    return df


def _regime_transition_rate(
    regime_df: pd.DataFrame,
    smoothed: bool = False,
    window: int = 3,
) -> float:
    """Compute regime transition rate per 12 months."""
    regime_series = regime_df["regime"].copy()

    if smoothed:
        from src.backtest.engine import _smooth_regime_labels

        regime_df_smoothed = _smooth_regime_labels(regime_df, window=window)
        regime_series = regime_df_smoothed["regime"]

    monthly = regime_series.resample("ME").last().dropna()
    transitions = (monthly != monthly.shift(1)).sum() - 1
    n_months = len(monthly)
    rate_per_12mo = transitions / n_months * 12 if n_months > 0 else 0.0
    return rate_per_12mo


def main():
    """Run baseline vs regime_smoothing experiment and generate comparison report."""
    logger.info("Starting regime smoothing experiment")

    # Run baseline (no smoothing)
    baseline = _run_experiment(
        "Baseline (no smoothing)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
    )

    # Run smoothed (3-month window)
    smoothed = _run_experiment(
        "Regime Smoothing (3-month window)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=True,
        regime_smoothing_window=3,
    )

    if baseline.empty or smoothed.empty:
        logger.error("Experiments failed.")
        sys.exit(1)

    # Extract OVERALL results
    baseline_overall = baseline[baseline["segment"] == "OVERALL"].iloc[0]
    smoothed_overall = smoothed[smoothed["segment"] == "OVERALL"].iloc[0]

    # Calculate regime transition rates
    logger.info("Computing regime transition rates...")
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()

    transition_rate_baseline = _regime_transition_rate(regime_df, smoothed=False)
    transition_rate_smoothed = _regime_transition_rate(
        regime_df, smoothed=True, window=3
    )

    # Build comparison report
    report_lines = [
        "# Regime Smoothing Experiment Report",
        "",
        "## Experiment Setup",
        "",
        "- **Baseline**: Use raw regime labels for allocation decisions",
        "- **Smoothed**: Apply rolling 3-month mode to regime labels before allocation",
        "",
        "## Overall Performance",
        "",
        "| Metric          | Baseline | Smoothed | Difference |",
        "|-----------------|----------|----------|------------|",
        f"| CAGR            | {baseline_overall['Strategy_CAGR']:.2%} | {smoothed_overall['Strategy_CAGR']:.2%} | {smoothed_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']:.2%} |",
        f"| Sharpe Ratio    | {baseline_overall['Strategy_Sharpe']:.3f} | {smoothed_overall['Strategy_Sharpe']:.3f} | {smoothed_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']:.3f} |",
        f"| Max Drawdown    | {baseline_overall['Strategy_MaxDD']:.2%} | {smoothed_overall['Strategy_MaxDD']:.2%} | {smoothed_overall['Strategy_MaxDD'] - baseline_overall['Strategy_MaxDD']:.2%} |",
        f"| Volatility      | {baseline_overall['Strategy_Vol']:.2%} | {smoothed_overall['Strategy_Vol']:.2%} | {smoothed_overall['Strategy_Vol'] - baseline_overall['Strategy_Vol']:.2%} |",
        f"| Turnover        | {baseline_overall.get('Strategy_Turnover', 0.0):.1%} | {smoothed_overall.get('Strategy_Turnover', 0.0):.1%} | {smoothed_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):.1%} |",
        "",
        "## Regime Stability",
        "",
        f"- **Baseline transitions**: {transition_rate_baseline:.1f} transitions per 12 months",
        f"- **Smoothed transitions**: {transition_rate_smoothed:.1f} transitions per 12 months",
        f"- **Reduction**: {transition_rate_baseline - transition_rate_smoothed:.1f} transitions",
        "",
        "## Benchmark Comparison",
        "",
        "### Baseline",
        "",
        "| Benchmark    | CAGR     | Sharpe   | MaxDD    |",
        "|--------------|----------|----------|----------|",
        f"| SPY          | {baseline_overall.get('SPY_CAGR', 0.0):.2%} | {baseline_overall.get('SPY_Sharpe', 0.0):.3f} | {baseline_overall.get('SPY_MaxDD', 0.0):.2%} |",
        f"| 60/40        | {baseline_overall.get('60/40_CAGR', 0.0):.2%} | {baseline_overall.get('60/40_Sharpe', 0.0):.3f} | {baseline_overall.get('60/40_MaxDD', 0.0):.2%} |",
        f"| Equal_Weight | {baseline_overall.get('Equal_Weight_CAGR', 0.0):.2%} | {baseline_overall.get('Equal_Weight_Sharpe', 0.0):.3f} | {baseline_overall.get('Equal_Weight_MaxDD', 0.0):.2%} |",
        f"| Risk_On_Off  | {baseline_overall.get('Risk_On_Off_CAGR', 0.0):.2%} | {baseline_overall.get('Risk_On_Off_Sharpe', 0.0):.3f} | {baseline_overall.get('Risk_On_Off_MaxDD', 0.0):.2%} |",
        "",
        "### Smoothed",
        "",
        "| Benchmark    | CAGR     | Sharpe   | MaxDD    |",
        "|--------------|----------|----------|----------|",
        f"| SPY          | {smoothed_overall.get('SPY_CAGR', 0.0):.2%} | {smoothed_overall.get('SPY_Sharpe', 0.0):.3f} | {smoothed_overall.get('SPY_MaxDD', 0.0):.2%} |",
        f"| 60/40        | {smoothed_overall.get('60/40_CAGR', 0.0):.2%} | {smoothed_overall.get('60/40_Sharpe', 0.0):.3f} | {smoothed_overall.get('60/40_MaxDD', 0.0):.2%} |",
        f"| Equal_Weight | {smoothed_overall.get('Equal_Weight_CAGR', 0.0):.2%} | {smoothed_overall.get('Equal_Weight_Sharpe', 0.0):.3f} | {smoothed_overall.get('Equal_Weight_MaxDD', 0.0):.2%} |",
        f"| Risk_On_Off  | {smoothed_overall.get('Risk_On_Off_CAGR', 0.0):.2%} | {smoothed_overall.get('Risk_On_Off_Sharpe', 0.0):.3f} | {smoothed_overall.get('Risk_On_Off_MaxDD', 0.0):.2%} |",
        "",
        "## Conclusion",
        "",
    ]

    cagr_diff = smoothed_overall["Strategy_CAGR"] - baseline_overall["Strategy_CAGR"]
    sharpe_diff = (
        smoothed_overall["Strategy_Sharpe"] - baseline_overall["Strategy_Sharpe"]
    )
    turnover_diff = smoothed_overall.get(
        "Strategy_Turnover", 0.0
    ) - baseline_overall.get("Strategy_Turnover", 0.0)

    if cagr_diff > 0.01 and sharpe_diff > 0.05:
        verdict = (
            "ACCEPT: Smoothing improves both CAGR and Sharpe with reduced turnover."
        )
    elif cagr_diff > 0 and sharpe_diff > 0:
        verdict = "MARGINAL: Small improvements in CAGR and Sharpe."
    elif turnover_diff < -0.1 and abs(cagr_diff) < 0.01:
        verdict = "NEUTRAL: Lower turnover with similar returns."
    else:
        verdict = "REJECT: Smoothing does not improve performance."

    report_lines.append(f"**{verdict}**")
    report_lines.append("")

    if cagr_diff > 0:
        report_lines.append(f"- Smoothing increased CAGR by {cagr_diff:.2%}")
    else:
        report_lines.append(f"- Smoothing decreased CAGR by {-cagr_diff:.2%}")

    if sharpe_diff > 0:
        report_lines.append(f"- Smoothing increased Sharpe by {sharpe_diff:.3f}")
    else:
        report_lines.append(f"- Smoothing decreased Sharpe by {-sharpe_diff:.3f}")

    report_lines.append(
        f"- Smoothing reduced regime transitions from {transition_rate_baseline:.1f} to {transition_rate_smoothed:.1f} per 12 months"
    )
    report_lines.append(f"- Smoothing changed turnover by {turnover_diff:.1%}")

    report = "\n".join(report_lines)

    output_path = OUTPUTS_DIR / "REGIME_SMOOTHING_EXPERIMENT.md"
    output_path.write_text(report)
    logger.info("Report saved to %s", output_path)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
