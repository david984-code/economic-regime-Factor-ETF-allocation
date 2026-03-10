"""Run baseline vs hybrid macro+market signal experiment and compare results."""

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
from src.features.transforms import sigmoid

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment(name: str, **kwargs) -> pd.DataFrame:
    """Run experiment and return results."""
    logger.info("=" * 80)
    logger.info("RUNNING: %s", name)
    logger.info("=" * 80)
    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        **kwargs
    )
    return df


def _compute_combined_signal_ic(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    macro_weight: float = 0.5,
) -> dict[str, float]:
    """Compute IC for combined signal vs forward returns."""
    # Compute monthly SPY returns
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_returns_monthly = spy_monthly.pct_change().dropna()
    spy_returns_monthly.index = spy_returns_monthly.index.to_period("M")
    
    # Compute 12M momentum
    momentum_12m = spy_monthly.pct_change(12)
    momentum_12m.index = momentum_12m.index.to_period("M")
    
    # Mean reversion signal
    mean_reversion = -momentum_12m
    
    # Align with regime data
    regime_monthly = regime_df.resample("ME").last()
    regime_monthly.index = regime_monthly.index.to_period("M")
    
    # Common index
    common = spy_returns_monthly.index.intersection(regime_monthly.index).intersection(mean_reversion.index)
    
    spy_ret = spy_returns_monthly.loc[common]
    macro_score = regime_monthly.loc[common, "macro_score"]
    mr_signal = mean_reversion.loc[common]
    
    # Normalize mean_reversion to z-score
    mr_z = (mr_signal - mr_signal.mean()) / mr_signal.std()
    
    # Combined score
    combined_score = macro_weight * macro_score + (1 - macro_weight) * mr_z
    
    # Transform to risk_on
    combined_risk_on = sigmoid(combined_score * 0.25)
    
    # Compute forward returns (1M, 3M, 6M)
    fwd_1m = (1 + spy_ret).rolling(1).apply(lambda x: x.prod() - 1, raw=True).shift(-1)
    fwd_3m = (1 + spy_ret).rolling(3).apply(lambda x: x.prod() - 1, raw=True).shift(-3)
    fwd_6m = (1 + spy_ret).rolling(6).apply(lambda x: x.prod() - 1, raw=True).shift(-6)
    
    # Compute ICs
    df = pd.DataFrame({
        "combined_score": combined_score,
        "combined_risk_on": combined_risk_on,
        "macro_score": macro_score,
        "mr_signal": mr_z,
        "fwd_1m": fwd_1m,
        "fwd_3m": fwd_3m,
        "fwd_6m": fwd_6m,
    }).dropna()
    
    if len(df) < 10:
        return {
            "combined_ic_1m": np.nan, "combined_ic_3m": np.nan, "combined_ic_6m": np.nan,
            "macro_ic_1m": np.nan, "macro_ic_3m": np.nan, "macro_ic_6m": np.nan,
            "mr_ic_1m": np.nan, "mr_ic_3m": np.nan, "mr_ic_6m": np.nan,
        }
    
    combined_ic_1m, _ = spearmanr(df["combined_score"], df["fwd_1m"])
    combined_ic_3m, _ = spearmanr(df["combined_score"], df["fwd_3m"])
    combined_ic_6m, _ = spearmanr(df["combined_score"], df["fwd_6m"])
    
    macro_ic_1m, _ = spearmanr(df["macro_score"], df["fwd_1m"])
    macro_ic_3m, _ = spearmanr(df["macro_score"], df["fwd_3m"])
    macro_ic_6m, _ = spearmanr(df["macro_score"], df["fwd_6m"])
    
    mr_ic_1m, _ = spearmanr(df["mr_signal"], df["fwd_1m"])
    mr_ic_3m, _ = spearmanr(df["mr_signal"], df["fwd_3m"])
    mr_ic_6m, _ = spearmanr(df["mr_signal"], df["fwd_6m"])
    
    return {
        "combined_ic_1m": combined_ic_1m,
        "combined_ic_3m": combined_ic_3m,
        "combined_ic_6m": combined_ic_6m,
        "combined_ic_avg": np.mean([combined_ic_1m, combined_ic_3m, combined_ic_6m]),
        "macro_ic_1m": macro_ic_1m,
        "macro_ic_3m": macro_ic_3m,
        "macro_ic_6m": macro_ic_6m,
        "macro_ic_avg": np.mean([macro_ic_1m, macro_ic_3m, macro_ic_6m]),
        "mr_ic_1m": mr_ic_1m,
        "mr_ic_3m": mr_ic_3m,
        "mr_ic_6m": mr_ic_6m,
        "mr_ic_avg": np.mean([mr_ic_1m, mr_ic_3m, mr_ic_6m]),
    }


def main():
    """Run baseline vs hybrid signal experiment and generate comparison report."""
    logger.info("Starting hybrid signal experiment")
    
    # Run baseline (macro_score only)
    baseline = _run_experiment(
        "Baseline (macro_score only)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=False,
    )
    
    # Run hybrid (macro_score + mean-reversion)
    hybrid = _run_experiment(
        "Hybrid (macro_score + mean-reversion)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.5,
    )
    
    if baseline.empty or hybrid.empty:
        logger.error("Experiments failed.")
        sys.exit(1)
    
    # Extract OVERALL results
    baseline_overall = baseline[baseline["segment"] == "OVERALL"].iloc[0]
    hybrid_overall = hybrid[hybrid["segment"] == "OVERALL"].iloc[0]
    
    # Compute IC statistics for combined signal
    logger.info("Computing IC statistics for combined signal...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes
        regime_df = load_regimes()
    
    ic_stats = _compute_combined_signal_ic(prices, regime_df, macro_weight=0.5)
    
    # Build comparison report
    report_lines = [
        "# Hybrid Macro + Market Signal Experiment Report",
        "",
        "## Experiment Setup",
        "",
        "- **Baseline**: Pure macro_score model (risk_on = sigmoid(macro_score * 0.25))",
        "- **Hybrid**: Combined signal model:",
        "  - mean_reversion = -momentum_12m (SPY)",
        "  - combined_score = 0.5 * macro_score + 0.5 * mean_reversion_z",
        "  - risk_on = sigmoid(combined_score * 0.25)",
        "",
        "## Overall Walk-Forward Performance",
        "",
        "| Metric          | Baseline | Hybrid | Difference |",
        "|-----------------|----------|--------|------------|",
        f"| CAGR            | {baseline_overall['Strategy_CAGR']:.2%} | {hybrid_overall['Strategy_CAGR']:.2%} | {hybrid_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']:.2%} |",
        f"| Sharpe Ratio    | {baseline_overall['Strategy_Sharpe']:.3f} | {hybrid_overall['Strategy_Sharpe']:.3f} | {hybrid_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']:.3f} |",
        f"| Max Drawdown    | {baseline_overall['Strategy_MaxDD']:.2%} | {hybrid_overall['Strategy_MaxDD']:.2%} | {hybrid_overall['Strategy_MaxDD'] - baseline_overall['Strategy_MaxDD']:.2%} |",
        f"| Volatility      | {baseline_overall['Strategy_Vol']:.2%} | {hybrid_overall['Strategy_Vol']:.2%} | {hybrid_overall['Strategy_Vol'] - baseline_overall['Strategy_Vol']:.2%} |",
        f"| Turnover        | {baseline_overall.get('Strategy_Turnover', 0.0):.1%} | {hybrid_overall.get('Strategy_Turnover', 0.0):.1%} | {hybrid_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):.1%} |",
        "",
        "## Information Coefficient Analysis",
        "",
        "### Combined Signal IC (Hybrid Model)",
        "",
        "| Horizon | Combined IC | Macro IC | Mean Rev IC |",
        "|---------|-------------|----------|-------------|",
        f"| 1M | {ic_stats['combined_ic_1m']:.4f} | {ic_stats['macro_ic_1m']:.4f} | {ic_stats['mr_ic_1m']:.4f} |",
        f"| 3M | {ic_stats['combined_ic_3m']:.4f} | {ic_stats['macro_ic_3m']:.4f} | {ic_stats['mr_ic_3m']:.4f} |",
        f"| 6M | {ic_stats['combined_ic_6m']:.4f} | {ic_stats['macro_ic_6m']:.4f} | {ic_stats['mr_ic_6m']:.4f} |",
        f"| **Average** | **{ic_stats['combined_ic_avg']:.4f}** | **{ic_stats['macro_ic_avg']:.4f}** | **{ic_stats['mr_ic_avg']:.4f}** |",
        "",
        "**Note**: IC > 0.05 is meaningful, IC > 0.10 is strong.",
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
        "### Hybrid",
        "",
        "| Benchmark    | CAGR     | Sharpe   | MaxDD    |",
        "|--------------|----------|----------|----------|",
        f"| SPY          | {hybrid_overall.get('SPY_CAGR', 0.0):.2%} | {hybrid_overall.get('SPY_Sharpe', 0.0):.3f} | {hybrid_overall.get('SPY_MaxDD', 0.0):.2%} |",
        f"| 60/40        | {hybrid_overall.get('60/40_CAGR', 0.0):.2%} | {hybrid_overall.get('60/40_Sharpe', 0.0):.3f} | {hybrid_overall.get('60/40_MaxDD', 0.0):.2%} |",
        f"| Equal_Weight | {hybrid_overall.get('Equal_Weight_CAGR', 0.0):.2%} | {hybrid_overall.get('Equal_Weight_Sharpe', 0.0):.3f} | {hybrid_overall.get('Equal_Weight_MaxDD', 0.0):.2%} |",
        f"| Risk_On_Off  | {hybrid_overall.get('Risk_On_Off_CAGR', 0.0):.2%} | {hybrid_overall.get('Risk_On_Off_Sharpe', 0.0):.3f} | {hybrid_overall.get('Risk_On_Off_MaxDD', 0.0):.2%} |",
        "",
        "## Analysis",
        "",
    ]
    
    cagr_diff = hybrid_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']
    sharpe_diff = hybrid_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']
    maxdd_diff = hybrid_overall['Strategy_MaxDD'] - baseline_overall['Strategy_MaxDD']
    turnover_diff = hybrid_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0)
    
    report_lines.append("### Performance Impact")
    report_lines.append("")
    
    if cagr_diff > 0:
        report_lines.append(f"- Hybrid increased CAGR by {cagr_diff:.2%}")
    else:
        report_lines.append(f"- Hybrid decreased CAGR by {-cagr_diff:.2%}")
    
    if sharpe_diff > 0:
        report_lines.append(f"- Hybrid increased Sharpe by {sharpe_diff:.3f}")
    else:
        report_lines.append(f"- Hybrid decreased Sharpe by {-sharpe_diff:.3f}")
    
    if maxdd_diff < 0:
        report_lines.append(f"- Hybrid reduced Max Drawdown by {-maxdd_diff:.2%}")
    else:
        report_lines.append(f"- Hybrid increased Max Drawdown by {maxdd_diff:.2%}")
    
    report_lines.append(f"- Hybrid changed turnover by {turnover_diff:.1%}")
    report_lines.append("")
    
    report_lines.append("### Signal Strength")
    report_lines.append("")
    
    ic_improvement = ic_stats['combined_ic_avg'] - ic_stats['macro_ic_avg']
    
    if ic_stats['combined_ic_avg'] > ic_stats['macro_ic_avg'] + 0.02:
        report_lines.append(f"- **Combined signal IC improved by {ic_improvement:.4f}** (meaningful improvement)")
    elif ic_stats['combined_ic_avg'] > ic_stats['macro_ic_avg']:
        report_lines.append(f"- Combined signal IC improved by {ic_improvement:.4f} (marginal improvement)")
    else:
        report_lines.append(f"- Combined signal IC did not improve (change: {ic_improvement:.4f})")
    
    report_lines.append(f"- Macro-only IC: {ic_stats['macro_ic_avg']:.4f}")
    report_lines.append(f"- Mean-reversion IC: {ic_stats['mr_ic_avg']:.4f}")
    report_lines.append(f"- Combined IC: {ic_stats['combined_ic_avg']:.4f}")
    report_lines.append("")
    
    # Check if mean reversion signal is contributing
    if abs(ic_stats['mr_ic_avg']) > abs(ic_stats['macro_ic_avg']) * 1.5:
        report_lines.append("- **Mean-reversion signal is substantially stronger than macro_score**")
    elif abs(ic_stats['mr_ic_avg']) < abs(ic_stats['macro_ic_avg']) * 0.5:
        report_lines.append("- **Mean-reversion signal is weaker than macro_score**")
    else:
        report_lines.append("- **Mean-reversion and macro_score have similar strength**")
    
    report_lines.append("")
    
    report_lines.append("## Conclusion")
    report_lines.append("")
    
    # Final verdict
    if cagr_diff > 0.01 and sharpe_diff > 0.05:
        verdict = "ACCEPT: Hybrid signal improves both CAGR and Sharpe."
    elif ic_stats['combined_ic_avg'] > ic_stats['macro_ic_avg'] + 0.02 and cagr_diff > 0:
        verdict = "ACCEPT: Hybrid signal improves IC and CAGR."
    elif cagr_diff > 0 and sharpe_diff > 0:
        verdict = "MARGINAL: Small improvements in CAGR and Sharpe."
    elif abs(cagr_diff) < 0.005 and abs(sharpe_diff) < 0.02:
        verdict = "NEUTRAL: Hybrid signal has minimal impact on performance."
    else:
        verdict = "REJECT: Hybrid signal does not improve performance."
    
    report_lines.append(f"**{verdict}**")
    report_lines.append("")
    
    # Recommendation
    if "ACCEPT" in verdict:
        report_lines.append("**Recommendation**: Adopt hybrid signal model for improved robustness.")
        report_lines.append("")
        report_lines.append("Next steps:")
        report_lines.append("- Test alternative weight combinations (e.g., 0.3 macro / 0.7 market)")
        report_lines.append("- Consider adding volatility regime or trend strength to the hybrid")
    elif "MARGINAL" in verdict or "NEUTRAL" in verdict:
        report_lines.append("**Recommendation**: Test alternative market signals before deciding.")
        report_lines.append("")
        report_lines.append("Alternatives to consider:")
        report_lines.append("- Trend strength (distance from moving average)")
        report_lines.append("- Realized volatility regime")
        report_lines.append("- Different weight combinations (e.g., 0.3/0.7 instead of 0.5/0.5)")
    else:
        report_lines.append("**Recommendation**: The mean-reversion signal may be too noisy or poorly timed.")
        report_lines.append("")
        report_lines.append("Alternative approaches:")
        report_lines.append("- Test different lookback periods (6M, 18M) for momentum")
        report_lines.append("- Test trend-following (positive momentum) instead of mean-reversion")
        report_lines.append("- Consider pure market-based model without macro signals")
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "HYBRID_SIGNAL_EXPERIMENT.md"
    output_path.write_text(report)
    logger.info("Report saved to %s", output_path)
    
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
