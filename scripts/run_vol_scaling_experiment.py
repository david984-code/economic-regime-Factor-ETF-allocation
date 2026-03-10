"""Test volatility scaling on 24M momentum signal."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment(vol_scaling_method: str) -> pd.DataFrame:
    """Run 24M momentum experiment with specified vol scaling."""
    logger.info("=" * 80)
    if vol_scaling_method == "none":
        logger.info("RUNNING: Baseline (no vol scaling)")
    else:
        logger.info("RUNNING: %s vol scaling", vol_scaling_method)
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
        trend_filter_type="none",  # No trend filter
        vol_scaling_method=vol_scaling_method,
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
        
        if (test_start.year >= 2021 and test_start.year <= 2022) or \
           (test_end.year >= 2021 and test_end.year <= 2022):
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
    """Run volatility scaling experiment on 24M momentum."""
    logger.info("Starting volatility scaling experiment")
    logger.info("Baseline: Pure market 24M momentum model (no vol scaling)")
    
    # Test configurations
    vol_methods = [
        {"method": "none", "name": "Baseline (no scaling)"},
        {"method": "realized_20d", "name": "20-day realized vol"},
        {"method": "realized_63d", "name": "63-day realized vol"},
        {"method": "percentile", "name": "Vol percentile"},
    ]
    
    results = []
    
    # Run experiments
    for config in vol_methods:
        df = _run_experiment(config["method"])
        
        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue
        
        overall = df[df["segment"] == "OVERALL"].iloc[0]
        
        # Difficult period metrics
        difficult = _difficult_period_metrics(df)
        
        result = {
            "scaling": config["name"],
            "method": config["method"],
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
    
    if not results:
        logger.error("All experiments failed.")
        sys.exit(1)
    
    results_df = pd.DataFrame(results)
    
    # Build report
    report_lines = [
        "# Volatility Scaling Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Baseline Model:**",
        "- Pure market model (0.0 macro / 1.0 market)",
        "- 24-month momentum signal",
        "- No trend filter",
        "- Corrected no-lookahead implementation",
        "",
        "**Volatility Scaling Methods Tested:**",
        "",
        "1. **20-day realized vol**: Scale by trailing 20-day realized vol vs long-run average",
        "   - High vol -> reduce risk_on (scale down)",
        "   - Low vol -> maintain risk_on (scale up)",
        "   - Scaling capped between 0.5x and 1.5x",
        "",
        "2. **63-day realized vol**: Scale by trailing 63-day realized vol vs long-run average",
        "   - Same logic as 20-day but with longer lookback",
        "   - Smoother, less reactive to short-term vol spikes",
        "",
        "3. **Vol percentile**: Scale by volatility percentile rank",
        "   - Reduce exposure when vol is in top quintile (80th+ percentile)",
        "   - Linear scale from 1.0x at 80th to 0.5x at 100th percentile",
        "   - Below 80th percentile: no scaling (1.0x)",
        "",
        "## Walk-Forward Performance by Scaling Method",
        "",
        "| Scaling Method | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|----------------|------|--------|-------|-----|----------|",
    ]
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['scaling']:>23} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.append("")
    
    # Best by different metrics
    baseline = results_df[results_df["method"] == "none"].iloc[0]
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_maxdd = results_df.loc[results_df["maxdd"].idxmax()]  # Least negative
    best_vol = results_df.loc[results_df["vol"].idxmin()]
    best_turnover = results_df.loc[results_df["turnover"].idxmin()]
    
    report_lines.extend([
        "### Top Performers",
        "",
        f"- **Best CAGR**: {best_cagr['scaling']} ({best_cagr['cagr']:.2%})",
        f"- **Best Sharpe**: {best_sharpe['scaling']} ({best_sharpe['sharpe']:.3f})",
        f"- **Best MaxDD**: {best_maxdd['scaling']} ({best_maxdd['maxdd']:.2%})",
        f"- **Lowest Vol**: {best_vol['scaling']} ({best_vol['vol']:.2%})",
        f"- **Lowest Turnover**: {best_turnover['scaling']} ({best_turnover['turnover']:.1%})",
        "",
        "## Difficult Period Performance (2021-2022)",
        "",
        "| Scaling Method | CAGR | Sharpe | MaxDD |",
        "|----------------|------|--------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['scaling']:>23} | {row['difficult_cagr']:>4.2%} | "
                f"{row['difficult_sharpe']:>6.3f} | {row['difficult_maxdd']:>5.2%} |"
            )
    
    report_lines.append("")
    
    # Difficult period improvement analysis
    baseline_difficult_cagr = baseline["difficult_cagr"]
    baseline_difficult_sharpe = baseline["difficult_sharpe"]
    baseline_difficult_maxdd = baseline["difficult_maxdd"]
    
    best_difficult_cagr = results_df.loc[results_df["difficult_cagr"].idxmax()]
    best_difficult_sharpe = results_df.loc[results_df["difficult_sharpe"].idxmax()]
    best_difficult_maxdd = results_df.loc[results_df["difficult_maxdd"].idxmax()]
    
    report_lines.extend([
        "### Difficult Period Analysis",
        "",
        f"**Baseline:** {baseline_difficult_cagr:.2%} CAGR, {baseline_difficult_sharpe:.3f} Sharpe, {baseline_difficult_maxdd:.2%} MaxDD",
        "",
        f"**Best Difficult CAGR**: {best_difficult_cagr['scaling']} "
        f"({best_difficult_cagr['difficult_cagr']:.2%}, "
        f"+{best_difficult_cagr['difficult_cagr'] - baseline_difficult_cagr:.2%} vs baseline)",
        "",
        f"**Best Difficult Sharpe**: {best_difficult_sharpe['scaling']} "
        f"({best_difficult_sharpe['difficult_sharpe']:.3f}, "
        f"+{best_difficult_sharpe['difficult_sharpe'] - baseline_difficult_sharpe:.3f} vs baseline)",
        "",
        f"**Best Difficult MaxDD**: {best_difficult_maxdd['scaling']} "
        f"({best_difficult_maxdd['difficult_maxdd']:.2%}, "
        f"+{best_difficult_maxdd['difficult_maxdd'] - baseline_difficult_maxdd:.2%} vs baseline)",
        "",
        "## Analysis",
        "",
        "### Impact vs Baseline",
        "",
        "| Scaling Method | dCAGR | dSharpe | dMaxDD | dVol | dTurnover |",
        "|----------------|-------|---------|--------|------|-----------|",
    ])
    
    for _, row in results_df.iterrows():
        if row["method"] == "none":
            continue
        delta_cagr = row["cagr"] - baseline["cagr"]
        delta_sharpe = row["sharpe"] - baseline["sharpe"]
        delta_maxdd = row["maxdd"] - baseline["maxdd"]
        delta_vol = row["vol"] - baseline["vol"]
        delta_turnover = row["turnover"] - baseline["turnover"]
        report_lines.append(
            f"| {row['scaling']:>23} | {delta_cagr:>5.2%} | {delta_sharpe:>7.3f} | "
            f"{delta_maxdd:>6.2%} | {delta_vol:>4.2%} | {delta_turnover:>9.1%} |"
        )
    
    report_lines.append("")
    
    # Vol reduction analysis
    report_lines.extend([
        "### Volatility Reduction",
        "",
    ])
    
    vol_reduced = results_df[results_df["vol"] < baseline["vol"]]
    if len(vol_reduced) > 0:
        report_lines.append("**Methods that reduce volatility:**")
        report_lines.append("")
        for _, row in vol_reduced.iterrows():
            if row["method"] == "none":
                continue
            vol_reduction = baseline["vol"] - row["vol"]
            vol_reduction_pct = (vol_reduction / baseline["vol"]) * 100
            cagr_impact = row["cagr"] - baseline["cagr"]
            sharpe_impact = row["sharpe"] - baseline["sharpe"]
            report_lines.append(
                f"- {row['scaling']}: Vol -{vol_reduction:.2%} (-{vol_reduction_pct:.1f}%), "
                f"CAGR {cagr_impact:+.2%}, Sharpe {sharpe_impact:+.3f}"
            )
        report_lines.append("")
    else:
        report_lines.append("No methods reduce volatility vs baseline.")
        report_lines.append("")
    
    # Drawdown control analysis
    report_lines.extend([
        "### Drawdown Control",
        "",
    ])
    
    dd_improved = results_df[results_df["maxdd"] > baseline["maxdd"]]
    if len(dd_improved) > 0:
        report_lines.append("**Methods that improve MaxDD:**")
        report_lines.append("")
        for _, row in dd_improved.iterrows():
            if row["method"] == "none":
                continue
            dd_improvement = row["maxdd"] - baseline["maxdd"]
            difficult_dd_improvement = row["difficult_maxdd"] - baseline_difficult_maxdd
            report_lines.append(
                f"- {row['scaling']}: Overall MaxDD {dd_improvement:+.2%}, "
                f"Difficult MaxDD {difficult_dd_improvement:+.2%}"
            )
        report_lines.append("")
    else:
        report_lines.append("No methods improve drawdown vs baseline.")
        report_lines.append("")
    
    # Robustness score
    results_df["robustness_score"] = (
        results_df["sharpe"] / results_df["sharpe"].max() * 0.3 +
        (1 - (results_df["maxdd"].abs() / results_df["maxdd"].abs().max())) * 0.2 +
        (1 - results_df["vol"] / results_df["vol"].max()) * 0.2 +
        (1 - results_df["turnover"] / results_df["turnover"].max()) * 0.1 +
        (results_df["difficult_sharpe"] / results_df["difficult_sharpe"].max()) * 0.2
    )
    
    best_robust = results_df.loc[results_df["robustness_score"].idxmax()]
    
    report_lines.extend([
        "### Robustness Score",
        "",
        "Robustness = 0.3*Sharpe + 0.2*(1-|MaxDD|) + 0.2*(1-Vol) + 0.1*(1-Turnover) + 0.2*Difficult_Sharpe",
        "",
        "| Scaling Method | Score |",
        "|----------------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(f"| {row['scaling']:>23} | {row['robustness_score']:>5.3f} |")
    
    report_lines.extend([
        "",
        f"**Most Robust**: {best_robust['scaling']}",
        "",
        "## Recommendation",
        "",
    ])
    
    # Final recommendation
    baseline_sharpe = baseline["sharpe"]
    best_scaling = results_df[results_df["method"] != "none"].loc[
        results_df[results_df["method"] != "none"]["sharpe"].idxmax()
    ]
    
    # Check multiple criteria
    sharpe_improvement = best_scaling["sharpe"] - baseline_sharpe
    maxdd_improvement = best_scaling["maxdd"] - baseline["maxdd"]
    vol_improvement = baseline["vol"] - best_scaling["vol"]
    difficult_improvement = best_scaling["difficult_sharpe"] - baseline_difficult_sharpe
    
    if sharpe_improvement > 0.01 and (maxdd_improvement > 0 or vol_improvement > 0.001):
        verdict = f"**ADD VOLATILITY SCALING: {best_scaling['scaling']}**"
        explanation = (
            f"The {best_scaling['scaling']} method improves risk-adjusted returns:\n"
            f"- Sharpe: {sharpe_improvement:+.3f}\n"
            f"- MaxDD: {maxdd_improvement:+.2%}\n"
            f"- Vol: {-vol_improvement:+.2%}\n"
            f"- Difficult Sharpe: {difficult_improvement:+.3f}"
        )
    elif difficult_improvement > 0.05:  # 0.05 is significant for Sharpe
        verdict = f"**CONSIDER SCALING: {best_scaling['scaling']}**"
        explanation = (
            f"The {best_scaling['scaling']} method significantly improves difficult period performance "
            f"(Sharpe {difficult_improvement:+.3f}), but overall improvement is modest."
        )
    else:
        verdict = "**NO VOL SCALING: Keep baseline 24M momentum**"
        explanation = (
            "Volatility scaling does not materially improve risk-adjusted returns, drawdown control, "
            "or difficult period performance. The baseline 24M momentum without vol scaling is optimal."
        )
    
    report_lines.append(verdict)
    report_lines.append("")
    report_lines.append(explanation)
    report_lines.append("")
    
    if sharpe_improvement <= 0.01:
        report_lines.extend([
            "### Why Vol Scaling Doesn't Help",
            "",
        ])
        
        avg_scaled_sharpe = results_df[results_df["method"] != "none"]["sharpe"].mean()
        avg_scaled_cagr = results_df[results_df["method"] != "none"]["cagr"].mean()
        avg_scaled_vol = results_df[results_df["method"] != "none"]["vol"].mean()
        
        if avg_scaled_sharpe < baseline_sharpe:
            report_lines.append(f"- Scaling reduces Sharpe on average (avg: {avg_scaled_sharpe:.3f} vs baseline: {baseline_sharpe:.3f})")
        if avg_scaled_cagr < baseline["cagr"]:
            report_lines.append(f"- Scaling reduces CAGR on average (avg: {avg_scaled_cagr:.2%} vs baseline: {baseline['cagr']:.2%})")
        if avg_scaled_vol >= baseline["vol"]:
            report_lines.append(f"- Scaling does not reduce volatility (avg: {avg_scaled_vol:.2%} vs baseline: {baseline['vol']:.2%})")
        
        report_lines.append("- Fixed risk_on blending already provides vol-responsive behavior via asset allocation")
        report_lines.append("- Additional vol scaling may be redundant or counterproductive")
        report_lines.append("")
    
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    if sharpe_improvement > 0.01:
        report_lines.append("1. **Fine-tune vol thresholds** - test different percentile cutoffs (70th, 85th, 90th)")
        report_lines.append("2. **Test vol scaling with trend filter** - combine vol + trend signals")
    else:
        report_lines.append("1. **Test dual-momentum** - add relative strength to absolute momentum")
        report_lines.append("2. **Test ensemble signals** - combine 12M + 24M momentum")
        report_lines.append("3. **Test alternative assets** - expand universe beyond current ETFs")
    
    report_lines.append("4. **Test regime-conditional momentum** - apply momentum only in specific macro regimes")
    report_lines.append("5. **Optimize portfolio weights** - revisit Sortino optimization or test alternatives")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "| Scaling Method | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe | Robustness |",
        "|----------------|------|--------|-------|-----|----------|------------------|------------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['scaling']:>23} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} | "
            f"{row['difficult_sharpe']:>16.3f} | {row['robustness_score']:>10.3f} |"
        )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "VOL_SCALING_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    logger.info("\nExperiment complete. Report generated.")


if __name__ == "__main__":
    main()
