"""Test trend filters on 24M momentum signal."""

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


def _run_experiment(filter_type: str, filter_cap: float = 0.3) -> pd.DataFrame:
    """Run 24M momentum experiment with specified trend filter."""
    logger.info("=" * 80)
    if filter_type == "none":
        logger.info("RUNNING: Baseline (no filter)")
    else:
        logger.info("RUNNING: %s filter (cap=%.2f)", filter_type, filter_cap)
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
        trend_filter_type=filter_type,
        trend_filter_risk_on_cap=filter_cap,
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
    """Run trend filter experiment on 24M momentum."""
    logger.info("Starting trend filter experiment")
    logger.info("Baseline: Pure market 24M momentum model")
    
    # Test configurations
    filter_configs = [
        {"type": "none", "cap": 0.3, "name": "Baseline (no filter)"},
        {"type": "200dma", "cap": 0.3, "name": "SPY > 200DMA"},
        {"type": "12m_return", "cap": 0.3, "name": "SPY 12M ret > 0"},
        {"type": "10mma", "cap": 0.3, "name": "SPY > 10M MA"},
    ]
    
    results = []
    
    # Run experiments
    for config in filter_configs:
        df = _run_experiment(config["type"], config["cap"])
        
        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue
        
        overall = df[df["segment"] == "OVERALL"].iloc[0]
        
        # Difficult period metrics
        difficult = _difficult_period_metrics(df)
        
        result = {
            "filter": config["name"],
            "filter_type": config["type"],
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
        "# Trend Filter Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Baseline Model:**",
        "- Pure market model (0.0 macro / 1.0 market)",
        "- 24-month momentum signal",
        "- Corrected no-lookahead implementation",
        "",
        "**Trend Filters Tested:**",
        "",
        "1. **SPY > 200DMA**: Price above 200-day moving average",
        "2. **SPY 12M ret > 0**: Trailing 12-month return is positive",
        "3. **SPY > 10M MA**: Price above 10-month moving average",
        "",
        "**Filter Behavior:**",
        "- When trend filter is ON: Allow momentum signal as usual",
        "- When trend filter is OFF: Cap risk_on at 0.3 (shift toward risk_off/defensive)",
        "",
        "## Walk-Forward Performance by Filter",
        "",
        "| Filter | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|--------|------|--------|-------|-----|----------|",
    ]
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['filter']:>20} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.append("")
    
    # Best by different metrics
    baseline = results_df[results_df["filter_type"] == "none"].iloc[0]
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_maxdd = results_df.loc[results_df["maxdd"].idxmax()]  # Least negative
    best_turnover = results_df.loc[results_df["turnover"].idxmin()]
    
    report_lines.extend([
        "### Top Performers",
        "",
        f"- **Best CAGR**: {best_cagr['filter']} ({best_cagr['cagr']:.2%})",
        f"- **Best Sharpe**: {best_sharpe['filter']} ({best_sharpe['sharpe']:.3f})",
        f"- **Best MaxDD**: {best_maxdd['filter']} ({best_maxdd['maxdd']:.2%})",
        f"- **Lowest Turnover**: {best_turnover['filter']} ({best_turnover['turnover']:.1%})",
        "",
        "## Difficult Period Performance (2021-2022)",
        "",
        "| Filter | CAGR | Sharpe | MaxDD |",
        "|--------|------|--------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['filter']:>20} | {row['difficult_cagr']:>4.2%} | "
                f"{row['difficult_sharpe']:>6.3f} | {row['difficult_maxdd']:>5.2%} |"
            )
    
    report_lines.append("")
    
    # Difficult period improvement analysis
    baseline_difficult_cagr = results_df[results_df["filter_type"] == "none"].iloc[0]["difficult_cagr"]
    baseline_difficult_sharpe = results_df[results_df["filter_type"] == "none"].iloc[0]["difficult_sharpe"]
    
    best_difficult_cagr = results_df.loc[results_df["difficult_cagr"].idxmax()]
    best_difficult_sharpe = results_df.loc[results_df["difficult_sharpe"].idxmax()]
    
    report_lines.extend([
        "### Difficult Period Analysis",
        "",
        f"**Baseline (no filter):** {baseline_difficult_cagr:.2%} CAGR, {baseline_difficult_sharpe:.3f} Sharpe",
        "",
        f"**Best Difficult CAGR**: {best_difficult_cagr['filter']} "
        f"({best_difficult_cagr['difficult_cagr']:.2%}, "
        f"+{best_difficult_cagr['difficult_cagr'] - baseline_difficult_cagr:.2%} vs baseline)",
        "",
        f"**Best Difficult Sharpe**: {best_difficult_sharpe['filter']} "
        f"({best_difficult_sharpe['difficult_sharpe']:.3f}, "
        f"+{best_difficult_sharpe['difficult_sharpe'] - baseline_difficult_sharpe:.3f} vs baseline)",
        "",
        "## Analysis",
        "",
        "### Impact vs Baseline",
        "",
        "| Filter | dCAGR | dSharpe | dMaxDD | dTurnover |",
        "|--------|-------|---------|--------|-----------|",
    ])
    
    for _, row in results_df.iterrows():
        if row["filter_type"] == "none":
            continue
        delta_cagr = row["cagr"] - baseline["cagr"]
        delta_sharpe = row["sharpe"] - baseline["sharpe"]
        delta_maxdd = row["maxdd"] - baseline["maxdd"]
        delta_turnover = row["turnover"] - baseline["turnover"]
        report_lines.append(
            f"| {row['filter']:>20} | {delta_cagr:>5.2%} | {delta_sharpe:>7.3f} | "
            f"{delta_maxdd:>6.2%} | {delta_turnover:>9.1%} |"
        )
    
    report_lines.append("")
    
    # Evaluation
    report_lines.extend([
        "### Filter Effectiveness",
        "",
    ])
    
    # Check if any filter improves both overall performance and difficult periods
    improved_filters = []
    for _, row in results_df.iterrows():
        if row["filter_type"] == "none":
            continue
        
        overall_better = (row["sharpe"] > baseline["sharpe"] * 1.01)  # 1% improvement threshold
        difficult_better = (row["difficult_sharpe"] > baseline_difficult_sharpe * 1.05)  # 5% improvement threshold
        maxdd_better = (row["maxdd"] > baseline["maxdd"])  # Less negative is better
        
        if overall_better or difficult_better or maxdd_better:
            improved_filters.append({
                "filter": row["filter"],
                "overall": overall_better,
                "difficult": difficult_better,
                "maxdd": maxdd_better,
            })
    
    if improved_filters:
        report_lines.append("**Filters that improve performance:**")
        report_lines.append("")
        for f in improved_filters:
            improvements = []
            if f["overall"]:
                improvements.append("overall Sharpe")
            if f["difficult"]:
                improvements.append("difficult period Sharpe")
            if f["maxdd"]:
                improvements.append("max drawdown")
            report_lines.append(f"- {f['filter']}: improves {', '.join(improvements)}")
        report_lines.append("")
    else:
        report_lines.append("**No filters improve performance meaningfully.**")
        report_lines.append("")
    
    # Trade-off analysis
    report_lines.extend([
        "### Return vs Risk Trade-off",
        "",
    ])
    
    filters_with_lower_dd = results_df[results_df["maxdd"] > baseline["maxdd"]]
    if len(filters_with_lower_dd) > 0:
        report_lines.append("**Filters that reduce drawdown:**")
        report_lines.append("")
        for _, row in filters_with_lower_dd.iterrows():
            if row["filter_type"] == "none":
                continue
            dd_improvement = row["maxdd"] - baseline["maxdd"]
            cagr_cost = row["cagr"] - baseline["cagr"]
            sharpe_cost = row["sharpe"] - baseline["sharpe"]
            report_lines.append(
                f"- {row['filter']}: MaxDD {dd_improvement:+.2%}, but CAGR {cagr_cost:+.2%}, Sharpe {sharpe_cost:+.3f}"
            )
        report_lines.append("")
    else:
        report_lines.append("No filters reduce drawdown vs baseline.")
        report_lines.append("")
    
    # Robustness score
    results_df["robustness_score"] = (
        results_df["sharpe"] / results_df["sharpe"].max() * 0.4 +
        (1 - (results_df["maxdd"].abs() / results_df["maxdd"].abs().max())) * 0.3 +
        (1 - results_df["turnover"] / results_df["turnover"].max()) * 0.1 +
        (results_df["difficult_sharpe"] / results_df["difficult_sharpe"].max()) * 0.2
    )
    
    best_robust = results_df.loc[results_df["robustness_score"].idxmax()]
    
    report_lines.extend([
        "### Robustness Score",
        "",
        "Robustness = 0.4 * Sharpe + 0.3 * (1 - |MaxDD|) + 0.1 * (1 - Turnover) + 0.2 * Difficult Sharpe",
        "",
        "| Filter | Score |",
        "|--------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(f"| {row['filter']:>20} | {row['robustness_score']:>5.3f} |")
    
    report_lines.extend([
        "",
        f"**Most Robust**: {best_robust['filter']}",
        "",
        "## Recommendation",
        "",
    ])
    
    # Final recommendation
    baseline_sharpe = baseline["sharpe"]
    best_filter = results_df[results_df["filter_type"] != "none"].loc[results_df[results_df["filter_type"] != "none"]["sharpe"].idxmax()]
    
    if best_filter["sharpe"] > baseline_sharpe * 1.02:  # 2% improvement threshold
        verdict = f"**ADD FILTER: {best_filter['filter']}**"
        explanation = (
            f"The {best_filter['filter']} filter improves Sharpe by "
            f"{best_filter['sharpe'] - baseline_sharpe:+.3f} "
            f"({((best_filter['sharpe'] / baseline_sharpe) - 1) * 100:+.1f}%) vs baseline."
        )
    elif best_difficult_sharpe["filter_type"] != "none" and \
         best_difficult_sharpe["difficult_sharpe"] > baseline_difficult_sharpe * 1.1:  # 10% improvement in difficult periods
        verdict = f"**CONSIDER FILTER: {best_difficult_sharpe['filter']}**"
        explanation = (
            f"The {best_difficult_sharpe['filter']} filter significantly improves difficult period performance "
            f"(Sharpe {best_difficult_sharpe['difficult_sharpe'] - baseline_difficult_sharpe:+.3f}), "
            f"but overall performance improvement is modest."
        )
    else:
        verdict = "**NO FILTER: Keep baseline 24M momentum**"
        explanation = (
            "Trend filters do not materially improve risk-adjusted returns or drawdown control. "
            "The added complexity is not justified by the marginal benefit."
        )
    
    report_lines.append(verdict)
    report_lines.append("")
    report_lines.append(explanation)
    report_lines.append("")
    
    if best_filter["sharpe"] <= baseline_sharpe * 1.02:
        report_lines.extend([
            "### Why Filters Don't Help",
            "",
        ])
        
        avg_filter_sharpe = results_df[results_df["filter_type"] != "none"]["sharpe"].mean()
        avg_filter_cagr = results_df[results_df["filter_type"] != "none"]["cagr"].mean()
        
        if avg_filter_sharpe < baseline_sharpe:
            report_lines.append(f"- Filters reduce Sharpe on average (avg: {avg_filter_sharpe:.3f} vs baseline: {baseline_sharpe:.3f})")
        if avg_filter_cagr < baseline["cagr"]:
            report_lines.append(f"- Filters reduce CAGR on average (avg: {avg_filter_cagr:.2%} vs baseline: {baseline['cagr']:.2%})")
        
        report_lines.append("- 24M momentum already captures long-term trends effectively")
        report_lines.append("- Additional filtering introduces false signals or whipsaws")
        report_lines.append("")
    
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    if best_filter["sharpe"] > baseline_sharpe * 1.02:
        report_lines.append("1. **Fine-tune filter threshold** - test different risk_on caps (0.2, 0.4, 0.5)")
        report_lines.append("2. **Test filter combinations** - combine multiple trend signals")
    else:
        report_lines.append("1. **Test volatility scaling** - scale risk_on by realized vol percentile")
        report_lines.append("2. **Test dual-momentum** - add relative strength component")
        report_lines.append("3. **Test ensemble** - combine 12M + 24M momentum signals")
    
    report_lines.append("4. **Test alternative assets** - expand universe beyond current ETFs")
    report_lines.append("5. **Test regime-conditional momentum** - apply momentum only in trending regimes")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "| Filter | CAGR | Sharpe | MaxDD | Turnover | Difficult Sharpe | Robustness |",
        "|--------|------|--------|-------|----------|------------------|------------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['filter']:>20} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} | {row['difficult_sharpe']:>16.3f} | "
            f"{row['robustness_score']:>10.3f} |"
        )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "TREND_FILTER_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
