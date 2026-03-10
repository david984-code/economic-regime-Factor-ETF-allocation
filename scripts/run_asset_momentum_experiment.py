"""Test asset-specific momentum vs market-level momentum."""

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


def _run_experiment(portfolio_method: str) -> pd.DataFrame:
    """Run 24M momentum experiment with specified asset selection method."""
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
    """Run asset-specific momentum experiment."""
    logger.info("Starting asset-specific momentum experiment")
    logger.info("Fixed signal: 24M momentum")
    logger.info("Testing: asset selection within risk-on sleeve")
    
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
        df = _run_experiment(config["method"])
        
        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue
        
        overall = df[df["segment"] == "OVERALL"].iloc[0]
        
        # Difficult period metrics
        difficult = _difficult_period_metrics(df)
        
        result = {
            "method_name": config["name"],
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
        "# Asset-Specific Momentum Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Fixed Setup:**",
        "- Pure market model (0.0 macro / 1.0 market)",
        "- 24-month momentum signal (SPY-based for risk_on calculation)",
        "- Corrected equal-weight sleeve construction",
        "- No trend filter",
        "- No volatility scaling",
        "- Corrected no-lookahead implementation",
        "",
        "**Risk-off Sleeve (unchanged across all variants):**",
        "- Equal weight across [IEF, TLT, GLD]",
        "",
        "**Risk-on Sleeve Variants Tested:**",
        "",
        "1. **Baseline (all risk-on assets)**: Equal weight across all 7 risk-on assets",
        "   - Assets: SPY, MTUM, VLUE, QUAL, USMV, IJR, VIG",
        "   - No filtering",
        "",
        "2. **Positive momentum only**: Equal weight among assets with positive 24M momentum",
        "   - Compute 24M momentum for each risk-on asset",
        "   - Include only assets with momentum > 0",
        "   - Fallback: top 3 if none qualify",
        "",
        "3. **Top 3 by momentum**: Equal weight among top 3 assets by 24M momentum",
        "   - Rank all risk-on assets by 24M momentum",
        "   - Select top 3",
        "   - More concentrated, more responsive",
        "",
        "4. **Top 5 by momentum**: Equal weight among top 5 assets by 24M momentum",
        "   - Rank all risk-on assets by 24M momentum",
        "   - Select top 5",
        "   - Balance between diversification and selectivity",
        "",
        "## Walk-Forward Performance by Asset Selection Method",
        "",
        "| Method | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|--------|------|--------|-------|-----|----------|",
    ]
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['method_name']:>30} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.append("")
    
    # Best by different metrics
    baseline = results_df[results_df["method"] == "equal_weight"].iloc[0]
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_maxdd = results_df.loc[results_df["maxdd"].idxmax()]
    best_turnover = results_df.loc[results_df["turnover"].idxmin()]
    
    report_lines.extend([
        "### Top Performers",
        "",
        f"- **Best CAGR**: {best_cagr['method_name']} ({best_cagr['cagr']:.2%})",
        f"- **Best Sharpe**: {best_sharpe['method_name']} ({best_sharpe['sharpe']:.3f})",
        f"- **Best MaxDD**: {best_maxdd['method_name']} ({best_maxdd['maxdd']:.2%})",
        f"- **Lowest Turnover**: {best_turnover['method_name']} ({best_turnover['turnover']:.1%})",
        "",
        "## Difficult Period Performance (2021-2022)",
        "",
        "| Method | CAGR | Sharpe | MaxDD |",
        "|--------|------|--------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['method_name']:>30} | {row['difficult_cagr']:>4.2%} | "
                f"{row['difficult_sharpe']:>6.3f} | {row['difficult_maxdd']:>5.2%} |"
            )
    
    report_lines.append("")
    
    # Difficult period analysis
    baseline_difficult_cagr = baseline["difficult_cagr"]
    baseline_difficult_sharpe = baseline["difficult_sharpe"]
    
    best_difficult_cagr = results_df.loc[results_df["difficult_cagr"].idxmax()]
    best_difficult_sharpe = results_df.loc[results_df["difficult_sharpe"].idxmax()]
    
    report_lines.extend([
        "### Difficult Period Analysis",
        "",
        f"**Baseline:** {baseline_difficult_cagr:.2%} CAGR, {baseline_difficult_sharpe:.3f} Sharpe",
        "",
        f"**Best Difficult CAGR**: {best_difficult_cagr['method_name']} "
        f"({best_difficult_cagr['difficult_cagr']:.2%}, "
        f"+{best_difficult_cagr['difficult_cagr'] - baseline_difficult_cagr:.2%} vs baseline)",
        "",
        f"**Best Difficult Sharpe**: {best_difficult_sharpe['method_name']} "
        f"({best_difficult_sharpe['difficult_sharpe']:.3f}, "
        f"+{best_difficult_sharpe['difficult_sharpe'] - baseline_difficult_sharpe:.3f} vs baseline)",
        "",
        "## Analysis",
        "",
        "### Impact vs Baseline",
        "",
        "| Method | dCAGR | dSharpe | dMaxDD | dVol | dTurnover |",
        "|--------|-------|---------|--------|------|-----------|",
    ])
    
    for _, row in results_df.iterrows():
        if row["method"] == "equal_weight":
            continue
        delta_cagr = row["cagr"] - baseline["cagr"]
        delta_sharpe = row["sharpe"] - baseline["sharpe"]
        delta_maxdd = row["maxdd"] - baseline["maxdd"]
        delta_vol = row["vol"] - baseline["vol"]
        delta_turnover = row["turnover"] - baseline["turnover"]
        report_lines.append(
            f"| {row['method_name']:>30} | {delta_cagr:>5.2%} | {delta_sharpe:>7.3f} | "
            f"{delta_maxdd:>6.2%} | {delta_vol:>4.2%} | {delta_turnover:>9.1%} |"
        )
    
    report_lines.append("")
    
    # Concentration vs diversification
    report_lines.extend([
        "### Concentration vs Diversification",
        "",
    ])
    
    report_lines.append("**Baseline (all 7 assets):**")
    report_lines.append(f"- Maximum diversification within risk-on sleeve")
    report_lines.append(f"- Sharpe: {baseline['sharpe']:.3f}")
    report_lines.append(f"- Turnover: {baseline['turnover']:.1%}")
    report_lines.append("")
    
    for _, row in results_df.iterrows():
        if row["method"] == "equal_weight":
            continue
        
        if "top3" in row["method"]:
            n_assets = 3
        elif "top5" in row["method"]:
            n_assets = 5
        else:
            n_assets = "variable"
        
        report_lines.append(f"**{row['method_name']} (~{n_assets} assets):**")
        report_lines.append(f"- Sharpe: {row['sharpe']:.3f} ({row['sharpe'] - baseline['sharpe']:+.3f} vs baseline)")
        report_lines.append(f"- Turnover: {row['turnover']:.1%} ({row['turnover'] - baseline['turnover']:+.1%} vs baseline)")
        report_lines.append("")
    
    # Robustness score
    results_df["robustness_score"] = (
        results_df["sharpe"] / results_df["sharpe"].max() * 0.4 +
        (1 - (results_df["maxdd"].abs() / results_df["maxdd"].abs().max())) * 0.2 +
        (1 - results_df["turnover"] / results_df["turnover"].max()) * 0.2 +
        (results_df["difficult_sharpe"] / results_df["difficult_sharpe"].max()) * 0.2
    )
    
    best_robust = results_df.loc[results_df["robustness_score"].idxmax()]
    
    report_lines.extend([
        "### Robustness Score",
        "",
        "Robustness = 0.4*Sharpe + 0.2*(1-|MaxDD|) + 0.2*(1-Turnover) + 0.2*Difficult_Sharpe",
        "",
        "| Method | Score |",
        "|--------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(f"| {row['method_name']:>30} | {row['robustness_score']:>5.3f} |")
    
    report_lines.extend([
        "",
        f"**Most Robust**: {best_robust['method_name']}",
        "",
        "## Recommendation",
        "",
    ])
    
    # Final recommendation
    baseline_sharpe = baseline["sharpe"]
    best_method = results_df.loc[results_df["sharpe"].idxmax()]
    
    sharpe_improvement = best_method["sharpe"] - baseline_sharpe
    turnover_diff = best_method["turnover"] - baseline["turnover"]
    
    if best_method["method"] == "equal_weight":
        verdict = "**KEEP BASELINE: Market-level momentum with all assets**"
        explanation = (
            "Asset-specific momentum filtering does not improve risk-adjusted returns.\n"
            "The baseline approach (SPY momentum + equal weight all risk-on assets) is optimal."
        )
    elif sharpe_improvement > 0.03:
        verdict = f"**ADOPT ASSET-SPECIFIC MOMENTUM: {best_method['method_name']}**"
        explanation = (
            f"The {best_method['method_name']} method materially outperforms the baseline:\n"
            f"- Sharpe: {sharpe_improvement:+.3f}\n"
            f"- CAGR: {best_method['cagr'] - baseline['cagr']:+.2%}\n"
            f"- Turnover: {turnover_diff:+.1%}\n"
            f"- Benefit justifies the added complexity"
        )
    else:
        verdict = "**KEEP BASELINE: Improvements are marginal**"
        explanation = (
            f"Best alternative ({best_method['method_name']}) improves Sharpe by only {sharpe_improvement:+.3f}.\n"
            "The improvement does not justify the added complexity of asset-level filtering.\n"
            "The baseline (market-level momentum + equal weight) is simpler and nearly as good."
        )
    
    report_lines.append(verdict)
    report_lines.append("")
    report_lines.append(explanation)
    report_lines.append("")
    
    if best_method["method"] == "equal_weight":
        report_lines.extend([
            "### Why Asset-Specific Momentum Doesn't Help",
            "",
        ])
        
        avg_filtered_sharpe = results_df[results_df["method"] != "equal_weight"]["sharpe"].mean()
        avg_filtered_turnover = results_df[results_df["method"] != "equal_weight"]["turnover"].mean()
        
        if avg_filtered_sharpe < baseline_sharpe:
            report_lines.append(f"- Asset filtering reduces Sharpe on average (avg: {avg_filtered_sharpe:.3f} vs baseline: {baseline_sharpe:.3f})")
        if avg_filtered_turnover > baseline["turnover"]:
            report_lines.append(f"- Asset filtering increases turnover (avg: {avg_filtered_turnover:.1%} vs baseline: {baseline['turnover']:.1%})")
        
        report_lines.append("- Maximum diversification (7 assets) is better than concentration")
        report_lines.append("- SPY momentum already captures market-level trends effectively")
        report_lines.append("- Individual asset momentum adds noise without signal improvement")
        report_lines.append("")
    
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    if best_method["method"] != "equal_weight":
        report_lines.append("1. **Fine-tune filtering thresholds** - test different momentum cutoffs")
        report_lines.append("2. **Test dynamic N** - vary number of assets based on signal strength")
    else:
        report_lines.append("1. **Expand asset universe** - test sector ETFs, international, alternatives")
        report_lines.append("2. **Test regime-conditional universes** - vary risk-on sleeve by macro regime")
        report_lines.append("3. **Test concentration limits** - cap max position (e.g., 10-20% per asset)")
    
    report_lines.append("4. **Test alternative signal construction** - cross-sectional momentum, relative strength")
    report_lines.append("5. **Test rebalance frequency** - quarterly vs monthly")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "| Method | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe | Robustness |",
        "|--------|------|--------|-------|-----|----------|------------------|------------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['method_name']:>30} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} | "
            f"{row['difficult_sharpe']:>16.3f} | {row['robustness_score']:>10.3f} |"
        )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "ASSET_MOMENTUM_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    logger.info("\nExperiment complete. Report generated.")


if __name__ == "__main__":
    main()
