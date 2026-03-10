"""Test portfolio construction methods on 24M momentum signal."""

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
    """Run 24M momentum experiment with specified portfolio construction."""
    logger.info("=" * 80)
    logger.info("RUNNING: %s portfolio construction", portfolio_method)
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
        vol_scaling_method="none",  # No vol scaling
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
    """Run portfolio construction experiment on 24M momentum."""
    logger.info("Starting portfolio construction experiment")
    logger.info("Fixed setup: Pure market 24M momentum (testing portfolio construction only)")
    
    # Test configurations
    methods = [
        {"method": "optimizer", "name": "Optimizer (Sortino)"},
        {"method": "equal_weight", "name": "Equal Weight"},
        {"method": "risk_parity", "name": "Risk Parity"},
        {"method": "heuristic", "name": "Heuristic (60/40)"},
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
        "# Portfolio Construction Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Fixed Signal:**",
        "- Pure market model (0.0 macro / 1.0 market)",
        "- 24-month momentum signal",
        "- No trend filter",
        "- No volatility scaling",
        "- Corrected no-lookahead implementation",
        "",
        "**Portfolio Construction Methods Tested:**",
        "",
        "1. **Optimizer (Sortino)**: Current baseline using Sortino ratio optimization per regime",
        "   - Optimizes asset weights to maximize Sortino ratio",
        "   - Different allocations for each economic regime",
        "   - Blends risk-on and risk-off allocations based on risk_on score",
        "",
        "2. **Equal Weight**: Simple 1/N allocation across all assets",
        "   - Equal weight for risk-on sleeve",
        "   - Equal weight for risk-off sleeve",
        "   - Simplest possible construction",
        "",
        "3. **Risk Parity**: Inverse volatility weighted allocation",
        "   - Weight assets by inverse of trailing 63-day volatility",
        "   - Lower vol assets get higher weight",
        "   - Volatility-balanced portfolio",
        "",
        "4. **Heuristic (60/40)**: Fixed rules-based allocation",
        "   - Risk-on: 60% equity + factors, 30% bonds, 10% cash",
        "   - Risk-off: 20% equity, 70% bonds, 10% cash",
        "   - Simple, interpretable rules",
        "",
        "## Walk-Forward Performance by Construction Method",
        "",
        "| Method | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|--------|------|--------|-------|-----|----------|",
    ]
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['method_name']:>20} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.append("")
    
    # Best by different metrics
    baseline = results_df[results_df["method"] == "optimizer"].iloc[0]
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_maxdd = results_df.loc[results_df["maxdd"].idxmax()]  # Least negative
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
                f"| {row['method_name']:>20} | {row['difficult_cagr']:>4.2%} | "
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
        f"**Baseline (Optimizer):** {baseline_difficult_cagr:.2%} CAGR, {baseline_difficult_sharpe:.3f} Sharpe",
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
        "### Impact vs Optimizer Baseline",
        "",
        "| Method | dCAGR | dSharpe | dMaxDD | dVol | dTurnover |",
        "|--------|-------|---------|--------|------|-----------|",
    ])
    
    for _, row in results_df.iterrows():
        if row["method"] == "optimizer":
            continue
        delta_cagr = row["cagr"] - baseline["cagr"]
        delta_sharpe = row["sharpe"] - baseline["sharpe"]
        delta_maxdd = row["maxdd"] - baseline["maxdd"]
        delta_vol = row["vol"] - baseline["vol"]
        delta_turnover = row["turnover"] - baseline["turnover"]
        report_lines.append(
            f"| {row['method_name']:>20} | {delta_cagr:>5.2%} | {delta_sharpe:>7.3f} | "
            f"{delta_maxdd:>6.2%} | {delta_vol:>4.2%} | {delta_turnover:>9.1%} |"
        )
    
    report_lines.append("")
    
    # Simplicity vs performance tradeoff
    report_lines.extend([
        "### Simplicity vs Performance",
        "",
    ])
    
    # Rank by simplicity
    simplicity_ranking = {
        "equal_weight": 1,  # Simplest
        "heuristic": 2,
        "risk_parity": 3,
        "optimizer": 4,  # Most complex
    }
    
    for _, row in results_df.iterrows():
        simplicity = simplicity_ranking[row["method"]]
        sharpe_vs_baseline = row["sharpe"] - baseline["sharpe"]
        cagr_vs_baseline = row["cagr"] - baseline["cagr"]
        turnover_vs_baseline = row["turnover"] - baseline["turnover"]
        
        if row["method"] == "optimizer":
            report_lines.append(f"**{row['method_name']}** (Complexity: {simplicity}/4):")
            report_lines.append("- Baseline (most complex)")
        else:
            report_lines.append(f"**{row['method_name']}** (Complexity: {simplicity}/4):")
            report_lines.append(f"- Sharpe: {sharpe_vs_baseline:+.3f} vs optimizer")
            report_lines.append(f"- CAGR: {cagr_vs_baseline:+.2%} vs optimizer")
            report_lines.append(f"- Turnover: {turnover_vs_baseline:+.1%} vs optimizer")
        report_lines.append("")
    
    # Value assessment
    report_lines.extend([
        "### Does the Optimizer Add Value?",
        "",
    ])
    
    # Check if any simpler method beats optimizer on Sharpe
    simpler_better = results_df[
        (results_df["method"] != "optimizer") & 
        (results_df["sharpe"] > baseline["sharpe"])
    ]
    
    if len(simpler_better) > 0:
        report_lines.append("**NO - Simpler methods outperform:**")
        report_lines.append("")
        for _, row in simpler_better.iterrows():
            sharpe_improvement = row["sharpe"] - baseline["sharpe"]
            report_lines.append(
                f"- {row['method_name']}: +{sharpe_improvement:.3f} Sharpe, "
                f"simpler construction"
            )
        report_lines.append("")
    else:
        report_lines.append("**YES - Optimizer provides best risk-adjusted returns:**")
        report_lines.append("")
        avg_simpler_sharpe = results_df[results_df["method"] != "optimizer"]["sharpe"].mean()
        sharpe_advantage = baseline["sharpe"] - avg_simpler_sharpe
        report_lines.append(f"- Optimizer Sharpe: {baseline['sharpe']:.3f}")
        report_lines.append(f"- Average simpler Sharpe: {avg_simpler_sharpe:.3f}")
        report_lines.append(f"- Advantage: +{sharpe_advantage:.3f}")
        report_lines.append("")
    
    # Turnover comparison
    report_lines.extend([
        "### Turnover Analysis",
        "",
    ])
    
    low_turnover = results_df[results_df["turnover"] < baseline["turnover"] * 0.9]
    if len(low_turnover) > 0:
        report_lines.append("**Methods with materially lower turnover:**")
        report_lines.append("")
        for _, row in low_turnover.iterrows():
            turnover_reduction = baseline["turnover"] - row["turnover"]
            turnover_reduction_pct = (turnover_reduction / baseline["turnover"]) * 100
            sharpe_cost = row["sharpe"] - baseline["sharpe"]
            report_lines.append(
                f"- {row['method_name']}: {row['turnover']:.1%} turnover "
                f"(-{turnover_reduction:.1%}, -{turnover_reduction_pct:.1f}% vs optimizer), "
                f"Sharpe impact: {sharpe_cost:+.3f}"
            )
        report_lines.append("")
    else:
        report_lines.append("No methods reduce turnover materially vs optimizer.")
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
        report_lines.append(f"| {row['method_name']:>20} | {row['robustness_score']:>5.3f} |")
    
    report_lines.extend([
        "",
        f"**Most Robust**: {best_robust['method_name']}",
        "",
        "## Recommendation",
        "",
    ])
    
    # Final recommendation
    baseline_sharpe = baseline["sharpe"]
    best_alternative = results_df[results_df["method"] != "optimizer"].loc[
        results_df[results_df["method"] != "optimizer"]["sharpe"].idxmax()
    ]
    
    sharpe_diff = best_alternative["sharpe"] - baseline_sharpe
    turnover_diff = baseline["turnover"] - best_alternative["turnover"]
    
    if sharpe_diff > 0.02:  # Material improvement
        verdict = f"**REPLACE OPTIMIZER: Use {best_alternative['method_name']}**"
        explanation = (
            f"The {best_alternative['method_name']} method outperforms the optimizer:\n"
            f"- Sharpe: {sharpe_diff:+.3f}\n"
            f"- CAGR: {best_alternative['cagr'] - baseline['cagr']:+.2%}\n"
            f"- Simpler to implement and maintain\n"
            f"- {'Lower' if turnover_diff > 0 else 'Similar'} turnover"
        )
    elif abs(sharpe_diff) < 0.01 and turnover_diff > 20:  # Similar performance, much lower turnover
        verdict = f"**CONSIDER {best_alternative['method_name']}: Simpler with similar performance**"
        explanation = (
            f"The {best_alternative['method_name']} method provides similar risk-adjusted returns "
            f"with simpler construction:\n"
            f"- Sharpe: {sharpe_diff:+.3f} (negligible difference)\n"
            f"- Turnover: {turnover_diff:+.1%} lower\n"
            f"- Much simpler to implement and explain"
        )
    else:
        verdict = "**KEEP OPTIMIZER: Provides best risk-adjusted returns**"
        explanation = (
            f"The optimizer outperforms simpler methods:\n"
            f"- Optimizer Sharpe: {baseline_sharpe:.3f}\n"
            f"- Best alternative ({best_alternative['method_name']}): {best_alternative['sharpe']:.3f}\n"
            f"- Advantage: +{-sharpe_diff:.3f}\n"
            f"- The added complexity is justified by improved performance"
        )
    
    report_lines.append(verdict)
    report_lines.append("")
    report_lines.append(explanation)
    report_lines.append("")
    
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    if sharpe_diff > 0.02 or (abs(sharpe_diff) < 0.01 and turnover_diff > 20):
        report_lines.append("1. **Test alternative optimization objectives** - test mean-variance, CVaR, or max Sharpe")
        report_lines.append("2. **Test dynamic portfolio construction** - switch between methods based on market conditions")
    else:
        report_lines.append("1. **Test ensemble signals** - combine 12M + 24M momentum")
        report_lines.append("2. **Test dual-momentum** - add relative strength component")
        report_lines.append("3. **Expand asset universe** - test sector rotation, international, alternatives")
    
    report_lines.append("4. **Test regime-conditional strategies** - apply different rules per macro regime")
    report_lines.append("5. **Test alternative optimization constraints** - add turnover penalty, concentration limits")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "| Method | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe | Robustness |",
        "|--------|------|--------|-------|-----|----------|------------------|------------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['method_name']:>20} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} | "
            f"{row['difficult_sharpe']:>16.3f} | {row['robustness_score']:>10.3f} |"
        )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "PORTFOLIO_CONSTRUCTION_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    logger.info("\nExperiment complete. Report generated.")


if __name__ == "__main__":
    main()
