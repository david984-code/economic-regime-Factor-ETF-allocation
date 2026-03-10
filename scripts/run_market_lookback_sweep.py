"""Test different momentum lookback windows for pure market signal model."""

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


def _run_experiment(lookback: int) -> pd.DataFrame:
    """Run pure market experiment with specified lookback."""
    logger.info("=" * 80)
    logger.info("RUNNING: %dM Lookback", lookback)
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
        market_lookback_months=lookback,
    )
    return df


def _compute_signal_ic(
    prices: pd.DataFrame,
    lookback: int,
) -> dict[str, float]:
    """Compute IC for mean-reversion signal at given lookback (no lookahead)."""
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_returns_monthly = spy_monthly.pct_change().dropna()
    spy_returns_monthly.index = spy_returns_monthly.index.to_period("M")
    
    # Compute momentum incrementally (no lookahead)
    mean_reversion_list = []
    for i in range(len(spy_monthly)):
        if i < lookback:
            mean_reversion_list.append(np.nan)
        else:
            price_now = spy_monthly.iloc[i]
            price_lookback = spy_monthly.iloc[i - lookback]
            momentum = (price_now / price_lookback) - 1
            mean_reversion_list.append(-momentum)
    
    mean_reversion = pd.Series(mean_reversion_list, index=spy_monthly.index)
    mean_reversion.index = mean_reversion.index.to_period("M")
    
    # Expanding window z-score (no lookahead)
    mr_z = mean_reversion.copy()
    min_history = max(lookback, 12)
    for i in range(len(mean_reversion)):
        trailing = mean_reversion.iloc[:i + 1].dropna()
        if len(trailing) >= min_history:
            mr_z.iloc[i] = (mean_reversion.iloc[i] - trailing.mean()) / trailing.std()
        else:
            mr_z.iloc[i] = 0.0
    
    # Align indices
    common = spy_returns_monthly.index.intersection(mr_z.index)
    spy_ret = spy_returns_monthly.loc[common]
    signal = mr_z.loc[common]
    
    # Forward returns
    fwd_1m = (1 + spy_ret).rolling(1).apply(lambda x: x.prod() - 1, raw=True).shift(-1)
    fwd_3m = (1 + spy_ret).rolling(3).apply(lambda x: x.prod() - 1, raw=True).shift(-3)
    fwd_6m = (1 + spy_ret).rolling(6).apply(lambda x: x.prod() - 1, raw=True).shift(-6)
    
    df = pd.DataFrame({
        "signal": signal,
        "fwd_1m": fwd_1m,
        "fwd_3m": fwd_3m,
        "fwd_6m": fwd_6m,
    }).dropna()
    
    if len(df) < 10:
        return {
            "ic_1m": np.nan, "ic_3m": np.nan, "ic_6m": np.nan, "ic_avg": np.nan,
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
        
        if (test_start.year >= 2021 and test_start.year <= 2022) or \
           (test_end.year >= 2021 and test_end.year <= 2022):
            difficult_segments.append(row)
    
    if not difficult_segments:
        return {
            "n_difficult": 0,
            "difficult_cagr": np.nan,
            "difficult_sharpe": np.nan,
        }
    
    difficult_df = pd.DataFrame(difficult_segments)
    
    return {
        "n_difficult": len(difficult_df),
        "difficult_cagr": difficult_df["Strategy_CAGR"].mean(),
        "difficult_sharpe": difficult_df["Strategy_Sharpe"].mean(),
        "difficult_maxdd": difficult_df["Strategy_MaxDD"].mean(),
    }


def main():
    """Run lookback sweep experiment for pure market model."""
    logger.info("Starting market lookback sweep experiment")
    
    # Lookback windows to test (in months)
    lookbacks = [3, 6, 9, 12, 18, 24]
    
    results = []
    ic_results = []
    
    # Load price data once for IC calculations
    logger.info("Loading data for IC calculations...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    
    # Run experiments
    for lookback in lookbacks:
        df = _run_experiment(lookback)
        
        if df.empty:
            logger.error(f"Experiment failed for {lookback}M lookback")
            continue
        
        overall = df[df["segment"] == "OVERALL"].iloc[0]
        
        # Compute IC
        ic = _compute_signal_ic(prices, lookback)
        
        # Difficult period metrics
        difficult = _difficult_period_metrics(df)
        
        result = {
            "lookback": lookback,
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
            "difficult_cagr": difficult.get("difficult_cagr", np.nan),
            "difficult_sharpe": difficult.get("difficult_sharpe", np.nan),
        }
        results.append(result)
        
        ic_result = {
            "lookback": lookback,
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
        "# Market Signal Lookback Sweep Experiment",
        "",
        "## Experiment Setup",
        "",
        "Pure market model (0.0 macro / 1.0 market) tested with different momentum lookback windows:",
        "- mean_reversion = -(SPY price return over N months)",
        "- risk_on = sigmoid(mean_reversion_z * 0.25)",
        "",
        f"Tested {len(lookbacks)} lookback windows: {', '.join([f'{lb}M' for lb in lookbacks])}",
        "",
        "All signals use expanding window normalization (no lookahead bias).",
        "",
        "## Walk-Forward Performance by Lookback",
        "",
        "| Lookback | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|----------|------|--------|-------|-----|----------|",
    ]
    
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.append("")
    
    # Best by different metrics
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_turnover = results_df.loc[results_df["turnover"].idxmin()]
    
    report_lines.extend([
        "### Top Performers",
        "",
        f"- **Best CAGR**: {best_cagr['lookback']:.0f}M ({best_cagr['cagr']:.2%})",
        f"- **Best Sharpe**: {best_sharpe['lookback']:.0f}M ({best_sharpe['sharpe']:.3f})",
        f"- **Lowest Turnover**: {best_turnover['lookback']:.0f}M ({best_turnover['turnover']:.1%})",
        "",
        "## Information Coefficient by Lookback",
        "",
        "| Lookback | 1M IC | 3M IC | 6M IC | Avg IC |",
        "|----------|-------|-------|-------|--------|",
    ])
    
    for _, row in ic_df.iterrows():
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['ic_1m']:>5.4f} | {row['ic_3m']:>5.4f} | "
            f"{row['ic_6m']:>5.4f} | {row['ic_avg']:>6.4f} |"
        )
    
    report_lines.append("")
    
    best_ic = ic_df.loc[ic_df["ic_avg"].idxmax()]
    report_lines.append(f"**Best IC**: {best_ic['lookback']:.0f}M (avg IC = {best_ic['ic_avg']:.4f})")
    report_lines.append("")
    
    # Difficult period performance
    report_lines.extend([
        "## Difficult Period Performance (2021-2022)",
        "",
        "| Lookback | CAGR | Sharpe |",
        "|----------|------|--------|",
    ])
    
    for _, row in results_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['lookback']:>8}M | {row['difficult_cagr']:>4.2%} | {row['difficult_sharpe']:>6.3f} |"
            )
    
    report_lines.append("")
    
    # Analysis
    report_lines.extend([
        "## Analysis",
        "",
        "### Relationship Between Lookback and Performance",
        "",
    ])
    
    # Check trends
    cagr_vs_lookback = results_df["cagr"].corr(results_df["lookback"])
    sharpe_vs_lookback = results_df["sharpe"].corr(results_df["lookback"])
    ic_vs_lookback = ic_df["ic_avg"].corr(ic_df["lookback"])
    turnover_vs_lookback = results_df["turnover"].corr(results_df["lookback"])
    
    if abs(cagr_vs_lookback) > 0.5:
        direction = "increases" if cagr_vs_lookback > 0 else "decreases"
        report_lines.append(f"- CAGR {direction} with longer lookback (correlation: {cagr_vs_lookback:.2f})")
    else:
        report_lines.append(f"- CAGR shows no clear trend with lookback (correlation: {cagr_vs_lookback:.2f})")
    
    if abs(sharpe_vs_lookback) > 0.5:
        direction = "increases" if sharpe_vs_lookback > 0 else "decreases"
        report_lines.append(f"- Sharpe {direction} with longer lookback (correlation: {sharpe_vs_lookback:.2f})")
    else:
        report_lines.append(f"- Sharpe shows no clear trend with lookback (correlation: {sharpe_vs_lookback:.2f})")
    
    if abs(ic_vs_lookback) > 0.5:
        direction = "increases" if ic_vs_lookback > 0 else "decreases"
        report_lines.append(f"- IC {direction} with longer lookback (correlation: {ic_vs_lookback:.2f})")
    else:
        report_lines.append(f"- IC shows no clear trend with lookback (correlation: {ic_vs_lookback:.2f})")
    
    if abs(turnover_vs_lookback) > 0.5:
        direction = "increases" if turnover_vs_lookback > 0 else "decreases"
        report_lines.append(f"- Turnover {direction} with longer lookback (correlation: {turnover_vs_lookback:.2f})")
    else:
        report_lines.append(f"- Turnover shows no clear trend with lookback (correlation: {turnover_vs_lookback:.2f})")
    
    report_lines.append("")
    
    # Performance range
    cagr_range = results_df["cagr"].max() - results_df["cagr"].min()
    sharpe_range = results_df["sharpe"].max() - results_df["sharpe"].min()
    ic_range = ic_df["ic_avg"].max() - ic_df["ic_avg"].min()
    
    report_lines.append(f"### Sensitivity to Lookback Choice")
    report_lines.append("")
    report_lines.append(f"- CAGR range: {cagr_range:.2%}")
    report_lines.append(f"- Sharpe range: {sharpe_range:.3f}")
    report_lines.append(f"- IC range: {ic_range:.4f}")
    report_lines.append("")
    
    if cagr_range < 0.005 and sharpe_range < 0.02:
        report_lines.append("**Low sensitivity**: Performance is stable across lookback choices.")
    elif cagr_range > 0.02:
        report_lines.append("**High sensitivity**: Lookback choice has substantial impact on performance.")
    else:
        report_lines.append("**Moderate sensitivity**: Lookback choice has noticeable but not dramatic impact.")
    
    report_lines.append("")
    
    # Robustness score
    results_df["robustness_score"] = (
        results_df["sharpe"] / results_df["sharpe"].max() * 0.5 +
        ic_df["ic_avg"] / ic_df["ic_avg"].max() * 0.3 +
        (1 - results_df["turnover"] / results_df["turnover"].max()) * 0.2
    )
    
    best_robust = results_df.loc[results_df["robustness_score"].idxmax()]
    
    report_lines.extend([
        "### Robustness Score",
        "",
        "Robustness = 0.5 * Sharpe + 0.3 * IC + 0.2 * (1 - Turnover)",
        "",
        "| Lookback | Score |",
        "|----------|-------|",
    ])
    
    for _, row in results_df.iterrows():
        report_lines.append(f"| {row['lookback']:>8}M | {row['robustness_score']:>5.3f} |")
    
    report_lines.append("")
    report_lines.append(f"**Most Robust**: {best_robust['lookback']:.0f}M")
    report_lines.append("")
    
    # Mean reversion vs trend following interpretation
    report_lines.extend([
        "## Signal Interpretation",
        "",
    ])
    
    # Check if signal is truly mean-reverting (negative IC) or trend-following (positive IC)
    ic_signs = ic_df[["ic_1m", "ic_3m", "ic_6m"]].apply(lambda col: (col > 0).sum())
    
    if (ic_df["ic_avg"] < 0).all():
        report_lines.append("**Signal type: MEAN REVERSION** (all ICs negative)")
        report_lines.append("- Past losers outperform past winners at all lookbacks")
        report_lines.append("- Consistent with contrarian/value investing")
    elif (ic_df["ic_avg"] > 0).all():
        report_lines.append("**Signal type: TREND FOLLOWING** (all ICs positive)")
        report_lines.append("- Past winners continue to outperform")
        report_lines.append("- Consistent with momentum investing")
    else:
        report_lines.append("**Signal type: MIXED** (varies by lookback)")
        report_lines.append("- Some lookbacks show mean reversion, others show momentum")
        report_lines.append("- Signal behavior is not consistent")
    
    report_lines.append("")
    
    # Best performing lookback interpretation
    if best_robust["lookback"] <= 6:
        report_lines.append(f"Best lookback ({best_robust['lookback']:.0f}M) is SHORT-TERM:")
        report_lines.append("- Captures recent price action")
        report_lines.append("- More responsive to changing conditions")
        report_lines.append("- May have higher turnover")
    elif best_robust["lookback"] >= 18:
        report_lines.append(f"Best lookback ({best_robust['lookback']:.0f}M) is LONG-TERM:")
        report_lines.append("- Captures multi-year trends/cycles")
        report_lines.append("- More stable signal with lower turnover")
        report_lines.append("- Less responsive to short-term volatility")
    else:
        report_lines.append(f"Best lookback ({best_robust['lookback']:.0f}M) is MEDIUM-TERM:")
        report_lines.append("- Balances responsiveness and stability")
        report_lines.append("- Standard tactical allocation horizon")
    
    report_lines.append("")
    
    report_lines.extend([
        "## Recommendation",
        "",
    ])
    
    # Final recommendation
    if best_robust["lookback"] == 12:
        verdict = "KEEP 12M: Current lookback is optimal."
        explanation = "The 12-month lookback provides the best risk-adjusted performance."
    else:
        verdict = f"CHANGE TO {best_robust['lookback']:.0f}M: Improves on current 12M baseline."
        cagr_improvement = best_robust["cagr"] - results_df[results_df["lookback"] == 12].iloc[0]["cagr"]
        sharpe_improvement = best_robust["sharpe"] - results_df[results_df["lookback"] == 12].iloc[0]["sharpe"]
        explanation = f"Improves CAGR by {cagr_improvement:+.2%} and Sharpe by {sharpe_improvement:+.3f} vs 12M."
    
    report_lines.append(f"**{verdict}**")
    report_lines.append("")
    report_lines.append(explanation)
    report_lines.append("")
    
    report_lines.extend([
        "### Performance at Recommended Lookback",
        "",
        f"- Lookback: {best_robust['lookback']:.0f} months",
        f"- CAGR: {best_robust['cagr']:.2%}",
        f"- Sharpe: {best_robust['sharpe']:.3f}",
        f"- Max Drawdown: {best_robust['maxdd']:.2%}",
        f"- Volatility: {best_robust['vol']:.2%}",
        f"- Turnover: {best_robust['turnover']:.1%}",
        "",
    ])
    
    # IC at recommended lookback
    best_ic_row = ic_df[ic_df["lookback"] == best_robust["lookback"]].iloc[0]
    report_lines.extend([
        "### IC at Recommended Lookback",
        "",
        f"- 1M IC: {best_ic_row['ic_1m']:.4f}",
        f"- 3M IC: {best_ic_row['ic_3m']:.4f}",
        f"- 6M IC: {best_ic_row['ic_6m']:.4f}",
        f"- Average IC: {best_ic_row['ic_avg']:.4f}",
        "",
    ])
    
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    # Based on results, suggest next steps
    if ic_range > 0.05:
        report_lines.append("1. **Test ensemble of multiple lookbacks** - IC varies significantly, combining may help")
    
    if (ic_df["ic_avg"] > 0).any() and (ic_df["ic_avg"] < 0).any():
        report_lines.append("2. **Test trend-following vs mean-reversion** - signal switches behavior by horizon")
    
    if best_robust["turnover"] > 150:
        report_lines.append("3. **Add turnover constraint or smoothing** - high turnover erodes returns")
    
    report_lines.append("4. **Add volatility regime filter** - scale risk_on by realized vol")
    report_lines.append("5. **Test alternative market signals** - trend strength, breadth, vol percentile")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "| Lookback | CAGR | Sharpe | Turnover | IC | Robustness |",
        "|----------|------|--------|----------|-----|------------|",
    ])
    
    for i, row in results_df.iterrows():
        ic_row = ic_df.iloc[i]
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['turnover']:>8.1%} | {ic_row['ic_avg']:>7.4f} | {row['robustness_score']:>10.3f} |"
        )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "MARKET_LOOKBACK_SWEEP_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
