"""Compare macro-only baseline vs hybrid vs pure market signal models."""

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


def _segment_metrics(df: pd.DataFrame, regime_df: pd.DataFrame) -> dict:
    """Compute metrics for difficult periods (2021-2022 style markets)."""
    # Filter to test segments only
    test_segments = df[df["segment"] != "OVERALL"].copy()
    
    if test_segments.empty:
        return {}
    
    # Identify difficult periods: high inflation + negative equity
    difficult_segments = []
    for _, row in test_segments.iterrows():
        test_start = pd.Period(row["test_start"], freq="M").to_timestamp()
        test_end = pd.Period(row["test_end"], freq="M").to_timestamp()
        
        # Check if this segment overlaps with 2021-2022
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


def _compute_signal_ic(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    macro_weight: float,
) -> dict[str, float]:
    """Compute IC for combined signal at given macro weight."""
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_returns_monthly = spy_monthly.pct_change().dropna()
    spy_returns_monthly.index = spy_returns_monthly.index.to_period("M")
    
    # Compute momentum incrementally (no lookahead)
    mean_reversion_list = []
    for i in range(len(spy_monthly)):
        if i < 12:
            mean_reversion_list.append(np.nan)
        else:
            price_now = spy_monthly.iloc[i]
            price_12m_ago = spy_monthly.iloc[i - 12]
            momentum = (price_now / price_12m_ago) - 1
            mean_reversion_list.append(-momentum)
    
    mean_reversion = pd.Series(mean_reversion_list, index=spy_monthly.index)
    mean_reversion.index = mean_reversion.index.to_period("M")
    
    # Align with regime data
    regime_monthly = regime_df.resample("ME").last()
    regime_monthly.index = regime_monthly.index.to_period("M")
    
    common = spy_returns_monthly.index.intersection(regime_monthly.index).intersection(mean_reversion.index)
    
    spy_ret = spy_returns_monthly.loc[common]
    macro_score = regime_monthly.loc[common, "macro_score"]
    mr_signal = mean_reversion.loc[common]
    
    # Expanding window z-score (no lookahead)
    mr_z = mr_signal.copy()
    for i in range(len(mr_signal)):
        trailing = mr_signal.iloc[:i + 1].dropna()
        if len(trailing) >= 12:
            mr_z.iloc[i] = (mr_signal.iloc[i] - trailing.mean()) / trailing.std()
        else:
            mr_z.iloc[i] = 0.0
    
    # Combined score
    if macro_weight == 1.0:
        combined_score = macro_score
    elif macro_weight == 0.0:
        combined_score = mr_z
    else:
        combined_score = macro_weight * macro_score + (1 - macro_weight) * mr_z
    
    # Forward returns
    fwd_1m = (1 + spy_ret).rolling(1).apply(lambda x: x.prod() - 1, raw=True).shift(-1)
    fwd_3m = (1 + spy_ret).rolling(3).apply(lambda x: x.prod() - 1, raw=True).shift(-3)
    fwd_6m = (1 + spy_ret).rolling(6).apply(lambda x: x.prod() - 1, raw=True).shift(-6)
    
    df = pd.DataFrame({
        "combined_score": combined_score,
        "fwd_1m": fwd_1m,
        "fwd_3m": fwd_3m,
        "fwd_6m": fwd_6m,
    }).dropna()
    
    if len(df) < 10:
        return {
            "ic_1m": np.nan, "ic_3m": np.nan, "ic_6m": np.nan, "ic_avg": np.nan,
        }
    
    ic_1m, _ = spearmanr(df["combined_score"], df["fwd_1m"])
    ic_3m, _ = spearmanr(df["combined_score"], df["fwd_3m"])
    ic_6m, _ = spearmanr(df["combined_score"], df["fwd_6m"])
    
    return {
        "ic_1m": ic_1m,
        "ic_3m": ic_3m,
        "ic_6m": ic_6m,
        "ic_avg": np.mean([ic_1m, ic_3m, ic_6m]),
    }


def main():
    """Run baseline vs hybrid vs pure market comparison."""
    logger.info("Starting pure market comparison experiment")
    
    # Run baseline (macro-only)
    baseline = _run_experiment(
        "Baseline (macro-only)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=False,
    )
    
    # Run hybrid (0.2 macro / 0.8 market)
    hybrid = _run_experiment(
        "Hybrid (0.2 macro / 0.8 market)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.2,
    )
    
    # Run pure market (0.0 macro / 1.0 market)
    pure_market = _run_experiment(
        "Pure Market (mean-reversion only)",
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.0,
    )
    
    if baseline.empty or hybrid.empty or pure_market.empty:
        logger.error("One or more experiments failed.")
        sys.exit(1)
    
    # Extract OVERALL results
    baseline_overall = baseline[baseline["segment"] == "OVERALL"].iloc[0]
    hybrid_overall = hybrid[hybrid["segment"] == "OVERALL"].iloc[0]
    pure_overall = pure_market[pure_market["segment"] == "OVERALL"].iloc[0]
    
    # Load data for IC and segment analysis
    logger.info("Computing IC statistics...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes
        regime_df = load_regimes()
    
    ic_baseline = _compute_signal_ic(prices, regime_df, macro_weight=1.0)
    ic_hybrid = _compute_signal_ic(prices, regime_df, macro_weight=0.2)
    ic_pure = _compute_signal_ic(prices, regime_df, macro_weight=0.0)
    
    # Segment analysis for difficult periods
    difficult_baseline = _segment_metrics(baseline, regime_df)
    difficult_hybrid = _segment_metrics(hybrid, regime_df)
    difficult_pure = _segment_metrics(pure_market, regime_df)
    
    # Build comparison report
    report_lines = [
        "# Pure Market Signal Comparison",
        "",
        "## Experiment Setup",
        "",
        "Three models tested with corrected, lookahead-free implementation:",
        "",
        "1. **Baseline (Macro-only)**:",
        "   - risk_on = sigmoid(macro_score * 0.25)",
        "",
        "2. **Hybrid (0.2 macro / 0.8 market)**:",
        "   - combined_score = 0.2 * macro_score + 0.8 * mean_reversion_z",
        "   - risk_on = sigmoid(combined_score * 0.25)",
        "",
        "3. **Pure Market (mean-reversion only)**:",
        "   - combined_score = mean_reversion_z",
        "   - risk_on = sigmoid(combined_score * 0.25)",
        "",
        "All signals use expanding window normalization (no lookahead bias).",
        "",
        "## Overall Walk-Forward Performance",
        "",
        "| Metric          | Baseline (1.0 macro) | Hybrid (0.2 macro) | Pure Market (0.0 macro) |",
        "|-----------------|----------------------|--------------------|-------------------------|",
        f"| CAGR            | {baseline_overall['Strategy_CAGR']:>20.2%} | {hybrid_overall['Strategy_CAGR']:>18.2%} | {pure_overall['Strategy_CAGR']:>23.2%} |",
        f"| Sharpe Ratio    | {baseline_overall['Strategy_Sharpe']:>20.3f} | {hybrid_overall['Strategy_Sharpe']:>18.3f} | {pure_overall['Strategy_Sharpe']:>23.3f} |",
        f"| Max Drawdown    | {baseline_overall['Strategy_MaxDD']:>20.2%} | {hybrid_overall['Strategy_MaxDD']:>18.2%} | {pure_overall['Strategy_MaxDD']:>23.2%} |",
        f"| Volatility      | {baseline_overall['Strategy_Vol']:>20.2%} | {hybrid_overall['Strategy_Vol']:>18.2%} | {pure_overall['Strategy_Vol']:>23.2%} |",
        f"| Turnover        | {baseline_overall.get('Strategy_Turnover', 0.0):>20.1%} | {hybrid_overall.get('Strategy_Turnover', 0.0):>18.1%} | {pure_overall.get('Strategy_Turnover', 0.0):>23.1%} |",
        "",
        "### Differences vs Baseline",
        "",
        "| Metric          | Hybrid vs Baseline | Pure vs Baseline | Pure vs Hybrid |",
        "|-----------------|-------------------|------------------|----------------|",
        f"| CAGR            | {hybrid_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']:>18.2%} | {pure_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']:>16.2%} | {pure_overall['Strategy_CAGR'] - hybrid_overall['Strategy_CAGR']:>14.2%} |",
        f"| Sharpe          | {hybrid_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']:>18.3f} | {pure_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']:>16.3f} | {pure_overall['Strategy_Sharpe'] - hybrid_overall['Strategy_Sharpe']:>14.3f} |",
        f"| Turnover        | {hybrid_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):>18.1%} | {pure_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):>16.1%} | {pure_overall.get('Strategy_Turnover', 0.0) - hybrid_overall.get('Strategy_Turnover', 0.0):>14.1%} |",
        "",
        "## Information Coefficient Analysis",
        "",
        "| Model | 1M IC | 3M IC | 6M IC | Avg IC |",
        "|-------|-------|-------|-------|--------|",
        f"| Baseline (1.0 macro) | {ic_baseline['ic_1m']:>7.4f} | {ic_baseline['ic_3m']:>7.4f} | {ic_baseline['ic_6m']:>7.4f} | {ic_baseline['ic_avg']:>8.4f} |",
        f"| Hybrid (0.2 macro)   | {ic_hybrid['ic_1m']:>7.4f} | {ic_hybrid['ic_3m']:>7.4f} | {ic_hybrid['ic_6m']:>7.4f} | {ic_hybrid['ic_avg']:>8.4f} |",
        f"| Pure Market          | {ic_pure['ic_1m']:>7.4f} | {ic_pure['ic_3m']:>7.4f} | {ic_pure['ic_6m']:>7.4f} | {ic_pure['ic_avg']:>8.4f} |",
        "",
        "**Note**: IC > 0.05 is meaningful, IC > 0.10 is strong.",
        "",
        "## Benchmark Comparison",
        "",
        "### Baseline (Macro-only)",
        "",
        "| Benchmark    | CAGR     | Sharpe   | MaxDD    |",
        "|--------------|----------|----------|----------|",
        f"| SPY          | {baseline_overall.get('SPY_CAGR', 0.0):>8.2%} | {baseline_overall.get('SPY_Sharpe', 0.0):>8.3f} | {baseline_overall.get('SPY_MaxDD', 0.0):>8.2%} |",
        f"| 60/40        | {baseline_overall.get('60/40_CAGR', 0.0):>8.2%} | {baseline_overall.get('60/40_Sharpe', 0.0):>8.3f} | {baseline_overall.get('60/40_MaxDD', 0.0):>8.2%} |",
        f"| Equal_Weight | {baseline_overall.get('Equal_Weight_CAGR', 0.0):>8.2%} | {baseline_overall.get('Equal_Weight_Sharpe', 0.0):>8.3f} | {baseline_overall.get('Equal_Weight_MaxDD', 0.0):>8.2%} |",
        f"| Risk_On_Off  | {baseline_overall.get('Risk_On_Off_CAGR', 0.0):>8.2%} | {baseline_overall.get('Risk_On_Off_Sharpe', 0.0):>8.3f} | {baseline_overall.get('Risk_On_Off_MaxDD', 0.0):>8.2%} |",
        "",
        "### Hybrid (0.2 macro / 0.8 market)",
        "",
        "| Benchmark    | CAGR     | Sharpe   | MaxDD    |",
        "|--------------|----------|----------|----------|",
        f"| SPY          | {hybrid_overall.get('SPY_CAGR', 0.0):>8.2%} | {hybrid_overall.get('SPY_Sharpe', 0.0):>8.3f} | {hybrid_overall.get('SPY_MaxDD', 0.0):>8.2%} |",
        f"| 60/40        | {hybrid_overall.get('60/40_CAGR', 0.0):>8.2%} | {hybrid_overall.get('60/40_Sharpe', 0.0):>8.3f} | {hybrid_overall.get('60/40_MaxDD', 0.0):>8.2%} |",
        f"| Equal_Weight | {hybrid_overall.get('Equal_Weight_CAGR', 0.0):>8.2%} | {hybrid_overall.get('Equal_Weight_Sharpe', 0.0):>8.3f} | {hybrid_overall.get('Equal_Weight_MaxDD', 0.0):>8.2%} |",
        f"| Risk_On_Off  | {hybrid_overall.get('Risk_On_Off_CAGR', 0.0):>8.2%} | {hybrid_overall.get('Risk_On_Off_Sharpe', 0.0):>8.3f} | {hybrid_overall.get('Risk_On_Off_MaxDD', 0.0):>8.2%} |",
        "",
        "### Pure Market",
        "",
        "| Benchmark    | CAGR     | Sharpe   | MaxDD    |",
        "|--------------|----------|----------|----------|",
        f"| SPY          | {pure_overall.get('SPY_CAGR', 0.0):>8.2%} | {pure_overall.get('SPY_Sharpe', 0.0):>8.3f} | {pure_overall.get('SPY_MaxDD', 0.0):>8.2%} |",
        f"| 60/40        | {pure_overall.get('60/40_CAGR', 0.0):>8.2%} | {pure_overall.get('60/40_Sharpe', 0.0):>8.3f} | {pure_overall.get('60/40_MaxDD', 0.0):>8.2%} |",
        f"| Equal_Weight | {pure_overall.get('Equal_Weight_CAGR', 0.0):>8.2%} | {pure_overall.get('Equal_Weight_Sharpe', 0.0):>8.3f} | {pure_overall.get('Equal_Weight_MaxDD', 0.0):>8.2%} |",
        f"| Risk_On_Off  | {pure_overall.get('Risk_On_Off_CAGR', 0.0):>8.2%} | {pure_overall.get('Risk_On_Off_Sharpe', 0.0):>8.3f} | {pure_overall.get('Risk_On_Off_MaxDD', 0.0):>8.2%} |",
        "",
        "## Difficult Period Performance (2021-2022)",
        "",
    ]
    
    if difficult_baseline.get("n_difficult", 0) > 0:
        report_lines.extend([
            f"Analyzing {difficult_baseline['n_difficult']} test segments overlapping 2021-2022:",
            "",
            "| Model | CAGR | Sharpe | MaxDD |",
            "|-------|------|--------|-------|",
            f"| Baseline | {difficult_baseline['difficult_cagr']:>4.2%} | {difficult_baseline['difficult_sharpe']:>6.3f} | {difficult_baseline['difficult_maxdd']:>5.2%} |",
            f"| Hybrid   | {difficult_hybrid['difficult_cagr']:>4.2%} | {difficult_hybrid['difficult_sharpe']:>6.3f} | {difficult_hybrid['difficult_maxdd']:>5.2%} |",
            f"| Pure     | {difficult_pure['difficult_cagr']:>4.2%} | {difficult_pure['difficult_sharpe']:>6.3f} | {difficult_pure['difficult_maxdd']:>5.2%} |",
            "",
        ])
    else:
        report_lines.append("No test segments overlap 2021-2022 period.")
        report_lines.append("")
    
    report_lines.extend([
        "## Analysis",
        "",
        "### Does Macro Add Value?",
        "",
    ])
    
    # Compare hybrid vs pure market
    hybrid_vs_pure_cagr = hybrid_overall['Strategy_CAGR'] - pure_overall['Strategy_CAGR']
    hybrid_vs_pure_sharpe = hybrid_overall['Strategy_Sharpe'] - pure_overall['Strategy_Sharpe']
    hybrid_vs_pure_ic = ic_hybrid['ic_avg'] - ic_pure['ic_avg']
    
    if hybrid_vs_pure_cagr > 0.002 and hybrid_vs_pure_sharpe > 0.01:
        report_lines.append("**YES**: The 20% macro overlay improves both CAGR and Sharpe vs pure market.")
    elif hybrid_vs_pure_cagr < -0.002 or hybrid_vs_pure_sharpe < -0.01:
        report_lines.append("**NO**: Pure market model outperforms hybrid.")
    else:
        report_lines.append("**NEUTRAL**: Macro overlay has minimal impact on performance.")
    
    report_lines.append("")
    report_lines.append(f"- Hybrid vs Pure: CAGR {hybrid_vs_pure_cagr:+.2%}, Sharpe {hybrid_vs_pure_sharpe:+.3f}, IC {hybrid_vs_pure_ic:+.4f}")
    report_lines.append("")
    
    # Best model overall
    models = [
        ("Baseline", baseline_overall['Strategy_CAGR'], baseline_overall['Strategy_Sharpe'], baseline_overall.get('Strategy_Turnover', 0.0), ic_baseline['ic_avg']),
        ("Hybrid", hybrid_overall['Strategy_CAGR'], hybrid_overall['Strategy_Sharpe'], hybrid_overall.get('Strategy_Turnover', 0.0), ic_hybrid['ic_avg']),
        ("Pure Market", pure_overall['Strategy_CAGR'], pure_overall['Strategy_Sharpe'], pure_overall.get('Strategy_Turnover', 0.0), ic_pure['ic_avg']),
    ]
    
    best_cagr = max(models, key=lambda x: x[1])
    best_sharpe = max(models, key=lambda x: x[2])
    best_turnover = min(models, key=lambda x: x[3])
    best_ic = max(models, key=lambda x: x[4])
    
    report_lines.extend([
        "### Top Performers",
        "",
        f"- **Best CAGR**: {best_cagr[0]} ({best_cagr[1]:.2%})",
        f"- **Best Sharpe**: {best_sharpe[0]} ({best_sharpe[2]:.3f})",
        f"- **Lowest Turnover**: {best_turnover[0]} ({best_turnover[3]:.1%})",
        f"- **Best IC**: {best_ic[0]} ({best_ic[4]:.4f})",
        "",
        "### Signal Strength Comparison",
        "",
        f"- **Baseline IC**: {ic_baseline['ic_avg']:.4f} (macro-only)",
        f"- **Hybrid IC**: {ic_hybrid['ic_avg']:.4f} (0.2 macro / 0.8 market)",
        f"- **Pure IC**: {ic_pure['ic_avg']:.4f} (market-only)",
        "",
    ])
    
    ic_improvement_hybrid = ic_hybrid['ic_avg'] - ic_baseline['ic_avg']
    ic_improvement_pure = ic_pure['ic_avg'] - ic_baseline['ic_avg']
    
    report_lines.append(f"- Hybrid improves IC by {ic_improvement_hybrid:+.4f} vs baseline")
    report_lines.append(f"- Pure improves IC by {ic_improvement_pure:+.4f} vs baseline")
    report_lines.append("")
    
    # Turnover analysis
    report_lines.extend([
        "### Turnover Reduction",
        "",
        f"- Baseline: {baseline_overall.get('Strategy_Turnover', 0.0):.1%} annual",
        f"- Hybrid: {hybrid_overall.get('Strategy_Turnover', 0.0):.1%} annual ({hybrid_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):+.1%})",
        f"- Pure: {pure_overall.get('Strategy_Turnover', 0.0):.1%} annual ({pure_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):+.1%})",
        "",
    ])
    
    # Robustness across segments
    baseline_segs = baseline[baseline["segment"] != "OVERALL"]
    hybrid_segs = hybrid[hybrid["segment"] != "OVERALL"]
    pure_segs = pure_market[pure_market["segment"] != "OVERALL"]
    
    baseline_win_rate = (baseline_segs["Strategy_CAGR"] > baseline_segs["SPY_CAGR"]).mean()
    hybrid_win_rate = (hybrid_segs["Strategy_CAGR"] > hybrid_segs["SPY_CAGR"]).mean()
    pure_win_rate = (pure_segs["Strategy_CAGR"] > pure_segs["SPY_CAGR"]).mean()
    
    report_lines.extend([
        "### Segment-Level Robustness",
        "",
        f"- Baseline beat rate vs SPY: {baseline_win_rate:.1%}",
        f"- Hybrid beat rate vs SPY: {hybrid_win_rate:.1%}",
        f"- Pure beat rate vs SPY: {pure_win_rate:.1%}",
        "",
    ])
    
    report_lines.extend([
        "## Conclusion",
        "",
    ])
    
    # Determine final verdict
    pure_is_best = (pure_overall['Strategy_CAGR'] >= hybrid_overall['Strategy_CAGR'] and 
                    pure_overall['Strategy_Sharpe'] >= hybrid_overall['Strategy_Sharpe'] - 0.005)
    
    if pure_is_best:
        verdict = "PURE MARKET: The 20% macro overlay does not add value."
        explanation = [
            "The pure mean-reversion signal performs as well or better than the hybrid.",
            "Adding macro_score introduces complexity without improving results.",
            "The market signal alone captures the predictive information needed.",
        ]
    else:
        verdict = "HYBRID: The 20% macro overlay adds modest value."
        explanation = [
            "The hybrid model provides a better balance than pure market.",
            f"Improvements: CAGR {hybrid_vs_pure_cagr:+.2%}, Sharpe {hybrid_vs_pure_sharpe:+.3f}",
            "Macro provides diversification from pure momentum/mean-reversion.",
        ]
    
    report_lines.append(f"**{verdict}**")
    report_lines.append("")
    
    for line in explanation:
        report_lines.append(f"- {line}")
    
    report_lines.append("")
    
    # Recommendation
    report_lines.extend([
        "## Recommendation",
        "",
    ])
    
    if pure_is_best:
        report_lines.extend([
            "**Adopt the pure market model as the new production baseline:**",
            "",
            f"- CAGR: {pure_overall['Strategy_CAGR']:.2%}",
            f"- Sharpe: {pure_overall['Strategy_Sharpe']:.3f}",
            f"- Turnover: {pure_overall.get('Strategy_Turnover', 0.0):.1%}",
            f"- IC: {ic_pure['ic_avg']:.4f}",
            "",
            "This represents a fundamental shift:",
            "- **Before**: Macro-driven regime model",
            "- **After**: Market-driven mean-reversion model",
            "",
            "The economic regime classification adds no value beyond the market signal.",
            "",
            "### Next Experiment",
            "",
            "Test alternative market signals to potentially improve further:",
            "1. Different momentum lookbacks (6M, 18M, 24M)",
            "2. Trend strength (distance from moving average)",
            "3. Volatility regime (realized vol vs historical average)",
            "4. Combined momentum + volatility signal",
        ])
    else:
        report_lines.extend([
            "**Keep the hybrid model (0.2 macro / 0.8 market) as production baseline:**",
            "",
            f"- CAGR: {hybrid_overall['Strategy_CAGR']:.2%}",
            f"- Sharpe: {hybrid_overall['Strategy_Sharpe']:.3f}",
            f"- Turnover: {hybrid_overall.get('Strategy_Turnover', 0.0):.1%}",
            f"- IC: {ic_hybrid['ic_avg']:.4f}",
            "",
            "The macro overlay provides:",
            "- Modest performance improvement",
            "- Fundamental context for allocation decisions",
            "- Connection to economic intuition",
            "",
            "### Next Experiment",
            "",
            "Test enhancements to the hybrid framework:",
            "1. Add volatility regime as third signal component",
            "2. Test dynamic weight adjustment (higher macro weight in low-vol periods)",
            "3. Test alternative market signals (trend, vol, breadth)",
            "4. Add regime-specific risk management overlays",
        ])
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "### Performance Summary",
        "",
        "| Model | CAGR | Sharpe | MaxDD | Vol | Turnover | IC |",
        "|-------|------|--------|-------|-----|----------|-----|",
        f"| Baseline | {baseline_overall['Strategy_CAGR']:.2%} | {baseline_overall['Strategy_Sharpe']:.3f} | {baseline_overall['Strategy_MaxDD']:.2%} | {baseline_overall['Strategy_Vol']:.2%} | {baseline_overall.get('Strategy_Turnover', 0.0):.1%} | {ic_baseline['ic_avg']:.4f} |",
        f"| Hybrid   | {hybrid_overall['Strategy_CAGR']:.2%} | {hybrid_overall['Strategy_Sharpe']:.3f} | {hybrid_overall['Strategy_MaxDD']:.2%} | {hybrid_overall['Strategy_Vol']:.2%} | {hybrid_overall.get('Strategy_Turnover', 0.0):.1%} | {ic_hybrid['ic_avg']:.4f} |",
        f"| Pure     | {pure_overall['Strategy_CAGR']:.2%} | {pure_overall['Strategy_Sharpe']:.3f} | {pure_overall['Strategy_MaxDD']:.2%} | {pure_overall['Strategy_Vol']:.2%} | {pure_overall.get('Strategy_Turnover', 0.0):.1%} | {ic_pure['ic_avg']:.4f} |",
        "",
        "### vs Baseline Improvements",
        "",
        "| Model | CAGR | Sharpe | Turnover | IC |",
        "|-------|------|--------|----------|-----|",
        f"| Hybrid | {hybrid_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']:+.2%} | {hybrid_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']:+.3f} | {hybrid_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):+.1%} | {ic_hybrid['ic_avg'] - ic_baseline['ic_avg']:+.4f} |",
        f"| Pure   | {pure_overall['Strategy_CAGR'] - baseline_overall['Strategy_CAGR']:+.2%} | {pure_overall['Strategy_Sharpe'] - baseline_overall['Strategy_Sharpe']:+.3f} | {pure_overall.get('Strategy_Turnover', 0.0) - baseline_overall.get('Strategy_Turnover', 0.0):+.1%} | {ic_pure['ic_avg'] - ic_baseline['ic_avg']:+.4f} |",
    ])
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "PURE_MARKET_COMPARISON.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
