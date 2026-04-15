"""Run controlled weight sweep for hybrid macro + market signal model."""

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


def _run_experiment(macro_weight: float) -> pd.DataFrame:
    """Run experiment with specified macro weight."""
    logger.info("=" * 80)
    logger.info(
        "RUNNING: Macro weight = %.1f / Market weight = %.1f",
        macro_weight,
        1 - macro_weight,
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
        hybrid_macro_weight=macro_weight,
    )
    return df


def _compute_signal_ic(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    macro_weight: float,
) -> dict[str, float]:
    """Compute IC for combined signal at given macro weight."""
    # Monthly SPY returns
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_returns_monthly = spy_monthly.pct_change().dropna()
    spy_returns_monthly.index = spy_returns_monthly.index.to_period("M")

    # 12M momentum and mean reversion
    momentum_12m = spy_monthly.pct_change(12)
    momentum_12m.index = momentum_12m.index.to_period("M")
    mean_reversion = -momentum_12m

    # Align with regime data
    regime_monthly = regime_df.resample("ME").last()
    regime_monthly.index = regime_monthly.index.to_period("M")

    common = spy_returns_monthly.index.intersection(regime_monthly.index).intersection(
        mean_reversion.index
    )

    spy_ret = spy_returns_monthly.loc[common]
    macro_score = regime_monthly.loc[common, "macro_score"]
    mr_signal = mean_reversion.loc[common]

    # Normalize
    mr_z = (mr_signal - mr_signal.mean()) / mr_signal.std()

    # Combined score
    combined_score = macro_weight * macro_score + (1 - macro_weight) * mr_z

    # Forward returns
    fwd_1m = (1 + spy_ret).rolling(1).apply(lambda x: x.prod() - 1, raw=True).shift(-1)
    fwd_3m = (1 + spy_ret).rolling(3).apply(lambda x: x.prod() - 1, raw=True).shift(-3)
    fwd_6m = (1 + spy_ret).rolling(6).apply(lambda x: x.prod() - 1, raw=True).shift(-6)

    df = pd.DataFrame(
        {
            "combined_score": combined_score,
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
    """Run weight sweep experiment and generate comparison report."""
    logger.info("Starting hybrid weight sweep experiment")

    # Weight combinations to test
    weights = [0.2, 0.4, 0.5, 0.6, 0.8]

    results = []
    ic_results = []

    # Load regime data once for IC calculations
    logger.info("Loading data for IC calculations...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())

    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()

    # Run experiments
    for macro_weight in weights:
        df = _run_experiment(macro_weight)

        if df.empty:
            logger.error(f"Experiment failed for weight {macro_weight}")
            continue

        overall = df[df["segment"] == "OVERALL"].iloc[0]

        # Compute IC
        ic = _compute_signal_ic(prices, regime_df, macro_weight)

        result = {
            "macro_weight": macro_weight,
            "market_weight": 1 - macro_weight,
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
        }
        results.append(result)

        ic_result = {
            "macro_weight": macro_weight,
            "market_weight": 1 - macro_weight,
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
        "# Hybrid Signal Weight Sweep Experiment",
        "",
        "## Experiment Setup",
        "",
        "Test different weight combinations for hybrid signal:",
        "- combined_score = macro_weight * macro_score + (1 - macro_weight) * mean_reversion_z",
        "- risk_on = sigmoid(combined_score * 0.25)",
        "",
        f"Tested {len(weights)} weight combinations with walk-forward evaluation.",
        "",
        "## Walk-Forward Performance by Weight",
        "",
        "| Macro | Market | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|-------|--------|------|--------|-------|-----|----------|",
    ]

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['macro_weight']:.1f} | {row['market_weight']:.1f} | "
            f"{row['cagr']:.2%} | {row['sharpe']:.3f} | {row['maxdd']:.2%} | "
            f"{row['vol']:.2%} | {row['turnover']:.1%} |"
        )

    report_lines.append("")

    # Best by different metrics
    best_cagr = results_df.loc[results_df["cagr"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    best_turnover = results_df.loc[results_df["turnover"].idxmin()]

    report_lines.extend(
        [
            "### Top Performers",
            "",
            f"- **Best CAGR**: {best_cagr['macro_weight']:.1f}/{best_cagr['market_weight']:.1f} ({best_cagr['cagr']:.2%})",
            f"- **Best Sharpe**: {best_sharpe['macro_weight']:.1f}/{best_sharpe['market_weight']:.1f} ({best_sharpe['sharpe']:.3f})",
            f"- **Lowest Turnover**: {best_turnover['macro_weight']:.1f}/{best_turnover['market_weight']:.1f} ({best_turnover['turnover']:.1%})",
            "",
            "## Information Coefficient by Weight",
            "",
            "| Macro | Market | 1M IC | 3M IC | 6M IC | Avg IC |",
            "|-------|--------|-------|-------|-------|--------|",
        ]
    )

    for _, row in ic_df.iterrows():
        report_lines.append(
            f"| {row['macro_weight']:.1f} | {row['market_weight']:.1f} | "
            f"{row['ic_1m']:.4f} | {row['ic_3m']:.4f} | {row['ic_6m']:.4f} | "
            f"{row['ic_avg']:.4f} |"
        )

    report_lines.append("")

    best_ic = ic_df.loc[ic_df["ic_avg"].idxmax()]
    report_lines.append(
        f"**Best IC**: {best_ic['macro_weight']:.1f}/{best_ic['market_weight']:.1f} (avg IC = {best_ic['ic_avg']:.4f})"
    )
    report_lines.append("")

    # Analysis
    report_lines.extend(
        [
            "## Analysis",
            "",
            "### Relationship Between Weight and Performance",
            "",
        ]
    )

    # Check if performance improves with more market weight
    cagr_vs_market = results_df["cagr"].corr(results_df["market_weight"])
    sharpe_vs_market = results_df["sharpe"].corr(results_df["market_weight"])
    ic_vs_market = ic_df["ic_avg"].corr(ic_df["market_weight"])

    if cagr_vs_market > 0.5:
        report_lines.append(
            f"- **CAGR increases with higher market weight** (correlation: {cagr_vs_market:.2f})"
        )
    elif cagr_vs_market < -0.5:
        report_lines.append(
            f"- **CAGR increases with higher macro weight** (correlation: {-cagr_vs_market:.2f})"
        )
    else:
        report_lines.append(
            f"- CAGR shows no clear trend with weight changes (correlation: {cagr_vs_market:.2f})"
        )

    if sharpe_vs_market > 0.5:
        report_lines.append(
            f"- **Sharpe increases with higher market weight** (correlation: {sharpe_vs_market:.2f})"
        )
    elif sharpe_vs_market < -0.5:
        report_lines.append(
            f"- **Sharpe increases with higher macro weight** (correlation: {-sharpe_vs_market:.2f})"
        )
    else:
        report_lines.append(
            f"- Sharpe shows no clear trend with weight changes (correlation: {sharpe_vs_market:.2f})"
        )

    if ic_vs_market > 0.5:
        report_lines.append(
            f"- **IC increases with higher market weight** (correlation: {ic_vs_market:.2f})"
        )
    elif ic_vs_market < -0.5:
        report_lines.append(
            f"- **IC increases with higher macro weight** (correlation: {-ic_vs_market:.2f})"
        )
    else:
        report_lines.append(
            f"- IC shows no clear trend with weight changes (correlation: {ic_vs_market:.2f})"
        )

    report_lines.append("")

    # Turnover analysis
    turnover_vs_market = results_df["turnover"].corr(results_df["market_weight"])
    if abs(turnover_vs_market) > 0.5:
        direction = "increases" if turnover_vs_market > 0 else "decreases"
        report_lines.append(
            f"- Turnover {direction} with market weight (correlation: {turnover_vs_market:.2f})"
        )
    else:
        report_lines.append(
            f"- Turnover is relatively stable across weights (correlation: {turnover_vs_market:.2f})"
        )

    report_lines.append("")

    # Robustness score: balance Sharpe, IC, and low turnover
    results_df["robustness_score"] = (
        results_df["sharpe"] / results_df["sharpe"].max() * 0.4
        + ic_df["ic_avg"] / ic_df["ic_avg"].max() * 0.4
        + (1 - results_df["turnover"] / results_df["turnover"].max()) * 0.2
    )

    best_robust = results_df.loc[results_df["robustness_score"].idxmax()]

    report_lines.extend(
        [
            "### Robustness Analysis",
            "",
            "Robustness score = 0.4 * Sharpe + 0.4 * IC + 0.2 * (1 - Turnover)",
            "",
            "| Macro | Market | Robustness Score |",
            "|-------|--------|------------------|",
        ]
    )

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {row['macro_weight']:.1f} | {row['market_weight']:.1f} | {row['robustness_score']:.3f} |"
        )

    report_lines.append("")
    report_lines.append(
        f"**Most Robust**: {best_robust['macro_weight']:.1f}/{best_robust['market_weight']:.1f}"
    )
    report_lines.append("")

    report_lines.extend(
        [
            "## Recommendation",
            "",
        ]
    )

    # Determine recommendation based on results
    if best_robust["macro_weight"] <= 0.3:
        role = "small overlay"
        explanation = "Market mean-reversion is the dominant signal; macro adds minor diversification."
    elif best_robust["macro_weight"] <= 0.5:
        role = "balanced component"
        explanation = "Macro and market signals contribute roughly equally."
    else:
        role = "major component"
        explanation = "Macro signal remains the primary driver with market as a stabilizing overlay."

    report_lines.append(
        f"**Recommended weight: {best_robust['macro_weight']:.1f} macro / {best_robust['market_weight']:.1f} market**"
    )
    report_lines.append("")
    report_lines.append(f"Macro should serve as a **{role}**:")
    report_lines.append(f"- {explanation}")
    report_lines.append("")

    # Performance at recommended weight
    report_lines.extend(
        [
            "### Performance at Recommended Weight",
            "",
            f"- CAGR: {best_robust['cagr']:.2%}",
            f"- Sharpe: {best_robust['sharpe']:.3f}",
            f"- Max Drawdown: {best_robust['maxdd']:.2%}",
            f"- Volatility: {best_robust['vol']:.2%}",
            f"- Turnover: {best_robust['turnover']:.1%}",
            "",
        ]
    )

    # IC at recommended weight
    best_ic_row = ic_df[ic_df["macro_weight"] == best_robust["macro_weight"]].iloc[0]
    report_lines.extend(
        [
            "### IC at Recommended Weight",
            "",
            f"- 1M IC: {best_ic_row['ic_1m']:.4f}",
            f"- 3M IC: {best_ic_row['ic_3m']:.4f}",
            f"- 6M IC: {best_ic_row['ic_6m']:.4f}",
            f"- Average IC: {best_ic_row['ic_avg']:.4f}",
            "",
        ]
    )

    # Key insights
    report_lines.extend(
        [
            "## Key Insights",
            "",
        ]
    )

    cagr_range = results_df["cagr"].max() - results_df["cagr"].min()
    sharpe_range = results_df["sharpe"].max() - results_df["sharpe"].min()
    ic_range = ic_df["ic_avg"].max() - ic_df["ic_avg"].min()

    report_lines.append(
        f"1. **Performance sensitivity**: CAGR range = {cagr_range:.2%}, Sharpe range = {sharpe_range:.3f}"
    )

    if cagr_range < 0.005 and sharpe_range < 0.02:
        report_lines.append("   - Performance is stable across weight choices")
    elif cagr_range > 0.02:
        report_lines.append("   - Performance is highly sensitive to weight choice")
    else:
        report_lines.append(
            "   - Performance shows moderate sensitivity to weight choice"
        )

    report_lines.append("")
    report_lines.append(f"2. **IC sensitivity**: IC range = {ic_range:.4f}")

    if ic_range < 0.02:
        report_lines.append("   - Signal strength is stable across weights")
    else:
        report_lines.append(
            "   - Signal strength varies meaningfully with weight choice"
        )

    report_lines.append("")

    turnover_range = results_df["turnover"].max() - results_df["turnover"].min()
    report_lines.append(f"3. **Turnover range**: {turnover_range:.1%}")

    if turnover_range > 50:
        report_lines.append(
            "   - Weight choice has substantial impact on trading costs"
        )
    else:
        report_lines.append("   - Turnover is relatively stable across weights")

    report_lines.append("")

    # Check if pure market (0.0) or pure macro (1.0) would be better
    if best_robust["macro_weight"] == min(weights):
        report_lines.append(
            "4. **Lower macro weights may be even better** - consider testing 0.1 or 0.0 (pure market)"
        )
    elif best_robust["macro_weight"] == max(weights):
        report_lines.append(
            "4. **Higher macro weights may be even better** - consider testing 0.9 or 1.0 (pure macro)"
        )
    else:
        report_lines.append(
            "4. **Optimal weight is in the interior** - balanced combination is best"
        )

    report = "\n".join(report_lines)

    output_path = OUTPUTS_DIR / "HYBRID_WEIGHT_SWEEP_EXPERIMENT.md"
    output_path.write_text(report)
    logger.info("Report saved to %s", output_path)

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
