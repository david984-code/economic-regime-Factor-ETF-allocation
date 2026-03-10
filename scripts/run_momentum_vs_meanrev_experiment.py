"""Test momentum vs mean-reversion signals at different lookback windows."""

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


def _run_experiment(lookback: int, use_momentum: bool) -> pd.DataFrame:
    """Run pure market experiment with specified lookback and signal type."""
    signal_type = "Momentum" if use_momentum else "Mean-Rev"
    logger.info("=" * 80)
    logger.info("RUNNING: %s %dM", signal_type, lookback)
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
        use_momentum=use_momentum,
    )
    return df


def _compute_signal_ic(
    prices: pd.DataFrame,
    lookback: int,
    use_momentum: bool,
) -> dict[str, float]:
    """Compute IC for signal at given lookback (no lookahead)."""
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_returns_monthly = spy_monthly.pct_change().dropna()
    spy_returns_monthly.index = spy_returns_monthly.index.to_period("M")
    
    # Compute signal incrementally (no lookahead)
    signal_list = []
    for i in range(len(spy_monthly)):
        if i < lookback:
            signal_list.append(np.nan)
        else:
            price_now = spy_monthly.iloc[i]
            price_lookback = spy_monthly.iloc[i - lookback]
            momentum = (price_now / price_lookback) - 1
            
            if use_momentum:
                signal = momentum
            else:
                signal = -momentum
            
            signal_list.append(signal)
    
    signal_raw = pd.Series(signal_list, index=spy_monthly.index)
    signal_raw.index = signal_raw.index.to_period("M")
    
    # Expanding window z-score (no lookahead)
    signal_z = signal_raw.copy()
    min_history = max(lookback, 12)
    for i in range(len(signal_raw)):
        trailing = signal_raw.iloc[:i + 1].dropna()
        if len(trailing) >= min_history:
            signal_z.iloc[i] = (signal_raw.iloc[i] - trailing.mean()) / trailing.std()
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
    """Run momentum vs mean-reversion experiment."""
    logger.info("Starting momentum vs mean-reversion experiment")
    
    # Test configurations
    lookbacks = [3, 6, 12, 24]
    signal_types = [
        {"use_momentum": False, "name": "Mean-Rev"},
        {"use_momentum": True, "name": "Momentum"},
    ]
    
    results = []
    ic_results = []
    
    # Load price data once for IC calculations
    logger.info("Loading data for IC calculations...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    
    # Run experiments
    for signal_config in signal_types:
        use_momentum = signal_config["use_momentum"]
        signal_name = signal_config["name"]
        
        for lookback in lookbacks:
            df = _run_experiment(lookback, use_momentum)
            
            if df.empty:
                logger.error(f"Experiment failed for {signal_name} {lookback}M")
                continue
            
            overall = df[df["segment"] == "OVERALL"].iloc[0]
            
            # Compute IC
            ic = _compute_signal_ic(prices, lookback, use_momentum)
            
            # Difficult period metrics
            difficult = _difficult_period_metrics(df)
            
            result = {
                "signal": signal_name,
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
                "signal": signal_name,
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
        "# Momentum vs Mean-Reversion Experiment",
        "",
        "## Experiment Setup",
        "",
        "Pure market model (0.0 macro / 1.0 market) tested with two signal types:",
        "",
        "**Momentum (trend-following):**",
        "- signal = +momentum",
        "- Bet on past winners continuing to win",
        "",
        "**Mean-Reversion (contrarian):**",
        "- signal = -momentum",
        "- Bet on past losers recovering",
        "",
        f"Tested {len(lookbacks)} lookback windows for each signal: {', '.join([f'{lb}M' for lb in lookbacks])}",
        "",
        "All signals use expanding window normalization (no lookahead bias).",
        "",
        "## Walk-Forward Performance by Signal Type",
        "",
        "### Mean-Reversion Results",
        "",
        "| Lookback | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|----------|------|--------|-------|-----|----------|",
    ]
    
    meanrev_df = results_df[results_df["signal"] == "Mean-Rev"].sort_values("lookback")
    for _, row in meanrev_df.iterrows():
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.extend([
        "",
        "### Momentum Results",
        "",
        "| Lookback | CAGR | Sharpe | MaxDD | Vol | Turnover |",
        "|----------|------|--------|-------|-----|----------|",
    ])
    
    momentum_df = results_df[results_df["signal"] == "Momentum"].sort_values("lookback")
    for _, row in momentum_df.iterrows():
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.append("")
    
    # Best configurations
    best_overall = results_df.loc[results_df["sharpe"].idxmax()]
    best_meanrev = meanrev_df.loc[meanrev_df["sharpe"].idxmax()]
    best_momentum = momentum_df.loc[momentum_df["sharpe"].idxmax()]
    
    report_lines.extend([
        "### Top Performers",
        "",
        f"- **Best Overall**: {best_overall['signal']} {best_overall['lookback']:.0f}M "
        f"(Sharpe: {best_overall['sharpe']:.3f}, CAGR: {best_overall['cagr']:.2%})",
        f"- **Best Mean-Rev**: {best_meanrev['lookback']:.0f}M "
        f"(Sharpe: {best_meanrev['sharpe']:.3f}, CAGR: {best_meanrev['cagr']:.2%})",
        f"- **Best Momentum**: {best_momentum['lookback']:.0f}M "
        f"(Sharpe: {best_momentum['sharpe']:.3f}, CAGR: {best_momentum['cagr']:.2%})",
        "",
        "## Information Coefficient by Signal Type",
        "",
        "### Mean-Reversion IC",
        "",
        "| Lookback | 1M IC | 3M IC | 6M IC | Avg IC |",
        "|----------|-------|-------|-------|--------|",
    ])
    
    meanrev_ic = ic_df[ic_df["signal"] == "Mean-Rev"].sort_values("lookback")
    for _, row in meanrev_ic.iterrows():
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['ic_1m']:>5.4f} | {row['ic_3m']:>5.4f} | "
            f"{row['ic_6m']:>5.4f} | {row['ic_avg']:>6.4f} |"
        )
    
    report_lines.extend([
        "",
        "### Momentum IC",
        "",
        "| Lookback | 1M IC | 3M IC | 6M IC | Avg IC |",
        "|----------|-------|-------|-------|--------|",
    ])
    
    momentum_ic = ic_df[ic_df["signal"] == "Momentum"].sort_values("lookback")
    for _, row in momentum_ic.iterrows():
        report_lines.append(
            f"| {row['lookback']:>8}M | {row['ic_1m']:>5.4f} | {row['ic_3m']:>5.4f} | "
            f"{row['ic_6m']:>5.4f} | {row['ic_avg']:>6.4f} |"
        )
    
    report_lines.append("")
    
    # IC comparison
    best_ic_meanrev = meanrev_ic.loc[meanrev_ic["ic_avg"].idxmax()]
    best_ic_momentum = momentum_ic.loc[momentum_ic["ic_avg"].idxmax()]
    
    report_lines.extend([
        f"**Best Mean-Rev IC**: {best_ic_meanrev['lookback']:.0f}M (IC = {best_ic_meanrev['ic_avg']:.4f})",
        f"**Best Momentum IC**: {best_ic_momentum['lookback']:.0f}M (IC = {best_ic_momentum['ic_avg']:.4f})",
        "",
        "## Difficult Period Performance (2021-2022)",
        "",
        "### Mean-Reversion in Difficult Period",
        "",
        "| Lookback | CAGR | Sharpe |",
        "|----------|------|--------|",
    ])
    
    for _, row in meanrev_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['lookback']:>8}M | {row['difficult_cagr']:>4.2%} | {row['difficult_sharpe']:>6.3f} |"
            )
    
    report_lines.extend([
        "",
        "### Momentum in Difficult Period",
        "",
        "| Lookback | CAGR | Sharpe |",
        "|----------|------|--------|",
    ])
    
    for _, row in momentum_df.iterrows():
        if not pd.isna(row["difficult_cagr"]):
            report_lines.append(
                f"| {row['lookback']:>8}M | {row['difficult_cagr']:>4.2%} | {row['difficult_sharpe']:>6.3f} |"
            )
    
    report_lines.append("")
    
    # Analysis
    report_lines.extend([
        "## Analysis",
        "",
        "### Signal Type Comparison",
        "",
    ])
    
    # Average performance by signal type
    meanrev_avg_cagr = meanrev_df["cagr"].mean()
    momentum_avg_cagr = momentum_df["cagr"].mean()
    meanrev_avg_sharpe = meanrev_df["sharpe"].mean()
    momentum_avg_sharpe = momentum_df["sharpe"].mean()
    meanrev_avg_ic = meanrev_ic["ic_avg"].mean()
    momentum_avg_ic = momentum_ic["ic_avg"].mean()
    
    report_lines.extend([
        "**Average Performance Across All Lookbacks:**",
        "",
        "| Signal | CAGR | Sharpe | Avg IC |",
        "|--------|------|--------|--------|",
        f"| Mean-Rev | {meanrev_avg_cagr:>4.2%} | {meanrev_avg_sharpe:>6.3f} | {meanrev_avg_ic:>6.4f} |",
        f"| Momentum | {momentum_avg_cagr:>4.2%} | {momentum_avg_sharpe:>6.3f} | {momentum_avg_ic:>6.4f} |",
        "",
    ])
    
    # Winner
    if momentum_avg_sharpe > meanrev_avg_sharpe:
        winner = "MOMENTUM"
        delta_sharpe = momentum_avg_sharpe - meanrev_avg_sharpe
        delta_cagr = momentum_avg_cagr - meanrev_avg_cagr
        report_lines.append(f"**Winner: {winner}** (+{delta_sharpe:.3f} Sharpe, +{delta_cagr:.2%} CAGR vs Mean-Rev)")
    else:
        winner = "MEAN-REVERSION"
        delta_sharpe = meanrev_avg_sharpe - momentum_avg_sharpe
        delta_cagr = meanrev_avg_cagr - momentum_avg_cagr
        report_lines.append(f"**Winner: {winner}** (+{delta_sharpe:.3f} Sharpe, +{delta_cagr:.2%} CAGR vs Momentum)")
    
    report_lines.append("")
    
    # Consistency analysis
    report_lines.extend([
        "### Consistency Across Lookbacks",
        "",
    ])
    
    meanrev_sharpe_std = meanrev_df["sharpe"].std()
    momentum_sharpe_std = momentum_df["sharpe"].std()
    
    if meanrev_sharpe_std < momentum_sharpe_std:
        more_consistent = "Mean-Reversion"
        less_consistent = "Momentum"
    else:
        more_consistent = "Momentum"
        less_consistent = "Mean-Reversion"
    
    report_lines.extend([
        f"- **{more_consistent}** is more consistent (Sharpe std: {min(meanrev_sharpe_std, momentum_sharpe_std):.3f})",
        f"- **{less_consistent}** is more variable (Sharpe std: {max(meanrev_sharpe_std, momentum_sharpe_std):.3f})",
        "",
    ])
    
    # Optimal lookback by signal type
    report_lines.extend([
        "### Optimal Lookback by Signal Type",
        "",
        f"**Mean-Reversion:** {best_meanrev['lookback']:.0f}M lookback performs best",
        f"- CAGR: {best_meanrev['cagr']:.2%}",
        f"- Sharpe: {best_meanrev['sharpe']:.3f}",
        f"- IC: {meanrev_ic[meanrev_ic['lookback'] == best_meanrev['lookback']].iloc[0]['ic_avg']:.4f}",
        "",
        f"**Momentum:** {best_momentum['lookback']:.0f}M lookback performs best",
        f"- CAGR: {best_momentum['cagr']:.2%}",
        f"- Sharpe: {best_momentum['sharpe']:.3f}",
        f"- IC: {momentum_ic[momentum_ic['lookback'] == best_momentum['lookback']].iloc[0]['ic_avg']:.4f}",
        "",
    ])
    
    # IC vs Performance relationship
    report_lines.extend([
        "### IC vs Realized Performance",
        "",
    ])
    
    # Check if higher IC leads to better performance
    ic_perf_corr = results_df.merge(ic_df, on=["signal", "lookback"])
    overall_ic_corr = ic_perf_corr[["sharpe", "ic_avg"]].corr().iloc[0, 1]
    
    if overall_ic_corr > 0.5:
        report_lines.append(f"- **Strong positive correlation** between IC and Sharpe (r={overall_ic_corr:.2f})")
        report_lines.append("- Higher predictive power translates to better realized performance")
    elif overall_ic_corr < -0.5:
        report_lines.append(f"- **Strong negative correlation** between IC and Sharpe (r={overall_ic_corr:.2f})")
        report_lines.append("- Counterintuitively, higher IC leads to worse performance")
    else:
        report_lines.append(f"- **Weak correlation** between IC and Sharpe (r={overall_ic_corr:.2f})")
        report_lines.append("- Predictive power does not strongly predict realized performance")
    
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
        "| Signal | Lookback | Sharpe | IC | Turnover | Robustness |",
        "|--------|----------|--------|-----|----------|------------|",
    ])
    
    for i, row in results_df.iterrows():
        ic_row = ic_df.iloc[i]
        report_lines.append(
            f"| {row['signal']:>8} | {row['lookback']:>8}M | {row['sharpe']:>6.3f} | "
            f"{ic_row['ic_avg']:>7.4f} | {row['turnover']:>8.1%} | {row['robustness_score']:>10.3f} |"
        )
    
    report_lines.extend([
        "",
        f"**Most Robust**: {best_robust['signal']} {best_robust['lookback']:.0f}M",
        "",
        "## Recommendation",
        "",
    ])
    
    # Final recommendation
    verdict_parts = []
    
    if winner == "MOMENTUM":
        verdict_parts.append("**USE MOMENTUM SIGNAL**")
        verdict_parts.append("")
        verdict_parts.append("Mean-reversion hypothesis is **rejected**.")
        verdict_parts.append("This market exhibits trend-following behavior.")
    else:
        verdict_parts.append("**USE MEAN-REVERSION SIGNAL**")
        verdict_parts.append("")
        verdict_parts.append("Mean-reversion hypothesis is **confirmed**.")
        verdict_parts.append("This market exhibits contrarian behavior.")
    
    verdict_parts.append("")
    verdict_parts.append(f"**Recommended Configuration**: {best_overall['signal']} {best_overall['lookback']:.0f}M")
    verdict_parts.append("")
    verdict_parts.append("### Performance at Recommended Configuration")
    verdict_parts.append("")
    verdict_parts.append(f"- Signal Type: {best_overall['signal']}")
    verdict_parts.append(f"- Lookback: {best_overall['lookback']:.0f} months")
    verdict_parts.append(f"- CAGR: {best_overall['cagr']:.2%}")
    verdict_parts.append(f"- Sharpe: {best_overall['sharpe']:.3f}")
    verdict_parts.append(f"- Max Drawdown: {best_overall['maxdd']:.2%}")
    verdict_parts.append(f"- Volatility: {best_overall['vol']:.2%}")
    verdict_parts.append(f"- Turnover: {best_overall['turnover']:.1%}")
    
    best_ic_row = ic_df[
        (ic_df["signal"] == best_overall["signal"]) & 
        (ic_df["lookback"] == best_overall["lookback"])
    ].iloc[0]
    
    verdict_parts.append("")
    verdict_parts.append("### IC at Recommended Configuration")
    verdict_parts.append("")
    verdict_parts.append(f"- 1M IC: {best_ic_row['ic_1m']:.4f}")
    verdict_parts.append(f"- 3M IC: {best_ic_row['ic_3m']:.4f}")
    verdict_parts.append(f"- 6M IC: {best_ic_row['ic_6m']:.4f}")
    verdict_parts.append(f"- Average IC: {best_ic_row['ic_avg']:.4f}")
    
    report_lines.extend(verdict_parts)
    report_lines.append("")
    
    # Next experiments
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    if winner == "MOMENTUM":
        report_lines.append("1. **Test dual-momentum** - relative strength + absolute momentum")
        report_lines.append("2. **Test momentum with vol scaling** - scale exposure by realized vol")
        report_lines.append("3. **Test momentum with trend filter** - only trade when above SMA")
    else:
        report_lines.append("1. **Test mean-reversion with RSI** - add overbought/oversold filter")
        report_lines.append("2. **Test mean-reversion with vol filter** - trade more after spikes")
        report_lines.append("3. **Test mean-reversion with regime** - mean-rev in ranging markets only")
    
    report_lines.append("4. **Test ensemble** - combine multiple lookbacks")
    report_lines.append("5. **Add volatility regime** - scale risk_on by vol percentile")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        "",
        "| Signal | Lookback | CAGR | Sharpe | Turnover | IC | Robustness |",
        "|--------|----------|------|--------|----------|-----|------------|",
    ])
    
    for i, row in results_df.iterrows():
        ic_row = ic_df.iloc[i]
        report_lines.append(
            f"| {row['signal']:>8} | {row['lookback']:>8}M | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['turnover']:>8.1%} | {ic_row['ic_avg']:>7.4f} | {row['robustness_score']:>10.3f} |"
        )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "MOMENTUM_VS_MEANREV_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
