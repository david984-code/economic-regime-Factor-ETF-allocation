"""Diagnose whether macro_score has predictive power for future market returns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import OUTPUTS_DIR, START_DATE, get_end_date
from src.data.market_ingestion import fetch_prices

CASH_MONTHLY = (1.05) ** (1 / 12) - 1


def _monthly_returns(prices: pd.Series) -> pd.Series:
    """Monthly returns from daily prices."""
    ret = prices.resample("ME").last().pct_change().dropna()
    ret.index = ret.index.to_period("M")
    return ret


def _forward_returns(ret: pd.Series, horizon: int) -> pd.Series:
    """Compute forward N-month total return."""
    fwd = (
        (1 + ret)
        .rolling(horizon)
        .apply(lambda x: x.prod() - 1, raw=True)
        .shift(-horizon)
    )
    return fwd


def _bucket_statistics(
    macro_score: pd.Series,
    forward_ret: pd.Series,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Compute statistics by macro_score bucket."""
    df = pd.DataFrame({"macro_score": macro_score, "fwd_ret": forward_ret}).dropna()
    if df.empty or len(df) < n_buckets:
        return pd.DataFrame()

    df["bucket"] = pd.qcut(
        df["macro_score"], q=n_buckets, labels=False, duplicates="drop"
    )

    stats = df.groupby("bucket").agg(
        {
            "macro_score": ["mean", "min", "max", "count"],
            "fwd_ret": ["mean", "std"],
        }
    )
    stats.columns = [
        "macro_score_mean",
        "macro_score_min",
        "macro_score_max",
        "n_obs",
        "fwd_ret_mean",
        "fwd_ret_std",
    ]
    stats["sharpe"] = stats["fwd_ret_mean"] / stats["fwd_ret_std"]
    stats = stats.reset_index()
    return stats


def _rolling_ic(
    macro_score: pd.Series,
    forward_ret: pd.Series,
    window: int = 24,
) -> pd.Series:
    """Compute rolling Spearman IC between macro_score and forward returns."""
    df = pd.DataFrame({"macro_score": macro_score, "fwd_ret": forward_ret}).dropna()
    if df.empty:
        return pd.Series()

    ic_series = []
    indices = []
    for i in range(window - 1, len(df)):
        window_data = df.iloc[i - window + 1 : i + 1]
        if len(window_data) >= 10:
            corr, _ = spearmanr(window_data["macro_score"], window_data["fwd_ret"])
            ic_series.append(corr)
            indices.append(df.index[i])

    return pd.Series(ic_series, index=indices)


def _baseline_signals(prices: pd.Series) -> pd.DataFrame:
    """Compute baseline trend signals for comparison."""
    monthly = prices.resample("ME").last()
    monthly.index = monthly.index.to_period("M")

    # 12M momentum
    momentum_12m = monthly.pct_change(12)

    # Simple moving average (50-day vs 200-day proxy: 2M vs 10M)
    sma_short = monthly.rolling(2).mean()
    sma_long = monthly.rolling(10).mean()
    sma_signal = (sma_short - sma_long) / sma_long

    signals = pd.DataFrame(
        {
            "momentum_12m": momentum_12m,
            "sma_signal": sma_signal,
        }
    )
    return signals


def _signal_ic(signal: pd.Series, forward_ret: pd.Series) -> tuple[float, float]:
    """Compute Spearman IC between signal and forward returns."""
    df = pd.DataFrame({"signal": signal, "fwd_ret": forward_ret}).dropna()
    if df.empty or len(df) < 10:
        return np.nan, np.nan
    corr, pval = spearmanr(df["signal"], df["fwd_ret"])
    return corr, pval


def main() -> None:
    """Diagnose macro_score predictive power."""
    print("Loading data...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    spy = prices["SPY"]
    spy_monthly = _monthly_returns(spy)

    regimes = pd.read_csv(
        OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regimes = regimes.dropna(subset=["macro_score"])
    regimes = regimes.set_index("date")
    regimes["month"] = regimes.index.to_period("M")

    # Align macro_score to monthly
    macro_monthly = regimes.groupby("month")["macro_score"].last()
    risk_on_monthly = regimes.groupby("month")["risk_on"].last()

    # Align indices
    common = spy_monthly.index.intersection(macro_monthly.index)
    spy_monthly = spy_monthly.loc[common]
    macro_monthly = macro_monthly.loc[common]
    risk_on_monthly = risk_on_monthly.loc[common]

    print(f"Analyzing {len(common)} months of data")

    # Compute forward returns at different horizons
    fwd_1m = _forward_returns(spy_monthly, 1)
    fwd_3m = _forward_returns(spy_monthly, 3)
    fwd_6m = _forward_returns(spy_monthly, 6)

    # 1. Bucket statistics
    print("Computing bucket statistics...")
    stats_1m = _bucket_statistics(macro_monthly, fwd_1m, n_buckets=5)
    stats_3m = _bucket_statistics(macro_monthly, fwd_3m, n_buckets=5)
    stats_6m = _bucket_statistics(macro_monthly, fwd_6m, n_buckets=5)

    # 2. Overall IC
    print("Computing information coefficients...")
    ic_1m, pval_1m = _signal_ic(macro_monthly, fwd_1m)
    ic_3m, pval_3m = _signal_ic(macro_monthly, fwd_3m)
    ic_6m, pval_6m = _signal_ic(macro_monthly, fwd_6m)

    ic_risk_on_1m, pval_risk_on_1m = _signal_ic(risk_on_monthly, fwd_1m)
    ic_risk_on_3m, pval_risk_on_3m = _signal_ic(risk_on_monthly, fwd_3m)
    ic_risk_on_6m, pval_risk_on_6m = _signal_ic(risk_on_monthly, fwd_6m)

    # 3. Rolling IC for stability
    print("Computing rolling IC...")
    rolling_ic_1m = _rolling_ic(macro_monthly, fwd_1m, window=24)
    rolling_ic_3m = _rolling_ic(macro_monthly, fwd_3m, window=24)

    # 4. Baseline signals
    print("Computing baseline signals...")
    baselines = _baseline_signals(spy)
    baselines = baselines.loc[common]

    ic_momentum_1m, pval_momentum_1m = _signal_ic(baselines["momentum_12m"], fwd_1m)
    ic_momentum_3m, pval_momentum_3m = _signal_ic(baselines["momentum_12m"], fwd_3m)
    ic_momentum_6m, pval_momentum_6m = _signal_ic(baselines["momentum_12m"], fwd_6m)

    ic_sma_1m, pval_sma_1m = _signal_ic(baselines["sma_signal"], fwd_1m)
    ic_sma_3m, pval_sma_3m = _signal_ic(baselines["sma_signal"], fwd_3m)
    ic_sma_6m, pval_sma_6m = _signal_ic(baselines["sma_signal"], fwd_6m)

    # Generate report
    print("Generating report...")

    report_lines = [
        "# Macro Score Predictive Power Diagnosis",
        "",
        "## Overview",
        "",
        f"- Analysis period: {common[0]} to {common[-1]}",
        f"- Total months: {len(common)}",
        f"- Macro score range: [{macro_monthly.min():.2f}, {macro_monthly.max():.2f}]",
        "",
        "## 1. Forward Returns by Macro Score Quintile",
        "",
        "### 1-Month Forward Returns",
        "",
    ]

    if not stats_1m.empty:
        report_lines.append(
            "| Bucket | Macro Score Range | N | Avg Fwd Ret | Vol | Sharpe |"
        )
        report_lines.append(
            "|--------|-------------------|---|-------------|-----|--------|"
        )
        for _, row in stats_1m.iterrows():
            bucket = int(row["bucket"])
            ms_range = f"[{row['macro_score_min']:.2f}, {row['macro_score_max']:.2f}]"
            n = int(row["n_obs"])
            fwd_ret = row["fwd_ret_mean"]
            vol = row["fwd_ret_std"]
            sharpe = row["sharpe"]
            report_lines.append(
                f"| {bucket} | {ms_range} | {n} | {fwd_ret:.2%} | {vol:.2%} | {sharpe:.3f} |"
            )
        report_lines.append("")

        # Monotonicity check
        is_monotonic = (stats_1m["fwd_ret_mean"].diff().dropna() > 0).all()
        report_lines.append(f"**Monotonic**: {'Yes' if is_monotonic else 'No'}")
        report_lines.append("")

    report_lines.append("### 3-Month Forward Returns")
    report_lines.append("")

    if not stats_3m.empty:
        report_lines.append(
            "| Bucket | Macro Score Range | N | Avg Fwd Ret | Vol | Sharpe |"
        )
        report_lines.append(
            "|--------|-------------------|---|-------------|-----|--------|"
        )
        for _, row in stats_3m.iterrows():
            bucket = int(row["bucket"])
            ms_range = f"[{row['macro_score_min']:.2f}, {row['macro_score_max']:.2f}]"
            n = int(row["n_obs"])
            fwd_ret = row["fwd_ret_mean"]
            vol = row["fwd_ret_std"]
            sharpe = row["sharpe"]
            report_lines.append(
                f"| {bucket} | {ms_range} | {n} | {fwd_ret:.2%} | {vol:.2%} | {sharpe:.3f} |"
            )
        report_lines.append("")

        is_monotonic = (stats_3m["fwd_ret_mean"].diff().dropna() > 0).all()
        report_lines.append(f"**Monotonic**: {'Yes' if is_monotonic else 'No'}")
        report_lines.append("")

    report_lines.append("### 6-Month Forward Returns")
    report_lines.append("")

    if not stats_6m.empty:
        report_lines.append(
            "| Bucket | Macro Score Range | N | Avg Fwd Ret | Vol | Sharpe |"
        )
        report_lines.append(
            "|--------|-------------------|---|-------------|-----|--------|"
        )
        for _, row in stats_6m.iterrows():
            bucket = int(row["bucket"])
            ms_range = f"[{row['macro_score_min']:.2f}, {row['macro_score_max']:.2f}]"
            n = int(row["n_obs"])
            fwd_ret = row["fwd_ret_mean"]
            vol = row["fwd_ret_std"]
            sharpe = row["sharpe"]
            report_lines.append(
                f"| {bucket} | {ms_range} | {n} | {fwd_ret:.2%} | {vol:.2%} | {sharpe:.3f} |"
            )
        report_lines.append("")

        is_monotonic = (stats_6m["fwd_ret_mean"].diff().dropna() > 0).all()
        report_lines.append(f"**Monotonic**: {'Yes' if is_monotonic else 'No'}")
        report_lines.append("")

    report_lines.extend(
        [
            "## 2. Information Coefficient (Spearman Correlation)",
            "",
            "### Macro Score",
            "",
            "| Horizon | IC | p-value | Interpretation |",
            "|---------|-------|---------|----------------|",
            f"| 1M | {ic_1m:.4f} | {pval_1m:.4f} | {'Significant' if pval_1m < 0.05 else 'Not significant'} |",
            f"| 3M | {ic_3m:.4f} | {pval_3m:.4f} | {'Significant' if pval_3m < 0.05 else 'Not significant'} |",
            f"| 6M | {ic_6m:.4f} | {pval_6m:.4f} | {'Significant' if pval_6m < 0.05 else 'Not significant'} |",
            "",
            "### Risk_on (transformed macro_score)",
            "",
            "| Horizon | IC | p-value | Interpretation |",
            "|---------|-------|---------|----------------|",
            f"| 1M | {ic_risk_on_1m:.4f} | {pval_risk_on_1m:.4f} | {'Significant' if pval_risk_on_1m < 0.05 else 'Not significant'} |",
            f"| 3M | {ic_risk_on_3m:.4f} | {pval_risk_on_3m:.4f} | {'Significant' if pval_risk_on_3m < 0.05 else 'Not significant'} |",
            f"| 6M | {ic_risk_on_6m:.4f} | {pval_risk_on_6m:.4f} | {'Significant' if pval_risk_on_6m < 0.05 else 'Not significant'} |",
            "",
            "**Note**: IC > 0.05 is considered meaningful in practice. IC > 0.10 is strong.",
            "",
            "## 3. IC Stability Over Time",
            "",
            f"- **1M IC**: Mean = {rolling_ic_1m.mean():.4f}, Std = {rolling_ic_1m.std():.4f}, % Positive = {(rolling_ic_1m > 0).mean():.1%}",
            f"- **3M IC**: Mean = {rolling_ic_3m.mean():.4f}, Std = {rolling_ic_3m.std():.4f}, % Positive = {(rolling_ic_3m > 0).mean():.1%}",
            "",
        ]
    )

    # IC stability interpretation
    ic_stability_1m = rolling_ic_1m.std()
    ic_stability_3m = rolling_ic_3m.std()
    if ic_stability_1m > 0.2 or ic_stability_3m > 0.2:
        report_lines.append(
            "**Stability**: High volatility in IC suggests signal is unstable over time."
        )
    elif (rolling_ic_1m > 0).mean() > 0.6 and (rolling_ic_3m > 0).mean() > 0.6:
        report_lines.append(
            "**Stability**: IC consistently positive, suggesting stable predictive power."
        )
    else:
        report_lines.append(
            "**Stability**: Mixed IC sign suggests weak or inconsistent signal."
        )
    report_lines.append("")

    report_lines.extend(
        [
            "## 4. Comparison vs Simple Baselines",
            "",
            "### 12-Month Momentum",
            "",
            "| Horizon | IC | p-value | Interpretation |",
            "|---------|-------|---------|----------------|",
            f"| 1M | {ic_momentum_1m:.4f} | {pval_momentum_1m:.4f} | {'Significant' if pval_momentum_1m < 0.05 else 'Not significant'} |",
            f"| 3M | {ic_momentum_3m:.4f} | {pval_momentum_3m:.4f} | {'Significant' if pval_momentum_3m < 0.05 else 'Not significant'} |",
            f"| 6M | {ic_momentum_6m:.4f} | {pval_momentum_6m:.4f} | {'Significant' if pval_momentum_6m < 0.05 else 'Not significant'} |",
            "",
            "### SMA Trend Signal (2M / 10M)",
            "",
            "| Horizon | IC | p-value | Interpretation |",
            "|---------|-------|---------|----------------|",
            f"| 1M | {ic_sma_1m:.4f} | {pval_sma_1m:.4f} | {'Significant' if pval_sma_1m < 0.05 else 'Not significant'} |",
            f"| 3M | {ic_sma_3m:.4f} | {pval_sma_3m:.4f} | {'Significant' if pval_sma_3m < 0.05 else 'Not significant'} |",
            f"| 6M | {ic_sma_6m:.4f} | {pval_sma_6m:.4f} | {'Significant' if pval_sma_6m < 0.05 else 'Not significant'} |",
            "",
            "## 5. Diagnosis",
            "",
        ]
    )

    # Diagnosis logic
    macro_ic_avg = np.mean([ic_1m, ic_3m, ic_6m])
    momentum_ic_avg = np.mean([ic_momentum_1m, ic_momentum_3m, ic_momentum_6m])
    sma_ic_avg = np.mean([ic_sma_1m, ic_sma_3m, ic_sma_6m])

    best_signal = max(
        [
            ("macro_score", macro_ic_avg),
            ("momentum_12m", momentum_ic_avg),
            ("sma_signal", sma_ic_avg),
        ],
        key=lambda x: abs(x[1]),
    )

    report_lines.append("### Predictive Power")
    report_lines.append("")

    if abs(macro_ic_avg) < 0.03:
        report_lines.append(
            "- **Macro score has near-zero predictive power** (avg IC < 0.03)"
        )
    elif abs(macro_ic_avg) < 0.05:
        report_lines.append(
            "- **Macro score has weak predictive power** (avg IC < 0.05)"
        )
    elif abs(macro_ic_avg) < 0.10:
        report_lines.append(
            "- **Macro score has moderate predictive power** (avg IC between 0.05-0.10)"
        )
    else:
        report_lines.append(
            "- **Macro score has strong predictive power** (avg IC > 0.10)"
        )

    report_lines.append(f"- Average IC across horizons: {macro_ic_avg:.4f}")
    report_lines.append("")

    report_lines.append("### Monotonicity")
    report_lines.append("")

    monotonic_1m = (
        not stats_1m.empty and (stats_1m["fwd_ret_mean"].diff().dropna() > 0).all()
    )
    monotonic_3m = (
        not stats_3m.empty and (stats_3m["fwd_ret_mean"].diff().dropna() > 0).all()
    )
    monotonic_6m = (
        not stats_6m.empty and (stats_6m["fwd_ret_mean"].diff().dropna() > 0).all()
    )

    if monotonic_1m and monotonic_3m and monotonic_6m:
        report_lines.append(
            "- **Higher macro_score consistently predicts higher forward returns** across all horizons"
        )
    elif monotonic_1m or monotonic_3m or monotonic_6m:
        report_lines.append("- **Partial monotonicity** observed at some horizons")
    else:
        report_lines.append(
            "- **Non-monotonic relationship** between macro_score and forward returns"
        )

    report_lines.append("")

    report_lines.append("### Comparison to Baselines")
    report_lines.append("")

    report_lines.append(
        f"- **Best signal**: {best_signal[0]} (avg IC = {best_signal[1]:.4f})"
    )
    report_lines.append(
        f"- Macro score vs Momentum: {macro_ic_avg:.4f} vs {momentum_ic_avg:.4f}"
    )
    report_lines.append(
        f"- Macro score vs SMA Trend: {macro_ic_avg:.4f} vs {sma_ic_avg:.4f}"
    )
    report_lines.append("")

    if abs(momentum_ic_avg) > abs(macro_ic_avg) * 1.5:
        report_lines.append(
            "- **Simple momentum is substantially stronger than macro_score**"
        )
    elif abs(sma_ic_avg) > abs(macro_ic_avg) * 1.5:
        report_lines.append(
            "- **Simple trend signal is substantially stronger than macro_score**"
        )
    elif abs(macro_ic_avg) > max(abs(momentum_ic_avg), abs(sma_ic_avg)) * 1.2:
        report_lines.append(
            "- **Macro score provides unique information beyond simple market signals**"
        )
    else:
        report_lines.append(
            "- **Macro score has similar strength to simple market signals**"
        )

    report_lines.append("")

    report_lines.append("### IC Stability")
    report_lines.append("")

    if (rolling_ic_1m > 0).mean() > 0.65:
        report_lines.append(
            "- **1M IC is consistently positive**, suggesting reliable short-term signal"
        )
    elif (rolling_ic_1m > 0).mean() > 0.5:
        report_lines.append("- **1M IC is mostly positive**, but with some instability")
    else:
        report_lines.append("- **1M IC is unstable**, frequently changing sign")

    if (rolling_ic_3m > 0).mean() > 0.65:
        report_lines.append(
            "- **3M IC is consistently positive**, suggesting reliable medium-term signal"
        )
    elif (rolling_ic_3m > 0).mean() > 0.5:
        report_lines.append("- **3M IC is mostly positive**, but with some instability")
    else:
        report_lines.append("- **3M IC is unstable**, frequently changing sign")

    report_lines.append("")

    report_lines.append("## 6. Conclusion")
    report_lines.append("")

    # Final verdict
    if abs(macro_ic_avg) > 0.05 and (rolling_ic_1m > 0).mean() > 0.6:
        verdict = "USEFUL: Macro score provides meaningful predictive information."
        recommendation = "Consider combining macro_score with market-based signals (momentum, trend) for a hybrid model."
    elif abs(macro_ic_avg) > 0.03:
        verdict = "WEAK: Macro score has some predictive power but is weak and noisy."
        recommendation = (
            "Test whether combining macro_score with market signals improves results."
        )
    else:
        verdict = "NOT USEFUL: Macro score has near-zero predictive power."
        if abs(momentum_ic_avg) > 0.05 or abs(sma_ic_avg) > 0.05:
            recommendation = "Consider replacing macro-based regime model with a pure market-based model (momentum, trend, volatility)."
        else:
            recommendation = "Neither macro nor simple market signals show strong predictive power. Consider alternative approaches."

    report_lines.append(f"**{verdict}**")
    report_lines.append("")
    report_lines.append(f"**Recommendation**: {recommendation}")
    report_lines.append("")

    # Detailed stats
    report_lines.extend(
        [
            "## Detailed Statistics",
            "",
            "### Information Coefficient Summary",
            "",
            "| Signal | 1M IC | 3M IC | 6M IC | Avg IC |",
            "|--------|-------|-------|-------|--------|",
            f"| Macro Score | {ic_1m:.4f} | {ic_3m:.4f} | {ic_6m:.4f} | {macro_ic_avg:.4f} |",
            f"| Risk_on | {ic_risk_on_1m:.4f} | {ic_risk_on_3m:.4f} | {ic_risk_on_6m:.4f} | {np.mean([ic_risk_on_1m, ic_risk_on_3m, ic_risk_on_6m]):.4f} |",
            f"| Momentum 12M | {ic_momentum_1m:.4f} | {ic_momentum_3m:.4f} | {ic_momentum_6m:.4f} | {momentum_ic_avg:.4f} |",
            f"| SMA Trend | {ic_sma_1m:.4f} | {ic_sma_3m:.4f} | {ic_sma_6m:.4f} | {sma_ic_avg:.4f} |",
            "",
        ]
    )

    report = "\n".join(report_lines)

    output_path = OUTPUTS_DIR / "MACRO_SCORE_PREDICTIVE_POWER_DIAGNOSIS.md"
    output_path.write_text(report)
    print(f"Report saved to {output_path}")

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()
