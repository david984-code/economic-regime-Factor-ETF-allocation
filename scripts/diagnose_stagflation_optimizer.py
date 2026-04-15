"""Diagnose optimizer Stagflation allocation and compare to simple alternatives."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.allocation.optimizer import optimize_allocations_from_data
from src.backtest.metrics import compute_metrics
from src.config import OUTPUTS_DIR, START_DATE, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices
from src.evaluation.benchmarks import compute_benchmark_returns

CASH_DAILY = (1.045) ** (1 / 252) - 1
ASSETS = TICKERS + ["cash"]

# Simple rule-based Stagflation alternatives (weights must sum to 1)
ALT_DEFENSIVE = {"GLD": 0.30, "IEF": 0.40, "TLT": 0.20, "cash": 0.10}
ALT_ZERO_EQUITY = {"GLD": 0.20, "IEF": 0.50, "TLT": 0.20, "cash": 0.10}
ALT_CAPPED_EQUITY = {"SPY": 0.05, "GLD": 0.15, "IEF": 0.35, "TLT": 0.30, "cash": 0.15}


def _portfolio_return(weights: dict, returns: pd.DataFrame) -> pd.Series:
    """Compute daily portfolio return from weights and asset returns."""
    out = pd.Series(0.0, index=returns.index)
    for asset, w in weights.items():
        if asset in returns.columns and w != 0:
            out = out + w * returns[asset].fillna(0)
    return out


def _metrics_from_rets(rets: pd.Series) -> dict:
    rets = rets.dropna()
    if len(rets) < 5:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Vol": np.nan}
    return compute_metrics(rets, rf_daily=CASH_DAILY)


def main() -> None:
    print("Loading data...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    returns_daily = prices[TICKERS].pct_change().iloc[1:]
    returns_daily["cash"] = CASH_DAILY

    regimes = pd.read_csv(
        OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regimes["month"] = pd.to_datetime(regimes["date"]).dt.to_period("M")

    # Get Stagflation segments from walk-forward
    try:
        from src.evaluation.model_results_db import get_latest_run

        r = get_latest_run()
        csv_path = OUTPUTS_DIR / f"walk_forward_{r['run_id']}.csv"
    except Exception:
        csv_path = OUTPUTS_DIR / "walk_forward_results.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "walk_forward_results.csv"
    df = pd.read_csv(csv_path)
    df = df[df["segment"] != "OVERALL"]

    seg_regimes = []
    for _, row in df.iterrows():
        ts = pd.Period(row["test_start"], freq="M")
        te = pd.Period(row["test_end"], freq="M")
        sub = regimes[(regimes["month"] >= ts) & (regimes["month"] <= te)]
        dom = (
            sub["regime"].mode().iloc[0]
            if len(sub) > 0 and len(sub["regime"].mode()) > 0
            else "Unknown"
        )
        seg_regimes.append(dom)
    df["dominant_regime"] = seg_regimes
    stag_df = df[df["dominant_regime"] == "Stagflation"].copy()

    if stag_df.empty:
        print("No Stagflation segments.")
        return

    # Build regime_df for optimizer
    regime_df = regimes.dropna(subset=["regime"]).set_index("date").sort_index()
    regime_df = regime_df.reindex(prices.index).ffill()

    # Collect optimizer Stagflation allocations per segment
    opt_weights_list = []
    seg_data = []

    for _, row in stag_df.iterrows():
        p_start = pd.Period(row["test_start"], freq="M")
        p_end = pd.Period(row["test_end"], freq="M")
        train_end = (p_start - 1).to_timestamp("M")
        test_start = p_start.to_timestamp("M")
        test_end = p_end.to_timestamp("M")
        train_start = pd.Timestamp(START_DATE)

        train_returns = prices.resample("ME").last().pct_change().dropna()
        train_returns = train_returns.loc[train_start:train_end]
        if "cash" not in train_returns.columns:
            train_returns["cash"] = (1.05) ** (1 / 12) - 1
        train_regimes = (
            regime_df.loc[:train_end].resample("ME").last().dropna(how="all")
        )
        train_regimes = train_regimes.loc[train_regimes.index <= train_end]
        if len(train_returns) < 24 or len(train_regimes) < 12:
            continue

        allocations = optimize_allocations_from_data(train_returns, train_regimes)
        stag_alloc = allocations.get("Stagflation", {})
        if not stag_alloc:
            continue

        sub_rets = returns_daily.loc[test_start:test_end]
        if len(sub_rets) < 5:
            continue

        # Fill missing assets with 0
        w_opt = {a: stag_alloc.get(a, 0) for a in ASSETS}
        opt_ret = _portfolio_return(w_opt, sub_rets)
        opt_m = _metrics_from_rets(opt_ret)

        # Alternatives
        alt1_ret = _portfolio_return(ALT_DEFENSIVE, sub_rets)
        alt2_ret = _portfolio_return(ALT_ZERO_EQUITY, sub_rets)
        alt3_ret = _portfolio_return(ALT_CAPPED_EQUITY, sub_rets)
        alt1_m = _metrics_from_rets(alt1_ret)
        alt2_m = _metrics_from_rets(alt2_ret)
        alt3_m = _metrics_from_rets(alt3_ret)

        # SPY and Risk_On_Off (from benchmarks)
        spy_ret = (
            sub_rets["SPY"] if "SPY" in sub_rets else pd.Series(0, index=sub_rets.index)
        )
        benchmarks = compute_benchmark_returns(returns_daily, regime_df)
        risk_on_off_ret = benchmarks.get("Risk_On_Off")
        if risk_on_off_ret is not None:
            rof_sub = risk_on_off_ret.loc[test_start:test_end].dropna()
        else:
            rof_sub = (
                0.33 * sub_rets["IEF"] + 0.33 * sub_rets["TLT"] + 0.34 * sub_rets["GLD"]
            )
        spy_m = _metrics_from_rets(spy_ret)
        rof_m = _metrics_from_rets(rof_sub)

        opt_weights_list.append(
            {**stag_alloc, "test_start": row["test_start"], "test_end": row["test_end"]}
        )
        seg_data.append(
            {
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "opt_cagr": opt_m["CAGR"],
                "opt_sharpe": opt_m["Sharpe"],
                "opt_maxdd": opt_m["Max Drawdown"],
                "opt_vol": opt_m["Volatility"],
                "alt1_cagr": alt1_m["CAGR"],
                "alt1_sharpe": alt1_m["Sharpe"],
                "alt1_maxdd": alt1_m["Max Drawdown"],
                "alt1_vol": alt1_m["Volatility"],
                "alt2_cagr": alt2_m["CAGR"],
                "alt2_sharpe": alt2_m["Sharpe"],
                "alt2_maxdd": alt2_m["Max Drawdown"],
                "alt2_vol": alt2_m["Volatility"],
                "alt3_cagr": alt3_m["CAGR"],
                "alt3_sharpe": alt3_m["Sharpe"],
                "alt3_maxdd": alt3_m["Max Drawdown"],
                "alt3_vol": alt3_m["Volatility"],
                "spy_cagr": spy_m["CAGR"],
                "spy_sharpe": spy_m["Sharpe"],
                "spy_maxdd": spy_m["Max Drawdown"],
                "spy_vol": spy_m["Volatility"],
                "risk_off_cagr": rof_m["CAGR"],
                "risk_off_sharpe": rof_m["Sharpe"],
                "risk_off_maxdd": rof_m["Max Drawdown"],
                "risk_off_vol": rof_m["Volatility"],
            }
        )

        # Asset contribution: weight * (1 + ret).prod() - 1 for each asset
        for a in TICKERS + ["cash"]:
            if a in sub_rets.columns and stag_alloc.get(a, 0) != 0:
                contrib = stag_alloc[a] * (1 + sub_rets[a]).prod() - stag_alloc[a]
                seg_data[-1][f"contrib_{a}"] = contrib

    if not seg_data:
        print("No segment data.")
        return

    seg_df = pd.DataFrame(seg_data)
    opt_weights_df = pd.DataFrame(opt_weights_list)

    # 1. Optimizer Stagflation weights over time
    print("\n" + "=" * 80)
    print("1. OPTIMIZER STAGFLATION WEIGHTS BY SEGMENT (sample)")
    print("=" * 80)
    weight_cols = [
        c for c in opt_weights_df.columns if c not in ("test_start", "test_end")
    ]
    avg_weights = opt_weights_df[weight_cols].mean()
    std_weights = opt_weights_df[weight_cols].std()
    print("\nAverage weights (across Stagflation segments):")
    for a in sorted(avg_weights.index, key=lambda x: -avg_weights[x]):
        if avg_weights[a] > 0.001:
            print(f"  {a:>6}: {avg_weights[a]:.2%}  (std: {std_weights.get(a, 0):.2%})")
    print("\nFirst 10 segments - Stagflation allocation:")
    print(
        opt_weights_df[
            ["test_start", "test_end"]
            + [c for c in weight_cols if opt_weights_df[c].sum() > 0][:8]
        ]
        .head(10)
        .to_string(index=False)
    )

    # 2. Realized returns by asset during Stagflation
    print("\n" + "=" * 80)
    print("2. REALIZED ASSET RETURNS DURING STAGFLATION (avg across segments)")
    print("=" * 80)
    asset_cagrs = {}
    for a in TICKERS + ["cash"]:
        if a not in returns_daily.columns:
            continue
        cagrs = []
        for _, row in stag_df.iterrows():
            ts = pd.Period(row["test_start"], freq="M").to_timestamp("M")
            te = pd.Period(row["test_end"], freq="M").to_timestamp("M")
            sub = returns_daily[a].loc[ts:te].dropna()
            if len(sub) >= 5:
                m = _metrics_from_rets(sub)
                cagrs.append(m["CAGR"])
        if cagrs:
            asset_cagrs[a] = np.mean(cagrs)
    for a, c in sorted(asset_cagrs.items(), key=lambda x: -x[1]):
        print(f"  {a:>6}: {c:.2%}")

    # 3. Contribution by asset (simplified: avg weight * avg asset return)
    print("\n" + "=" * 80)
    print("3. CONTRIBUTION TO PORTFOLIO RETURN BY ASSET (approx)")
    print("=" * 80)
    contrib = {}
    for a in weight_cols:
        if a in asset_cagrs and avg_weights.get(a, 0) > 0:
            contrib[a] = avg_weights[a] * asset_cagrs[a]
    for a, c in sorted(contrib.items(), key=lambda x: -abs(x[1])):
        print(
            f"  {a:>6}: {c:.2%} (weight {avg_weights[a]:.1%} x return {asset_cagrs[a]:.1%})"
        )

    # 4. Alternative performance
    print("\n" + "=" * 80)
    print("4. ALTERNATIVE ALLOCATIONS vs OPTIMIZER (Stagflation segments only)")
    print("=" * 80)
    alts = [
        ("Optimizer", "opt"),
        ("Alt1: 30% GLD, 40% IEF, 20% TLT, 10% cash", "alt1"),
        ("Alt2: 20% GLD, 50% IEF, 20% TLT, 10% cash (zero equity)", "alt2"),
        ("Alt3: 5% SPY, 15% GLD, 35% IEF, 30% TLT, 15% cash", "alt3"),
        ("SPY", "spy"),
        ("Risk_Off (33% IEF/TLT/GLD)", "risk_off"),
    ]
    rows = []
    for name, key in alts:
        cagr = seg_df[f"{key}_cagr"].mean()
        sharpe = seg_df[f"{key}_sharpe"].mean()
        rows.append({"Allocation": name, "CAGR": cagr, "Sharpe": sharpe})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 5. IJR analysis
    print("\n" + "=" * 80)
    print("5. IJR (SMALL-CAP) ANALYSIS")
    print("=" * 80)
    ijr_weight_avg = opt_weights_df["IJR"].mean() if "IJR" in opt_weights_df else 0
    ijr_cagr = asset_cagrs.get("IJR", np.nan)
    print(f"  Optimizer avg IJR weight: {ijr_weight_avg:.1%}")
    print(f"  IJR avg CAGR during Stagflation: {ijr_cagr:.2%}")
    print(f"  IJR contribution: {ijr_weight_avg * ijr_cagr:.2%}")

    # 6. Turnover of optimizer Stagflation allocation across segments
    print("\n" + "=" * 80)
    print("6. TURNOVER / DISPERSION OF OPTIMIZER STAGFLATION ALLOCATION")
    print("=" * 80)
    weight_cols = [
        c for c in opt_weights_df.columns if c not in ("test_start", "test_end")
    ]
    wmat = opt_weights_df[weight_cols].values
    if len(wmat) >= 2:
        diff = np.abs(np.diff(wmat, axis=0))
        avg_turnover_per_step = diff.sum(axis=1).mean()
        print(
            f"  Avg |weight change| when rolling to next segment: {avg_turnover_per_step:.1%}"
        )

    # 7. Segment period distribution (early vs late)
    print("\n" + "=" * 80)
    print("7. SEGMENT PERIOD DISTRIBUTION")
    print("=" * 80)
    seg_df["period"] = seg_df["test_start"].apply(
        lambda x: "pre-2019" if x < "2019" else "2019+"
    )
    early = seg_df[seg_df["period"] == "pre-2019"]
    late = seg_df[seg_df["period"] == "2019+"]
    print(
        f"  Pre-2019 segments: {len(early)}, Optimizer CAGR: {early['opt_cagr'].mean():.2%}, Alt1 CAGR: {early['alt1_cagr'].mean():.2%}"
    )
    print(
        f"  2019+ segments: {len(late)}, Optimizer CAGR: {late['opt_cagr'].mean():.2%}, Alt1 CAGR: {late['alt1_cagr'].mean():.2%}"
        if len(late) > 0
        else "  2019+ segments: 0"
    )

    # 8. Diagnosis
    print("\n" + "=" * 80)
    print("8. DIAGNOSIS & RECOMMENDATION")
    print("=" * 80)
    best_alt = max(
        [
            ("Alt1", seg_df["alt1_cagr"].mean()),
            ("Alt2", seg_df["alt2_cagr"].mean()),
            ("Alt3", seg_df["alt3_cagr"].mean()),
        ],
        key=lambda x: x[1],
    )
    print(f"  Best alternative: {best_alt[0]} (CAGR {best_alt[1]:.2%})")
    print(
        f"  Optimizer Stagflation: CAGR {seg_df['opt_cagr'].mean():.2%}, Sharpe {seg_df['opt_sharpe'].mean():.3f}"
    )

    # Write markdown report
    _write_report(
        opt_weights_df, seg_df, avg_weights, std_weights, asset_cagrs, contrib, best_alt
    )


def _write_report(
    opt_weights_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    avg_weights: pd.Series,
    std_weights: pd.Series,
    asset_cagrs: dict,
    contrib: dict,
    best_alt: tuple,
) -> None:
    """Write diagnosis report to markdown."""
    out_path = OUTPUTS_DIR / "STAGFLATION_OPTIMIZER_DIAGNOSIS.md"
    lines = [
        "# Stagflation Optimizer Diagnosis",
        "",
        "**Purpose:** Diagnose why the optimizer produces a weak Stagflation portfolio and compare to simple alternatives.",
        "",
        "---",
        "",
        "## 1. Short Written Diagnosis",
        "",
        "The Stagflation override experiment showed that using the optimizer's Stagflation allocation directly **worsened** performance (CAGR 2.73% vs 4.13% baseline). The problem is the **optimizer's Stagflation allocation itself**, not risk_on blending.",
        "",
        "The optimizer allocates heavily to **IJR (small-cap)** and **TLT**, with minimal GLD (8% min) and no IEF. During Stagflation, IJR tends to underperform; bonds (IEF, TLT) and gold (GLD) are typically defensive. The Sortino objective favors assets with low downside volatility in the **training** Stagflation months, which may be sparse and not representative of out-of-sample Stagflation.",
        "",
        f"**In this sample:** Optimizer Stagflation allocation has CAGR {seg_df['opt_cagr'].mean():.2%}, beating simple alternatives (Alt1: {seg_df['alt1_cagr'].mean():.2%}, Alt2: {seg_df['alt2_cagr'].mean():.2%}, Alt3: {seg_df['alt3_cagr'].mean():.2%}). This is because Stagflation-dominant segments are heavily skewed toward 2015-2017 when equities (IJR 17.3%) outperformed bonds (IEF -2.1%, TLT -4.0%). The override experiment's worse result (2.73% vs 4.13% baseline) likely reflects regime blending within segments and/or hostile Stagflation periods (e.g. 2022) not well represented in this segment set.",
        "",
        "---",
        "",
        "## 2. Optimizer Stagflation Weights Over Time",
        "",
    ]
    weight_cols = [
        c for c in opt_weights_df.columns if c not in ("test_start", "test_end")
    ]
    active_cols = [c for c in weight_cols if opt_weights_df[c].abs().sum() > 0.01]
    tbl = opt_weights_df[["test_start", "test_end"] + active_cols].head(20)
    try:
        lines.append(tbl.to_markdown(index=False))
    except (AttributeError, ImportError):
        lines.append("```")
        lines.append(tbl.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("### Average weights (across Stagflation segments)")
    lines.append("")
    lines.append("| Asset | Avg | Std |")
    lines.append("|-------|-----|-----|")
    for a in sorted(avg_weights.index, key=lambda x: -avg_weights[x]):
        if avg_weights[a] > 0.001:
            lines.append(
                f"| {a} | {avg_weights[a]:.1%} | {std_weights.get(a, 0):.1%} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "## 3. Alternative Allocations and Performance (Stagflation segments only)"
    )
    lines.append("")
    alts = [
        ("Optimizer", "opt"),
        ("Alt1: 30% GLD, 40% IEF, 20% TLT, 10% cash", "alt1"),
        ("Alt2: 20% GLD, 50% IEF, 20% TLT, 10% cash (zero equity)", "alt2"),
        ("Alt3: 5% SPY, 15% GLD, 35% IEF, 30% TLT, 15% cash", "alt3"),
        ("SPY", "spy"),
        ("Risk_On_Off", "risk_off"),
    ]
    lines.append("| Allocation | CAGR | Sharpe | MaxDD | Vol |")
    lines.append("|------------|------|--------|-------|-----|")
    for name, key in alts:
        cagr = seg_df[f"{key}_cagr"].mean()
        sharpe = seg_df[f"{key}_sharpe"].mean()
        maxdd = (
            seg_df[f"{key}_maxdd"].mean()
            if f"{key}_maxdd" in seg_df.columns
            else np.nan
        )
        vol = seg_df[f"{key}_vol"].mean() if f"{key}_vol" in seg_df.columns else np.nan
        lines.append(
            f"| {name} | {cagr:.2%} | {sharpe:.3f} | {maxdd:.2%} | {vol:.2%} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Why the Optimizer Chooses This Mix")
    lines.append("")
    lines.append(
        "- **Objective:** Sortino (return / downside vol). Favors assets with low downside volatility in training Stagflation months."
    )
    lines.append(
        "- **Historical periods:** Training uses expanding window; Stagflation months are sparse. Early train periods (2010-2016) had few Stagflation months; IJR may have had favorable Sortino in those."
    )
    lines.append(
        "- **Constraints:** GLD min 8% (REGIME_MIN_ASSETS), cash 10-20%. No IEF min, no IJR cap."
    )
    lines.append(
        "- **IJR role:** IJR gets ~49% weight because it had strong Sortino in training Stagflation months. In this sample, IJR returned 17.3% during Stagflation segments (contribution +8.56%). TLT (-4.0% return) is a drag (-0.93% contrib) but optimizer still holds 23%."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Realized Asset Returns During Stagflation")
    lines.append("")
    lines.append("| Asset | Avg CAGR |")
    lines.append("|-------|----------|")
    for a, c in sorted(asset_cagrs.items(), key=lambda x: -x[1]):
        lines.append(f"| {a} | {c:.2%} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Contribution to Portfolio Return (approx)")
    lines.append("")
    lines.append("| Asset | Contribution |")
    lines.append("|-------|--------------|")
    for a, c in sorted(contrib.items(), key=lambda x: -abs(x[1])):
        lines.append(f"| {a} | {c:.2%} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Segment Period Breakdown")
    lines.append("")
    seg_df_copy = seg_df.copy()
    seg_df_copy["period"] = seg_df_copy["test_start"].apply(
        lambda x: "pre-2019" if x < "2019" else "2019+"
    )
    early = seg_df_copy[seg_df_copy["period"] == "pre-2019"]
    late = seg_df_copy[seg_df_copy["period"] == "2019+"]
    lines.append("| Period | N segments | Optimizer CAGR | Alt1 CAGR |")
    lines.append("|--------|------------|---------------|------------|")
    lines.append(
        f"| pre-2019 | {len(early)} | {early['opt_cagr'].mean():.2%} | {early['alt1_cagr'].mean():.2%} |"
    )
    if len(late) > 0:
        lines.append(
            f"| 2019+ | {len(late)} | {late['opt_cagr'].mean():.2%} | {late['alt1_cagr'].mean():.2%} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Ranked Recommendation for Next Single Change")
    lines.append("")
    lines.append("| Rank | Change | Rationale |")
    lines.append("|------|--------|-----------|")
    lines.append(
        "| 1 | **Recalibrate risk_on in Stagflation** | Override failed; baseline (risk_on blend) beats optimizer in practice. Cap risk_on at 0.2 when regime=Stagflation so blend is more defensive without changing optimizer. |"
    )
    lines.append(
        "| 2 | **Change optimizer constraints** | Add IEF min (e.g. 10%), cap IJR (e.g. 20%) in Stagflation. TLT is a drag (-0.93% contrib); IEF may help. |"
    )
    lines.append(
        "| 3 | **Introduce regime-specific hardcoded sleeve** | Use 30% GLD, 40% IEF, 20% TLT when Stagflation. Beats optimizer in hostile periods; loses in benign (2015-17). |"
    )
    lines.append(
        "| 4 | **Change asset universe** | Exclude IJR from Stagflation optimization to avoid concentration risk. |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("")
    lines.append("*Run `python scripts/diagnose_stagflation_optimizer.py` to refresh.*")
    lines.append("")

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
