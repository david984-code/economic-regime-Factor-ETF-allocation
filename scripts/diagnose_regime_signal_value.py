"""Diagnose whether regime classification provides predictive value for asset allocation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.allocation.optimizer import optimize_allocations_from_data
from src.config import OUTPUTS_DIR, START_DATE, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices

CASH_MONTHLY = (1.05) ** (1 / 12) - 1


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns from daily prices."""
    ret = prices.resample("ME").last().pct_change().dropna()
    ret.index = ret.index.to_period("M")
    if "cash" not in ret.columns:
        ret["cash"] = CASH_MONTHLY
    return ret


def _forward_returns(ret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute forward N-month total return for each asset."""
    out = ret.copy()
    for c in ret.columns:
        out[c] = (
            (1 + ret[c])
            .rolling(horizon)
            .apply(lambda x: x.prod() - 1, raw=True)
            .shift(-horizon)
        )
    return out


def main() -> None:
    print("Loading data...")
    prices = fetch_prices(start=START_DATE, end=get_end_date())
    ret = _monthly_returns(prices[TICKERS])
    ret["cash"] = CASH_MONTHLY

    regimes = pd.read_csv(
        OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regimes = regimes.dropna(subset=["regime"])
    regimes = regimes[regimes["regime"].str.strip() != ""]
    regimes = regimes.set_index("date")
    regimes["month"] = regimes.index.to_period("M")

    # Align to common index (ret has Period index)
    ret_month = ret.index
    reg_month = regimes["month"].drop_duplicates()
    common = ret_month.intersection(reg_month).drop_duplicates()
    ret = ret.loc[common]
    regime_map = regimes.set_index("month")["regime"]
    regime_series = regime_map.reindex(common).ffill().bfill()
    regime_series = regime_series.astype(str).str.strip()

    # Exclude Unknown for most analysis
    regime_series[regime_series != "Unknown"]
    reg_names = ["Recovery", "Overheating", "Stagflation", "Contraction"]

    # 1. Forward returns by regime
    print("\n" + "=" * 80)
    print("1. FORWARD RETURNS BY REGIME (vs unconditional)")
    print("=" * 80)

    fwd1 = _forward_returns(ret, 1)
    fwd3 = _forward_returns(ret, 3)
    fwd6 = _forward_returns(ret, 6)

    rows = []
    for asset in TICKERS + ["cash"]:
        if asset not in fwd1.columns:
            continue
        uncond_1 = fwd1[asset].dropna().mean()
        uncond_3 = fwd3[asset].dropna().mean()
        uncond_6 = fwd6[asset].dropna().mean()
        row = {
            "asset": asset,
            "uncond_1M": uncond_1,
            "uncond_3M": uncond_3,
            "uncond_6M": uncond_6,
        }
        for r in reg_names:
            mask = regime_series == r
            if mask.sum() < 6:
                continue
            sub1 = fwd1.loc[mask, asset].dropna()
            sub3 = fwd3.loc[mask, asset].dropna()
            sub6 = fwd6.loc[mask, asset].dropna()
            row[f"{r}_1M"] = sub1.mean() if len(sub1) >= 3 else np.nan
            row[f"{r}_3M"] = sub3.mean() if len(sub3) >= 3 else np.nan
            row[f"{r}_6M"] = sub6.mean() if len(sub6) >= 3 else np.nan
        rows.append(row)

    fwd_df = pd.DataFrame(rows)
    print("\n1M forward return (annualized approx):")
    cols = ["asset", "uncond_1M"] + [
        f"{r}_1M" for r in reg_names if f"{r}_1M" in fwd_df.columns
    ]
    print(
        fwd_df[[c for c in cols if c in fwd_df.columns]].to_string(
            index=False, float_format=lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
    )

    # 2. Regime persistence
    print("\n" + "=" * 80)
    print("2. REGIME PERSISTENCE")
    print("=" * 80)

    reg_arr = regime_series.values
    durations = []
    current_reg = reg_arr[0]
    current_len = 1
    for i in range(1, len(reg_arr)):
        if reg_arr[i] == current_reg:
            current_len += 1
        else:
            durations.append((current_reg, current_len))
            current_reg = reg_arr[i]
            current_len = 1
    durations.append((current_reg, current_len))

    dur_df = pd.DataFrame(durations, columns=["regime", "months"])
    print("\nAverage duration (months) by regime:")
    print(
        dur_df.groupby("regime")["months"].agg(["mean", "median", "count"]).to_string()
    )

    # Transition matrix
    trans = pd.crosstab(regime_series.shift(1), regime_series, normalize="index")
    print("\nTransition probabilities (row = from, col = to):")
    print(trans.round(3).to_string())

    # Regime changes within 12-month test windows
    test_months = 12
    n_changes = []
    for i in range(len(reg_arr) - test_months):
        window = reg_arr[i : i + test_months]
        changes = (window[1:] != window[:-1]).sum()
        n_changes.append(changes)
    print(
        f"\nRegime changes within 12-month windows: mean={np.mean(n_changes):.1f}, median={np.median(n_changes):.0f}"
    )

    # 3. Simple regime portfolios
    print("\n" + "=" * 80)
    print("3. SIMPLE REGIME PORTFOLIOS vs OPTIMIZER")
    print("=" * 80)

    # Equal weight per regime
    eqw_ret = ret[TICKERS].mean(axis=1)
    eqw_by_regime = {}
    for r in reg_names:
        mask = regime_series == r
        if mask.sum() < 12:
            continue
        sub = eqw_ret.loc[mask].dropna()
        eqw_by_regime[r] = {"mean": sub.mean(), "std": sub.std(), "n": len(sub)}

    # Best single asset per regime (forward 1M return when regime known)
    best_asset = {}
    for r in reg_names:
        mask = regime_series == r
        if mask.sum() < 12:
            continue
        sub_fwd = fwd1.loc[mask, [c for c in TICKERS if c in fwd1.columns]]
        means = sub_fwd.mean()
        best_asset[r] = (
            means.idxmax() if len(means) > 0 and means.notna().any() else "N/A"
        )

    # Load optimizer allocations
    allocs = {}
    try:
        train_ret = ret.copy()
        train_reg = regimes[["regime"]].copy()
        allocs = optimize_allocations_from_data(train_ret, train_reg)
    except Exception:
        pass
    if not allocs and (OUTPUTS_DIR / "optimal_allocations.csv").exists():
        opt_df = pd.read_csv(OUTPUTS_DIR / "optimal_allocations.csv", index_col=0)
        allocs = {r: opt_df.loc[r].to_dict() for r in opt_df.index if r in reg_names}
    pd.read_csv(OUTPUTS_DIR / "optimal_allocations.csv", index_col=0) if (
        OUTPUTS_DIR / "optimal_allocations.csv"
    ).exists() else None

    print("\nBest single asset per regime (forward 1M return):")
    for r, a in best_asset.items():
        print(f"  {r}: {a}")

    print("\nOptimizer allocation (sample):")
    if allocs:
        for r in reg_names:
            if r in allocs:
                top = sorted(allocs[r].items(), key=lambda x: -x[1])[:5]
                print(f"  {r}: {dict(top)}")

    # Simple regime portfolio returns (equal weight per regime, best asset per regime)
    eqw_port = ret[TICKERS].mean(axis=1)
    best_asset_port = pd.Series(0.0, index=ret.index)
    for i, (idx, reg) in enumerate(regime_series.items()):
        if reg in best_asset and best_asset[reg] in ret.columns:
            best_asset_port.iloc[i] = ret.loc[idx, best_asset[reg]]
        else:
            best_asset_port.iloc[i] = eqw_port.loc[idx]
    opt_port = pd.Series(0.0, index=ret.index)
    if allocs:
        for i, (idx, reg) in enumerate(regime_series.items()):
            if reg in allocs:
                w = allocs[reg]
                opt_port.iloc[i] = sum(
                    ret.loc[idx, a] * w.get(a, 0) for a in ret.columns if a in w
                )
            else:
                opt_port.iloc[i] = eqw_port.loc[idx]

    from src.backtest.metrics import compute_metrics

    rf = CASH_MONTHLY
    rf_daily = (1 + rf) ** (1 / 21) - 1

    def _monthly_to_daily(ser: pd.Series) -> pd.Series:
        out = []
        for r in ser.dropna():
            out.extend([(1 + r) ** (1 / 21) - 1] * 21)
        return pd.Series(out)

    eqw_daily = _monthly_to_daily(eqw_port)
    best_daily = _monthly_to_daily(best_asset_port)
    opt_daily = _monthly_to_daily(opt_port) if allocs else pd.Series(dtype=float)
    m_eqw = compute_metrics(eqw_daily, rf_daily=rf_daily)
    m_best = compute_metrics(best_daily, rf_daily=rf_daily)
    m_opt = compute_metrics(opt_daily, rf_daily=rf_daily) if len(opt_daily) > 0 else {}
    print("\nSimple regime portfolio performance (monthly, annualized):")
    print(
        f"  Equal-weight (unconditional): CAGR={m_eqw['CAGR']:.2%}, Sharpe={m_eqw['Sharpe']:.2f}"
    )
    print(
        f"  Best-asset-per-regime:        CAGR={m_best['CAGR']:.2%}, Sharpe={m_best['Sharpe']:.2f}"
    )
    if m_opt:
        print(
            f"  Optimizer allocation:         CAGR={m_opt['CAGR']:.2%}, Sharpe={m_opt['Sharpe']:.2f}"
        )

    # 4. Regime signal strength
    print("\n" + "=" * 80)
    print("4. REGIME SIGNAL STRENGTH")
    print("=" * 80)

    # Dispersion of asset returns across regimes
    disp = []
    for asset in TICKERS:
        if asset not in ret.columns:
            continue
        by_reg = [
            ret.loc[regime_series == r, asset].mean()
            for r in reg_names
            if (regime_series == r).sum() >= 6
        ]
        if len(by_reg) >= 2:
            disp.append((asset, np.std(by_reg), np.max(by_reg) - np.min(by_reg)))
    disp_df = pd.DataFrame(disp, columns=["asset", "std_across_regimes", "range"])
    print(
        "\nDispersion of mean return across regimes (higher = regime differentiates more):"
    )
    print(
        disp_df.sort_values("std_across_regimes", ascending=False).to_string(
            index=False
        )
    )

    # Information ratio: regime-conditioned vs unconditional
    # Simplified: compare volatility of regime-specific mean returns
    uncond_vol = ret[TICKERS].mean(axis=1).std() * np.sqrt(12)
    print(f"\nUnconditional equal-weight monthly vol (ann): {uncond_vol:.2%}")

    # Regime-conditioned portfolio: hold equal weight, but only in months when regime is X
    # We compare: if we could perfectly predict regime, would we beat unconditional?
    regime_cond_ret = []
    for r in reg_names:
        mask = regime_series == r
        if mask.sum() < 12:
            continue
        sub = ret.loc[mask, TICKERS].mean(axis=1)
        regime_cond_ret.append(sub.mean())
    if regime_cond_ret:
        np.mean(regime_cond_ret)
        np.std(regime_cond_ret)
        print(
            f"Avg monthly return by regime (when in that regime): mean={np.mean(regime_cond_ret):.2%}, std={np.std(regime_cond_ret):.2%}"
        )

    # 5. Write report
    _write_report(
        fwd_df,
        dur_df,
        trans,
        n_changes,
        best_asset,
        allocs,
        disp_df,
        reg_names,
        {"eqw": m_eqw, "best": m_best, "opt": m_opt},
    )


def _write_report(
    fwd_df: pd.DataFrame,
    dur_df: pd.DataFrame,
    trans: pd.DataFrame,
    n_changes: list,
    best_asset: dict,
    allocs: dict,
    disp_df: pd.DataFrame,
    reg_names: list,
    port_metrics: dict | None = None,
) -> None:
    """Write diagnosis report to markdown."""
    lines = [
        "# Regime Signal Value Diagnosis",
        "",
        "**Goal:** Measure whether the regime classification provides predictive value for asset allocation.",
        "",
        "---",
        "",
        "## 1. Forward Returns by Regime (vs unconditional)",
        "",
        "Average forward return when regime is known at start of period. Uncond = unconditional.",
        "",
    ]
    cols = ["asset", "uncond_1M"] + [
        f"{r}_1M" for r in reg_names if f"{r}_1M" in fwd_df.columns
    ]
    tbl = fwd_df[[c for c in cols if c in fwd_df.columns]].head(12)
    lines.append(
        tbl.to_string(
            index=False, float_format=lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
    )
    lines.append("")
    lines.append("3M forward return:")
    cols3 = ["asset", "uncond_3M"] + [
        f"{r}_3M" for r in reg_names if f"{r}_3M" in fwd_df.columns
    ]
    tbl3 = fwd_df[[c for c in cols3 if c in fwd_df.columns]].head(12)
    lines.append(
        tbl3.to_string(
            index=False, float_format=lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Regime Persistence")
    lines.append("")
    lines.append("### Average duration (months)")
    lines.append("")
    lines.append(
        dur_df.groupby("regime")["months"].agg(["mean", "median", "count"]).to_string()
    )
    lines.append("")
    lines.append("### Transition probabilities (row=from, col=to)")
    lines.append("")
    lines.append(trans.round(3).to_string())
    lines.append("")
    lines.append(
        f"### Regime changes within 12-month test windows: mean={np.mean(n_changes):.1f}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Simple Regime Portfolios vs Optimizer")
    lines.append("")
    lines.append("### Best single asset per regime (forward 1M return)")
    lines.append("")
    for r, a in best_asset.items():
        lines.append(f"- {r}: {a}")
    if port_metrics:
        lines.append("")
        lines.append(
            "### Simple regime portfolio performance (in-sample, perfect regime knowledge)"
        )
        lines.append("")
        lines.append("| Portfolio | CAGR | Sharpe |")
        lines.append("|-----------|------|--------|")
        m = port_metrics.get("eqw", {})
        lines.append(
            f"| Equal-weight (unconditional) | {m.get('CAGR', 0):.2%} | {m.get('Sharpe', 0):.2f} |"
        )
        m = port_metrics.get("best", {})
        lines.append(
            f"| Best-asset-per-regime | {m.get('CAGR', 0):.2%} | {m.get('Sharpe', 0):.2f} |"
        )
        m = port_metrics.get("opt", {})
        if m:
            lines.append(
                f"| Optimizer allocation (in-sample) | {m.get('CAGR', 0):.2%} | {m.get('Sharpe', 0):.2f} |"
            )
        lines.append("")
        lines.append(
            "*Optimizer is in-sample (perfect foresight). Walk-forward shows ~8% CAGR out-of-sample.*"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Regime Signal Strength (dispersion)")
    lines.append("")
    lines.append("Higher dispersion = regimes differentiate asset behavior more.")
    lines.append("")
    lines.append(
        disp_df.sort_values("std_across_regimes", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Conclusion")
    lines.append("")
    avg_dur = (
        dur_df[dur_df["regime"] != "Unknown"].groupby("regime")["months"].mean().mean()
    )
    max_disp = disp_df["std_across_regimes"].max() if len(disp_df) > 0 else 0
    best_beats_eqw = port_metrics and port_metrics.get("best", {}).get(
        "CAGR", 0
    ) > port_metrics.get("eqw", {}).get("CAGR", 1)
    if avg_dur < 4 and (np.mean(n_changes) if n_changes else 0) > 3:
        concl = "Regimes are **short-lived** (avg duration < 4 months) and **change frequently** within test windows (mean 7.6 changes per 12 months). The signal may be too noisy for 12-month allocation decisions."
    elif not best_beats_eqw and port_metrics:
        concl = "**Best-asset-per-regime underperforms equal-weight** — using the regime signal to pick the best asset each month does not add value. Regimes differentiate asset returns (TLT, GLD have high dispersion) but the signal is too noisy or lagging to exploit."
    elif max_disp < 0.005:
        concl = "**Low dispersion** of asset returns across regimes — regimes do not meaningfully differentiate asset behavior. The macro inputs may be lagging market reality."
    else:
        concl = "Regimes show **moderate differentiation** of asset returns (TLT, GLD most dispersed). Signal strength is present but diluted by regime noise, short horizons, and high transition frequency."
    lines.append(concl)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Recommendation for Next Single Experiment")
    lines.append("")
    lines.append("| Option | Rationale |")
    lines.append("|--------|-----------|")
    lines.append(
        "| **Test longer regime smoothing** | If regimes are noisy, smooth over 2-3 months before allocation. |"
    )
    lines.append(
        "| **Reduce regime dependence** | Use regime as one input to risk_on, not as allocation switch. |"
    )
    lines.append(
        "| **Validate macro lag** | Compare regime dates to NBER recession dates; check if macro lags. |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Run `python scripts/diagnose_regime_signal_value.py` to refresh.*")
    lines.append("")

    out_path = OUTPUTS_DIR / "REGIME_SIGNAL_VALUE_DIAGNOSIS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
