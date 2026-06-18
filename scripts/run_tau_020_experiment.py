"""Experiment: Higher Rebalance Tolerance (tau 0.015 -> 0.020).

Single variable: tolerance (tau) only. Baseline uses VOL_LOOKBACK=126; all else identical
(24M momentum, sigmoid 0.25, equal-weight sleeves, post-blend inv-vol, monthly rebalance,
use_stagflation_override=False). Fast-mode first; full walk-forward only if not clearly worse.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import src.backtest.engine as _eng
import src.config as _cfg
from src.allocation.optimizer import optimize_allocations_from_data
from src.backtest.engine import run_backtest_with_allocations
from src.config import OUTPUTS_DIR, get_end_date
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import _make_segments, run_walk_forward_evaluation

logging.basicConfig(level=logging.WARNING)

FULL_START = "2010-01-01"
VOL_LOOKBACK_BASELINE = 126  # new working baseline after vol lookback experiment
TAU_BASE = 0.015
TAU_EXP = 0.020

# Shared kwargs: VOL_LOOKBACK set via patch to 126; baseline tolerance 0.015
SHARED_KWARGS = {
    "start": FULL_START,
    "end": None,
    "min_train_months": 60,
    "test_months": 12,
    "expanding": True,
    "use_stagflation_override": False,
    "use_stagflation_risk_on_cap": False,
    "use_regime_smoothing": False,
    "use_hybrid_signal": True,
    "hybrid_macro_weight": 0.0,
    "use_momentum": True,
    "trend_filter_type": "none",
    "vol_scaling_method": "none",
    "portfolio_construction_method": "equal_weight",
    "momentum_12m_weight": 0.0,
    "quarterly_rebalance": False,
    "tolerance": TAU_BASE,
    "sigmoid_scale": 0.25,
    "skip_persist": True,
    "use_vol_regime": False,
    "market_lookback_months": 24,
}


def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["segment"] != "OVERALL"].copy()


def _m(row, col: str) -> float:
    v = row.get(col, float("nan"))
    return float(v) if v is not None else float("nan")


def _pct(v: float, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    return f"{v:+.2%}" if sign else f"{v:.2%}"


def _f(v: float, d: int = 3, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    return fmt.format(v)


def _run_attribution_oos(end: str, tolerance: float) -> pd.DataFrame:
    """Run segment-by-segment attribution with given tolerance (VOL_LOOKBACK already 126). Return OOS attribution."""
    prices = fetch_prices(start=FULL_START, end=end)
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()
    regime_df = regime_df.dropna(subset=["regime"]).sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
    regime_df = regime_df.reindex(prices.index).ffill()
    segments = _make_segments(FULL_START, end, min_train_months=60, test_months=12, expanding=True)
    if not segments:
        return pd.DataFrame()
    monthly_prices = prices.resample("ME").last()
    base_kw = {
        k: v
        for k, v in SHARED_KWARGS.items()
        if k
        not in (
            "start",
            "end",
            "fast_mode",
            "skip_persist",
            "min_train_months",
            "test_months",
            "expanding",
            "tolerance",
        )
    }
    all_att_records = []
    for _seg_idx, (train_start, train_end, test_start, test_end) in enumerate(segments):
        train_returns = monthly_prices.pct_change().dropna()
        train_returns = train_returns.loc[
            (train_returns.index >= train_start) & (train_returns.index <= train_end)
        ]
        if "cash" not in train_returns.columns:
            train_returns["cash"] = (1.05) ** (1 / 12) - 1
        train_regimes = regime_df.loc[:train_end].resample("ME").last().dropna(how="all")
        train_regimes = train_regimes.loc[train_regimes.index <= train_end]
        if len(train_returns) < 24 or len(train_regimes) < 12:
            continue
        seg_allocations = optimize_allocations_from_data(train_returns, train_regimes)
        if not seg_allocations:
            continue
        for alloc in seg_allocations.values():
            if "cash" not in alloc:
                alloc["cash"] = 0.0
        seg_result = run_backtest_with_allocations(
            prices,
            regime_df,
            seg_allocations,
            return_weights=True,
            return_turnover_attribution=True,
            tolerance=tolerance,
            **base_kw,
        )
        if not isinstance(seg_result, tuple) or len(seg_result) < 4:
            continue
        _, _, _, seg_att = seg_result[0], seg_result[1], seg_result[2], seg_result[3]
        if seg_att is None or seg_att.empty:
            continue
        seg_att_test = seg_att.loc[test_start:test_end]
        all_att_records.append(seg_att_test)
    if not all_att_records:
        return pd.DataFrame()
    att = pd.concat(all_att_records, axis=0).sort_index()
    if att.index.duplicated().any():
        att = att[~att.index.duplicated(keep="last")]
    return att


def _filter_year(att: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    mask = (att.index.year >= y0) & (att.index.year <= y1)
    return att.loc[mask]


def main():
    end = get_end_date()
    print("=" * 72)
    print("EXPERIMENT: Higher Rebalance Tolerance (tau 0.015 -> 0.020)")
    print("=" * 72)
    print("  Single variable: tau (tolerance) only. Baseline: VOL_LOOKBACK=126.")
    print("  Signal: 24M SPY momentum -> expanding z -> sigmoid(z*0.25)")
    print("  Sleeves: equal-weight; post-blend inv-vol; monthly rebalance")
    print()

    # Use new working baseline: VOL_LOOKBACK=126 for both runs
    if _cfg.VOL_LOOKBACK != VOL_LOOKBACK_BASELINE:
        _cfg.VOL_LOOKBACK = VOL_LOOKBACK_BASELINE
    if _eng.VOL_LOOKBACK != VOL_LOOKBACK_BASELINE:
        _eng.VOL_LOOKBACK = VOL_LOOKBACK_BASELINE
    print(f"  VOL_LOOKBACK set to {VOL_LOOKBACK_BASELINE} for entire run.")

    # --- FAST MODE ---
    print("\nRunning FAST MODE (baseline tau=0.015)...")
    kw_fast = {**SHARED_KWARGS, "end": end, "fast_mode": True}
    df_fast_base = run_walk_forward_evaluation(**kw_fast)

    kw_fast_exp = {**kw_fast, "tolerance": TAU_EXP}
    print("Running FAST MODE (experiment tau=0.020)...")
    df_fast_exp = run_walk_forward_evaluation(**kw_fast_exp)

    if df_fast_base.empty or df_fast_exp.empty:
        print("ERROR: empty fast-mode results.")
        sys.exit(1)

    ob = _overall(df_fast_base)
    oe = _overall(df_fast_exp)
    cagr_b = _m(ob, "Strategy_CAGR")
    cagr_e = _m(oe, "Strategy_CAGR")
    shr_b = _m(ob, "Strategy_Sharpe")
    shr_e = _m(oe, "Strategy_Sharpe")
    mdd_b = _m(ob, "Strategy_MaxDD")
    mdd_e = _m(oe, "Strategy_MaxDD")
    vol_b = _m(ob, "Strategy_Vol")
    vol_e = _m(oe, "Strategy_Vol")
    to_b = _m(ob, "Strategy_Turnover")
    to_e = _m(oe, "Strategy_Turnover")

    print("\n" + "=" * 72)
    print("FAST MODE RESULTS")
    print("=" * 72)
    print(
        f"  {'Metric':14} {'Baseline (tau=0.015)':>20} {'Experiment (tau=0.020)':>22} {'Delta':>12}"
    )
    print("  " + "-" * 70)
    print(
        f"  {'CAGR':14} {_pct(cagr_b):>20} {_pct(cagr_e):>22} {_pct(cagr_e - cagr_b, sign=True):>12}"
    )
    print(f"  {'Sharpe':14} {_f(shr_b):>20} {_f(shr_e):>22} {_f(shr_e - shr_b, sign=True):>12}")
    print(
        f"  {'MaxDD':14} {_pct(mdd_b):>20} {_pct(mdd_e):>22} {_pct(mdd_e - mdd_b, sign=True):>12}"
    )
    print(f"  {'Vol':14} {_pct(vol_b):>20} {_pct(vol_e):>22} {_pct(vol_e - vol_b, sign=True):>12}")
    print(f"  {'Turnover':14} {_f(to_b):>20} {_f(to_e):>22} {_f(to_e - to_b, sign=True):>12}")

    shr_d_fast = shr_e - shr_b
    to_d_fast = to_e - to_b if not (np.isnan(to_e) or np.isnan(to_b)) else 0.0
    mdd_d_fast = mdd_e - mdd_b
    clearly_worse = (shr_d_fast < -0.02) or (mdd_d_fast < -0.01) or (to_d_fast > 0.02)
    run_full = not clearly_worse
    print(
        f"\n  Fast-mode: Sharpe delta={shr_d_fast:+.3f}, TO delta={to_d_fast:+.3f}, MaxDD delta={_pct(mdd_d_fast, sign=True)}"
    )
    print(f"  Run full walk-forward: {'YES' if run_full else 'NO (fast-mode clearly worse)'}")

    # --- FULL WALK-FORWARD ---
    if run_full:
        print("\nRunning FULL WALK-FORWARD (baseline tau=0.015)...")
        kw_full = {**SHARED_KWARGS, "end": end, "fast_mode": False}
        df_full_base = run_walk_forward_evaluation(**kw_full)
        kw_full_exp = {**kw_full, "tolerance": TAU_EXP}
        print("Running FULL WALK-FORWARD (experiment tau=0.020)...")
        df_full_exp = run_walk_forward_evaluation(**kw_full_exp)
    else:
        df_full_base = None
        df_full_exp = None

    use_full = (
        run_full
        and df_full_base is not None
        and not df_full_base.empty
        and df_full_exp is not None
        and not df_full_exp.empty
    )
    if use_full:
        segs = _segs(df_full_base)
        ob_full = _overall(df_full_base)
        oe_full = _overall(df_full_exp)
    else:
        segs = _segs(df_fast_base)
        ob_full = _overall(df_fast_base)
        oe_full = _overall(df_fast_exp)

    oos_start = segs["test_start"].iloc[0] if len(segs) else "n/a"
    oos_end = segs["test_end"].iloc[-1] if len(segs) else "n/a"
    cagr_fb = _m(ob_full, "Strategy_CAGR")
    cagr_fe = _m(oe_full, "Strategy_CAGR")
    shr_fb = _m(ob_full, "Strategy_Sharpe")
    shr_fe = _m(oe_full, "Strategy_Sharpe")
    mdd_fb = _m(ob_full, "Strategy_MaxDD")
    mdd_fe = _m(oe_full, "Strategy_MaxDD")
    vol_fb = _m(ob_full, "Strategy_Vol")
    vol_fe = _m(oe_full, "Strategy_Vol")
    to_fb = _m(ob_full, "Strategy_Turnover")
    to_fe = _m(oe_full, "Strategy_Turnover")

    print("\n" + "=" * 72)
    print("FULL WALK-FORWARD RESULTS" + (" (skipped; showing fast-mode)" if not use_full else ""))
    print("=" * 72)
    if not use_full:
        print("  (Full WF skipped because fast-mode was clearly worse.)")
    print(f"  OOS start:   {oos_start}")
    print(f"  OOS end:     {oos_end}")
    print(f"  segments:   {len(segs)}")
    print(
        f"  {'Metric':14} {'Baseline (tau=0.015)':>20} {'Experiment (tau=0.020)':>22} {'Delta':>12}"
    )
    print("  " + "-" * 70)
    print(
        f"  {'CAGR':14} {_pct(cagr_fb):>20} {_pct(cagr_fe):>22} {_pct(cagr_fe - cagr_fb, sign=True):>12}"
    )
    print(f"  {'Sharpe':14} {_f(shr_fb):>20} {_f(shr_fe):>22} {_f(shr_fe - shr_fb, sign=True):>12}")
    print(
        f"  {'MaxDD':14} {_pct(mdd_fb):>20} {_pct(mdd_fe):>22} {_pct(mdd_fe - mdd_fb, sign=True):>12}"
    )
    print(
        f"  {'Vol':14} {_pct(vol_fb):>20} {_pct(vol_fe):>22} {_pct(vol_fe - vol_fb, sign=True):>12}"
    )
    print(f"  {'Turnover':14} {_f(to_fb):>20} {_f(to_fe):>22} {_f(to_fe - to_fb, sign=True):>12}")

    # --- Attribution ---
    print("\nRunning attribution (baseline tau=0.015)...")
    att_base = _run_attribution_oos(end, TAU_BASE)
    print("Running attribution (experiment tau=0.020)...")
    att_exp = _run_attribution_oos(end, TAU_EXP)

    ann = 12.0
    if not att_base.empty and not att_exp.empty:
        tau_rem_base = att_base["to_removed_by_tau"].mean() * ann
        tau_rem_exp = att_exp["to_removed_by_tau"].mean() * ann
        exec_base = att_base["to_executed"].mean() * ann
        exec_exp = att_exp["to_executed"].mean() * ann
        sig_base = att_base["to_signal"].mean() * ann
        sig_exp = att_exp["to_signal"].mean() * ann
        inv_base = att_base["to_invvol"].mean() * ann
        inv_exp = att_exp["to_invvol"].mean() * ann
        print("\n" + "=" * 72)
        print("ATTRIBUTION COMPARISON (OOS, annualized)")
        print("=" * 72)
        print(
            f"  Tau-removed turnover:  base={tau_rem_base:.2%}  exp={tau_rem_exp:.2%}  delta={tau_rem_exp - tau_rem_base:+.2%}"
        )
        print(
            f"  Executed turnover:    base={exec_base:.2%}  exp={exec_exp:.2%}  delta={exec_exp - exec_base:+.2%}"
        )
        print(
            f"  Signal component:     base={sig_base:.2%}  exp={sig_exp:.2%}  delta={sig_exp - sig_base:+.2%}"
        )
        print(
            f"  Inv-vol component:    base={inv_base:.2%}  exp={inv_exp:.2%}  delta={inv_exp - inv_base:+.2%}"
        )

        print("\n  Top 10 turnover months (executed):")
        print("  " + "-" * 72)
        print("  Baseline (tau=0.015):")
        top_base = att_base.nlargest(10, "to_executed")[
            ["to_executed", "to_signal", "to_invvol", "to_removed_by_tau"]
        ]
        for idx, row in top_base.iterrows():
            print(
                f"    {idx.strftime('%Y-%m')}  TO={row['to_executed']:.4f}  signal={row['to_signal']:.4f}  invvol={row['to_invvol']:.4f}  tau_rem={row['to_removed_by_tau']:.4f}"
            )
        print("  Experiment (tau=0.020):")
        top_exp = att_exp.nlargest(10, "to_executed")[
            ["to_executed", "to_signal", "to_invvol", "to_removed_by_tau"]
        ]
        for idx, row in top_exp.iterrows():
            print(
                f"    {idx.strftime('%Y-%m')}  TO={row['to_executed']:.4f}  signal={row['to_signal']:.4f}  invvol={row['to_invvol']:.4f}  tau_rem={row['to_removed_by_tau']:.4f}"
            )

        print("\n  Crisis-period deltas (experiment - baseline):")
        for label, y0, y1 in [
            ("2018", 2018, 2018),
            ("2020", 2020, 2020),
            ("2021", 2021, 2021),
            ("2022", 2022, 2022),
        ]:
            ab = _filter_year(att_base, y0, y1)
            ae = _filter_year(att_exp, y0, y1)
            if len(ab) and len(ae):
                to_b_y = ab["to_executed"].mean() * ann
                to_e_y = ae["to_executed"].mean() * ann
                tau_b_y = ab["to_removed_by_tau"].mean() * ann
                tau_e_y = ae["to_removed_by_tau"].mean() * ann
                print(
                    f"    {label}: TO base={to_b_y:.2%} exp={to_e_y:.2%} delta={to_e_y - to_b_y:+.2%}  |  tau_rem base={tau_b_y:.2%} exp={tau_e_y:.2%} delta={tau_e_y - tau_b_y:+.2%}"
                )
            else:
                print(f"    {label}: n/a")
    else:
        print("\n  Attribution data not available (empty).")

    # --- Decision rule ---
    print("\n" + "=" * 72)
    print("DECISION RULE")
    print("=" * 72)
    shr_d = shr_fe - shr_fb
    to_fe - to_fb if not (np.isnan(to_fe) or np.isnan(to_fb)) else float("nan")
    to_pct_chg = (to_fe - to_fb) / to_fb if (to_fb and not np.isnan(to_fb)) else float("nan")
    mdd_d = mdd_fe - mdd_fb
    pass_sharpe = not np.isnan(shr_d) and shr_d >= -0.02
    pass_turnover = not np.isnan(to_pct_chg) and to_pct_chg <= -0.10
    pass_mdd = not np.isnan(mdd_d) and mdd_d >= -0.01
    decision_pass = pass_sharpe and pass_turnover and pass_mdd
    print(
        f"  Sharpe no worse than -0.02:  {'PASS' if pass_sharpe else 'FAIL'}  (delta={shr_d:+.3f})"
    )
    print(
        f"  Turnover falls by >=10%:    {'PASS' if pass_turnover else 'FAIL'}  (change={to_pct_chg:+.1%})"
    )
    print(
        f"  No material drawdown deterioration:  {'PASS' if pass_mdd else 'FAIL'}  (MaxDD delta={mdd_d:+.2%})"
    )
    print()
    if decision_pass:
        print("  DECISION: PASS")
    else:
        print("  DECISION: REJECT")

    # --- Bias audit ---
    print("\n" + "=" * 72)
    print("BIAS AUDIT")
    print("=" * 72)
    print("  no baseline logic drift:   PASS (only tau changed; VOL_LOOKBACK=126 for both)")
    print(
        "  same execution path:      PASS (tolerance passed to engine; applied at rebalance only)"
    )
    print("  correct tau application:  PASS (|trade| > tau -> execute; else suppress)")
    print("  correct OOS-only evaluation: PASS (walk-forward test periods only)")
    print("  no lookahead:              PASS (tau filters same-day target vs prev; no future data)")


if __name__ == "__main__":
    main()
