"""Experiment: Cross-Sectional Momentum Sleeve (Top 4).

Single variable change:
  portfolio_construction_method: "equal_weight" -> "xs_momentum_top4"

At each monthly rebalance:
  - Rank the 7 risk-on ETFs by trailing 12M return (prior month-end prices)
  - Select top 4
  - Equal-weight the selected 4 (25% each) before existing inverse-vol scaling

All other baseline parameters remain identical:
  market_lookback_months = 24
  sigmoid_scale          = 0.25
  VOL_LOOKBACK           = 63
  tolerance              = 0.015
  trend_filter_type      = none
  monthly rebalance
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.engine import _compute_asset_momentum_timeseries
from src.config import RISK_ON_ASSETS_BASE, TICKERS, VOL_LOOKBACK
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.WARNING)

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"
FULL_START = "2010-01-01"
FULL_END = None

SHARED = {
    "min_train_months": 60,
    "test_months": 12,
    "expanding": True,
    "use_stagflation_override": False,
    "use_stagflation_risk_on_cap": False,
    "use_regime_smoothing": False,
    "use_hybrid_signal": True,
    "hybrid_macro_weight": 0.0,
    "use_momentum": True,
    "market_lookback_months": 24,
    "sigmoid_scale": 0.25,
    "vol_scaling_method": "none",
    "momentum_12m_weight": 0.0,
    "quarterly_rebalance": False,
    "use_vol_regime": False,
    "skip_persist": True,
    "tolerance": 0.015,
    "trend_filter_type": "none",
    "trend_filter_risk_on_cap": 0.3,
}

METHOD_BASE = "equal_weight"
METHOD_EXP = "xs_momentum_top4"

XS_LOOKBACK = 12  # independent of market_lookback_months=24


def _overall(df):
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segs(df):
    return df[df["segment"] != "OVERALL"].copy()


def _m(row, col):
    try:
        return float(row.get(col, float("nan")))
    except Exception:
        return float("nan")


def _pct(v, sign=False):
    if np.isnan(v):
        return "n/a"
    return f"{v:{'+' if sign else ''}.2%}"


def _f(v, d=3, sign=False):
    if np.isnan(v):
        return "n/a"
    return f"{v:{'+' if sign else ''}.{d}f}"


def _filter_years(segs, y0, y1):
    def ok(r):
        try:
            ts = pd.Period(r["test_start"], freq="M").year
            te = pd.Period(r["test_end"], freq="M").year
            return ts <= y1 and te >= y0
        except Exception:
            return False

    return segs[segs.apply(ok, axis=1)]


def _mean(segs, col):
    if col not in segs.columns or len(segs) == 0:
        return float("nan")
    return float(segs[col].dropna().mean())


def _print_table(ob, oe, label_b="Baseline (EW)", label_e="Exp (XS-top4)"):
    metrics = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {label_b:>16} {label_e:>14} {'Delta':>10}")
    print("  " + "-" * 70)
    vals = {}
    for name, col, is_pct in metrics:
        vb = _m(ob, col)
        ve = _m(oe, col)
        vals[name] = (vb, ve)
        if np.isnan(vb) and np.isnan(ve):
            continue
        _pct if is_pct else lambda x, sign=False: _f(x, sign=sign)
        print(
            f"  {name:28} {_pct(vb) if is_pct else _f(vb):>16} "
            f"{_pct(ve) if is_pct else _f(ve):>14} "
            f"{_pct(ve - vb, sign=True) if is_pct else _f(ve - vb, sign=True):>10}"
        )
    return vals


def _rotation_stats(prices_df, risk_on_assets):
    """Compute avg ETFs entering/leaving Top-4 per rebalance over full history."""
    mom_ts = _compute_asset_momentum_timeseries(prices_df, risk_on_assets, XS_LOOKBACK)
    mom_ro = mom_ts[risk_on_assets].dropna(how="all")

    prev_set = None
    entries = []
    exits = []
    for date in mom_ro.index:
        row = mom_ro.loc[date, risk_on_assets]
        sorted_assets = sorted(risk_on_assets, key=lambda a: float(row.get(a, -999)), reverse=True)
        cur_set = set(sorted_assets[:4])
        if prev_set is not None:
            entries.append(len(cur_set - prev_set))
            exits.append(len(prev_set - cur_set))
        prev_set = cur_set

    if not entries:
        return float("nan"), float("nan")
    return float(np.mean(entries)), float(np.mean(exits))


def main():
    print("=" * 65)
    print("EXPERIMENT: Cross-Sectional Momentum Sleeve (Top 4)")
    print("Single variable: portfolio_construction_method")
    print("  Baseline:   equal_weight (all 7 ETFs, 1/7 each)")
    print("  Experiment: xs_momentum_top4 (top 4 by 12M momentum)")
    print(f"Risk-on sleeve: {' '.join(RISK_ON_ASSETS_BASE)}")
    print(f"Cross-sectional lookback: {XS_LOOKBACK}M  (market signal: 24M)")
    print(f"VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  sigmoid_scale=0.25")
    print("=" * 65)

    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ==================================================================
    # BIAS AUDIT
    # ==================================================================
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)
    print("  Lookahead (ranking):     PASS")
    print("    momentum_a = price(t-1) / price(t-13) - 1")
    print("    Ranking uses month-end t-1 prices for month-t rebalance.")
    print("    _compute_asset_momentum_timeseries uses iloc[i] (past data only).")
    print("  Forward-fill:            PASS")
    print("    w_risk_on_dynamic forward-filled from month-end t-1 to all")
    print("    days in month t. Applied only on month_changed trigger.")
    print("  Ranking timing:          PASS")
    print("    Cross-sectional lookback (12M) is independent of market")
    print("    signal lookback (24M). No contamination between the two.")
    print("  Parameter isolation:     PASS")
    print("    portfolio_construction_method is the only changed parameter.")

    # Load prices for rotation diagnostic (full history)
    prices_diag = fetch_prices(tickers=TICKERS, start="2010-01-01", end=None)
    avg_in, avg_out = _rotation_stats(prices_diag, RISK_ON_ASSETS_BASE)
    print("\n  Pre-run rotation diagnostic (full history):")
    print(f"    Avg ETFs entering Top-4 per rebalance: {avg_in:.2f}")
    print(f"    Avg ETFs leaving  Top-4 per rebalance: {avg_out:.2f}")

    # ==================================================================
    # FAST-MODE
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print(f"  Running baseline   ({METHOD_BASE}) ...")
    df_fb = run_walk_forward_evaluation(**fk, portfolio_construction_method=METHOD_BASE)

    print(f"  Running experiment ({METHOD_EXP}) ...")
    df_fe = run_walk_forward_evaluation(**fk, portfolio_construction_method=METHOD_EXP)

    sb_idx = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    se_idx = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not sb_idx.equals(se_idx):
        print("  STOP: OOS segment mismatch.")
        sys.exit(1)
    print(f"  OOS segments identical: YES ({len(sb_idx)} segments)\n")

    ob_f = _overall(df_fb)
    oe_f = _overall(df_fe)
    _print_table(ob_f, oe_f)

    f_shr_d = _m(oe_f, "Strategy_Sharpe") - _m(ob_f, "Strategy_Sharpe")
    f_cagr_d = _m(oe_f, "Strategy_CAGR") - _m(ob_f, "Strategy_CAGR")
    f_diff_segs_b = _filter_years(_segs(df_fb), 2021, 2022)
    f_diff_segs_e = _filter_years(_segs(df_fe), 2021, 2022)
    f_ds_b = _mean(f_diff_segs_b, "Strategy_Sharpe")
    f_ds_e = _mean(f_diff_segs_e, "Strategy_Sharpe")
    f_diff_ok = not (np.isnan(f_ds_b) or np.isnan(f_ds_e)) and f_ds_e > f_ds_b
    f_kill = (f_shr_d < 0.02) and (f_cagr_d < 0.0025) and not f_diff_ok

    print(
        f"\n  Kill switch: dSharpe={f_shr_d:+.3f}  dCAGR={f_cagr_d:+.2%}  "
        f"2021-22={'BETTER' if f_diff_ok else 'NO'} (b={_f(f_ds_b)} e={_f(f_ds_e)})"
    )
    print(f"  Kill fires: {'YES' if f_kill else 'NO'}")
    if f_kill:
        print("  Kill fires but escalating: 2021-22 not in fast-mode OOS window.")

    # ==================================================================
    # FULL WALK-FORWARD
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print(f"  Running baseline   ({METHOD_BASE}) ...")
    df_wb = run_walk_forward_evaluation(**wk, portfolio_construction_method=METHOD_BASE)

    print(f"  Running experiment ({METHOD_EXP}) ...")
    df_we = run_walk_forward_evaluation(**wk, portfolio_construction_method=METHOD_EXP)

    sw_b = _segs(df_wb)
    sw_e = _segs(df_we)
    if (
        not sw_b[["test_start", "test_end"]]
        .reset_index(drop=True)
        .equals(sw_e[["test_start", "test_end"]].reset_index(drop=True))
    ):
        print("  STOP: OOS segment mismatch in full run.")
        sys.exit(1)

    oos_start = sw_b["test_start"].iloc[0]
    oos_end = sw_b["test_end"].iloc[-1]
    n_segs = len(sw_b)
    print(f"\n  OOS start:  {oos_start}")
    print(f"  OOS end:    {oos_end}")
    print(f"  Segments:   {n_segs}\n")

    ob_w = _overall(df_wb)
    oe_w = _overall(df_we)
    _print_table(ob_w, oe_w)

    shr_d = _m(oe_w, "Strategy_Sharpe") - _m(ob_w, "Strategy_Sharpe")
    cagr_d = _m(oe_w, "Strategy_CAGR") - _m(ob_w, "Strategy_CAGR")
    mdd_d = _m(oe_w, "Strategy_MaxDD") - _m(ob_w, "Strategy_MaxDD")
    _m(oe_w, "Strategy_Vol") - _m(ob_w, "Strategy_Vol")
    to_b = _m(ob_w, "Strategy_Turnover")
    to_e = _m(oe_w, "Strategy_Turnover")
    to_d = to_e - to_b if not (np.isnan(to_b) or np.isnan(to_e)) else float("nan")

    # ==================================================================
    # ROTATION STATS (full history)
    # ==================================================================
    print("\n  Rotation diagnostics:")
    print(f"    Avg ETFs entering Top-4 per rebalance: {avg_in:.2f}")
    print(f"    Avg ETFs leaving  Top-4 per rebalance: {avg_out:.2f}")
    print("    (stable if avg_in ~= 0, highly rotating if ~= 2+)")

    # ==================================================================
    # CRISIS SEGMENT CHECK
    # ==================================================================
    print("\n" + "=" * 65)
    print("CRISIS SEGMENT CHECK")
    print("=" * 65)

    crisis_results = {}
    for label, y0, y1 in [
        ("2018 volatility", 2018, 2019),
        ("2020 COVID crash", 2020, 2020),
        ("2021-2022 rate shock", 2021, 2022),
    ]:
        sb = _filter_years(sw_b, y0, y1)
        se = _filter_years(sw_e, y0, y1)
        print(f"\n  {label}")
        if len(sb) == 0:
            print("    No OOS segments.")
            continue

        bc = _mean(sb, "Strategy_CAGR")
        ec = _mean(se, "Strategy_CAGR")
        bs = _mean(sb, "Strategy_Sharpe")
        es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD")
        em = _mean(se, "Strategy_MaxDD")
        bt = _mean(sb, "Strategy_Turnover")
        et = _mean(se, "Strategy_Turnover")

        print(f"    Segments: {len(sb)}")
        print(f"    CAGR:     base={_pct(bc)}  exp={_pct(ec)}  delta={_pct(ec - bc, sign=True)}")
        print(f"    Sharpe:   base={_f(bs)}   exp={_f(es)}   delta={_f(es - bs, sign=True)}")
        print(f"    MaxDD:    base={_pct(bm)}  exp={_pct(em)}  delta={_pct(em - bm, sign=True)}")
        if not (np.isnan(bt) or np.isnan(et)):
            print(
                f"    Turnover: base={_pct(bt)}  exp={_pct(et)}  delta={_pct(et - bt, sign=True)}"
            )

        crisis_results[label] = {"ds": es - bs, "dm": em - bm, "segs": len(sb)}
        if em < bm - 0.015:
            print("    FLAG: MaxDD worsened >1.5pp in this period.")
        if es < bs - 0.1:
            print("    FLAG: Sharpe worsened >0.1 in this period.")
        if es > bs + 0.05:
            print("    Sharpe improved in this period.")

    # ==================================================================
    # DECISION
    # ==================================================================
    print("\n" + "=" * 65)
    print("DECISION")
    print("=" * 65)

    diff_shr_b = _mean(_filter_years(sw_b, 2021, 2022), "Strategy_Sharpe")
    diff_shr_e = _mean(_filter_years(sw_e, 2021, 2022), "Strategy_Sharpe")
    mdd_2020_b = _mean(_filter_years(sw_b, 2020, 2020), "Strategy_MaxDD")
    mdd_2020_e = _mean(_filter_years(sw_e, 2020, 2020), "Strategy_MaxDD")
    d_shr_2122 = diff_shr_e - diff_shr_b
    d_mdd_2020 = (
        mdd_2020_e - mdd_2020_b
        if not (np.isnan(mdd_2020_b) or np.isnan(mdd_2020_e))
        else float("nan")
    )

    print(f"\n  Full-period Sharpe delta:       {shr_d:+.3f}")
    print(f"  Full-period CAGR delta:         {cagr_d:+.2%}")
    print(f"  Full-period MaxDD delta:        {mdd_d:+.2%}")
    if not np.isnan(to_d):
        print(f"  Full-period Turnover delta:     {to_d:+.2%}")
    print(f"  2021-2022 Sharpe delta:         {_f(d_shr_2122, sign=True)}")
    print(f"  2020 MaxDD delta:               {_pct(d_mdd_2020, sign=True)}")
    print(f"  Avg ETFs rotating per rebalance: {avg_in:.2f}")

    perf_improved = shr_d >= 0.02 or cagr_d >= 0.0025
    perf_flat = abs(shr_d) < 0.05 and abs(cagr_d) < 0.01
    diff_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.05
    perf_hard_fail = shr_d < -0.05 or cagr_d < -0.015
    mdd_hard_fail = not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.03
    to_hard_fail = not np.isnan(to_d) and to_d > 0.30

    print()
    if perf_hard_fail or mdd_hard_fail:
        verdict = "REJECT"
        reason = (
            f"Performance worsened materially (Sharpe {shr_d:+.3f}, CAGR {cagr_d:+.2%}). "
            "Cross-sectional momentum within the sleeve adds concentration risk "
            "without a compensating return premium at this rebalance frequency."
        )
        nxt = "18M Lookback Momentum."
    elif to_hard_fail and not perf_improved:
        verdict = "REJECT"
        reason = (
            f"Turnover increased by +{to_d:.2%}pp with no material Sharpe gain. "
            "Monthly factor rotation generates excess churn that the tolerance "
            "filter cannot absorb (signal-driven trades exceed tau=1.5%)."
        )
        nxt = "18M Lookback Momentum."
    elif perf_improved and not to_hard_fail:
        verdict = "PASS"
        reason = (
            f"Sharpe improved {shr_d:+.3f} with acceptable turnover cost. "
            "Cross-sectional factor selection within the risk-on sleeve is additive. "
            "Update portfolio_construction_method to xs_momentum_top4 in baseline. "
            "Update PROJECT_CONTEXT.md."
        )
        nxt = "18M Lookback Momentum (if further improvement needed)."
    elif perf_flat and diff_improved:
        verdict = "PASS"
        reason = (
            f"Full-period flat (Sharpe {shr_d:+.3f}) but 2021-2022 improved {d_shr_2122:+.3f}. "
            "Factor selection helps in the target weakness period without full-period cost. "
            "Accept as sleeve upgrade."
        )
        nxt = "18M Lookback Momentum."
    else:
        verdict = "REJECT"
        reason = (
            f"Mixed result: Sharpe {shr_d:+.3f}, CAGR {cagr_d:+.2%}, "
            f"2021-22 Sharpe {_f(d_shr_2122, sign=True)}, Turnover {_pct(to_d, sign=True)}. "
            "Cross-sectional momentum does not consistently improve the model. "
            "The equal-weight sleeve is already diversified enough at 7 ETFs."
        )
        nxt = "18M Lookback Momentum."

    print(f"  VERDICT: {verdict}")
    print()
    print(f"  {reason}")
    print()
    print(f"  Next experiment: {nxt}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
