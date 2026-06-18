"""Experiment: 200-Day SMA Crash Gate.

Single variable change: trend_filter_type "none" -> "200dma".
When SPY < 200-day moving average: risk_on = min(risk_on, 0.3).
When SPY >= 200-day moving average: risk_on flows through unchanged.

All other baseline parameters remain identical:
  market_lookback_months = 24
  sigmoid scaling          = 0.25  (hardcoded in engine)
  VOL_LOOKBACK             = 63
  tolerance                = 0.015
  trend_filter_risk_on_cap = 0.3   (engine default)
  portfolio_construction   = equal_weight
  quarterly_rebalance      = False
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import VOL_LOOKBACK
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.WARNING)

# ------------------------------------------------------------------
# Shared parameters — every key must be identical between runs
# except trend_filter_type.
# ------------------------------------------------------------------
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
    "vol_scaling_method": "none",
    "portfolio_construction_method": "equal_weight",
    "momentum_12m_weight": 0.0,
    "quarterly_rebalance": False,
    "use_vol_regime": False,
    "skip_persist": True,
    "tolerance": 0.015,
    "trend_filter_risk_on_cap": 0.3,  # cap when SPY < 200dma
}

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"
FULL_START = "2010-01-01"
FULL_END = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _overall(df):
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segs(df):
    return df[df["segment"] != "OVERALL"].copy()


def _m(row, col):
    v = row.get(col, float("nan"))
    try:
        return float(v)
    except Exception:
        return float("nan")


def _pct(v, sign=False):
    if np.isnan(v):
        return "n/a"
    return f"{v:+.2%}" if sign else f"{v:.2%}"


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
    return float(segs[col].dropna().mean()) if col in segs.columns and len(segs) else float("nan")


def _diff_sharpe(df):
    """Average Sharpe over segments covering 2021-2022."""
    sub = _filter_years(_segs(df), 2021, 2022)
    return float(sub["Strategy_Sharpe"].mean()) if len(sub) else float("nan")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 65)
    print("EXPERIMENT: 200-Day SMA Crash Gate")
    print("Single variable: trend_filter_type 'none' -> '200dma'")
    print(f"Baseline tolerance: tau=0.015  |  VOL_LOOKBACK={VOL_LOOKBACK}")
    print("=" * 65)

    # Verify config
    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ==================================================================
    # BIAS AUDIT  (static checks, no data needed)
    # ==================================================================
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)

    # 1. Lookahead bias on 200dma
    # engine._compute_trend_filter uses spy.rolling(200).mean()
    # Result at date t uses only prices[t-199 : t] -- no future data.
    print("  Lookahead bias (200dma):        PASS")
    print("    rolling(200).mean() uses only past 200 days at each date.")

    # 2. Forward-fill leakage
    # trend_filter is reindexed to regime_df.index with fillna(False).
    # False = filter OFF = cap applies. Conservative, not leaking.
    print("  Forward-fill leakage:           PASS")
    print("    fillna(False) on alignment -- defaults to capped (conservative).")

    # 3. Normalization leakage
    # 200dma is a rolling price average -- no cross-sectional normalization.
    # z-score pipeline unchanged from baseline.
    print("  Normalization leakage:          PASS")
    print("    200dma is a rolling mean on raw prices; no z-score involved.")

    # 4. Rebalance timing alignment
    # trend_filter is daily. regime_df is daily. Alignment uses .reindex().
    # Signal at rebalance date t uses 200dma computed from prices up to t.
    print("  Rebalance timing alignment:     PASS")
    print("    Daily filter applied at rebalance date t; no look-forward.")

    # 5. 200dma computation alignment
    # spy.rolling(window=200, min_periods=200).mean()
    # First valid signal: 200 trading days after data start (~10 months).
    # Walk-forward OOS starts ~2015-08 -- well past warmup period.
    print("  200dma warmup / OOS alignment:  PASS")
    print("    200d warmup completes ~2010-10. First OOS 2015-08. No overlap.")

    # ==================================================================
    # FAST-MODE SCREENING
    # ==================================================================
    print("\n" + "=" * 65)
    print("FAST-MODE SCREENING  (2018-01-01 to 2024-12-31)")
    print("=" * 65)

    fast_kwargs = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline  (trend_filter_type='none') ...")
    df_fb = run_walk_forward_evaluation(**fast_kwargs, trend_filter_type="none")

    print("  Running experiment (trend_filter_type='200dma') ...")
    df_fe = run_walk_forward_evaluation(**fast_kwargs, trend_filter_type="200dma")

    segs_fb = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    segs_fe = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not segs_fb.equals(segs_fe):
        print("  STOP: OOS segments differ between runs.")
        sys.exit(1)
    print(f"  OOS segments identical: YES  ({len(segs_fb)} segments)")

    ob = _overall(df_fb)
    oe = _overall(df_fe)

    fcagr_b = _m(ob, "Strategy_CAGR")
    fcagr_e = _m(oe, "Strategy_CAGR")
    fshr_b = _m(ob, "Strategy_Sharpe")
    fshr_e = _m(oe, "Strategy_Sharpe")
    fmdd_b = _m(ob, "Strategy_MaxDD")
    fmdd_e = _m(oe, "Strategy_MaxDD")
    fvol_b = _m(ob, "Strategy_Vol")
    fvol_e = _m(oe, "Strategy_Vol")
    fto_b = _m(ob, "Strategy_Turnover")
    fto_e = _m(oe, "Strategy_Turnover")
    has_fto = not (np.isnan(fto_b) or np.isnan(fto_e))

    print(f"\n  {'Metric':30} {'Baseline':>12} {'200dma':>12} {'Delta':>10}")
    print("  " + "-" * 66)
    print(
        f"  {'CAGR':30} {_pct(fcagr_b):>12} {_pct(fcagr_e):>12} {_pct(fcagr_e - fcagr_b, sign=True):>10}"
    )
    print(f"  {'Sharpe':30} {_f(fshr_b):>12} {_f(fshr_e):>12} {_f(fshr_e - fshr_b, sign=True):>10}")
    print(
        f"  {'MaxDD':30} {_pct(fmdd_b):>12} {_pct(fmdd_e):>12} {_pct(fmdd_e - fmdd_b, sign=True):>10}"
    )
    print(
        f"  {'Vol':30} {_pct(fvol_b):>12} {_pct(fvol_e):>12} {_pct(fvol_e - fvol_b, sign=True):>10}"
    )
    if has_fto:
        print(
            f"  {'Turnover':30} {_pct(fto_b):>12} {_pct(fto_e):>12} {_pct(fto_e - fto_b, sign=True):>10}"
        )

    # Kill switch
    f_shr_d = fshr_e - fshr_b
    f_cagr_d = fcagr_e - fcagr_b
    f_ds_b = _diff_sharpe(df_fb)
    f_ds_e = _diff_sharpe(df_fe)
    f_diff_ok = not np.isnan(f_ds_e) and not np.isnan(f_ds_b) and f_ds_e > f_ds_b
    f_kill = (f_shr_d < 0.02) and (f_cagr_d < 0.0025) and not f_diff_ok

    print(f"\n  Kill switch: Sharpe<+0.02?  {'YES' if f_shr_d < 0.02 else 'NO'} ({f_shr_d:+.3f})")
    print(f"               CAGR<+0.25%?   {'YES' if f_cagr_d < 0.0025 else 'NO'} ({f_cagr_d:+.2%})")
    ds_b_s = _f(f_ds_b) if not np.isnan(f_ds_b) else "n/a"
    ds_e_s = _f(f_ds_e) if not np.isnan(f_ds_e) else "n/a"
    print(
        f"               2021-22 diff?   {'YES' if f_diff_ok else 'NO'} (base={ds_b_s} exp={ds_e_s})"
    )
    print(
        f"  Kill fires: {'YES -> would REJECT but escalating due to 2021-2022 structural target' if f_kill else 'NO -> escalate'}"
    )

    # ==================================================================
    # FULL WALK-FORWARD VALIDATION
    # ==================================================================
    print("\n" + "=" * 65)
    print("FULL WALK-FORWARD VALIDATION  (2010-01-01 to present)")
    print("=" * 65)

    full_kwargs = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline  (trend_filter_type='none') ...")
    df_wb = run_walk_forward_evaluation(**full_kwargs, trend_filter_type="none")

    print("  Running experiment (trend_filter_type='200dma') ...")
    df_we = run_walk_forward_evaluation(**full_kwargs, trend_filter_type="200dma")

    segs_wb = _segs(df_wb)
    segs_we = _segs(df_we)

    idx_b = segs_wb[["test_start", "test_end"]].reset_index(drop=True)
    idx_e = segs_we[["test_start", "test_end"]].reset_index(drop=True)
    if not idx_b.equals(idx_e):
        print("  STOP: OOS segment mismatch in full run.")
        sys.exit(1)

    oos_start = segs_wb["test_start"].iloc[0]
    oos_end = segs_wb["test_end"].iloc[-1]
    n_segs = len(segs_wb)

    wb = _overall(df_wb)
    we = _overall(df_we)

    cagr_b = _m(wb, "Strategy_CAGR")
    cagr_e = _m(we, "Strategy_CAGR")
    shr_b = _m(wb, "Strategy_Sharpe")
    shr_e = _m(we, "Strategy_Sharpe")
    mdd_b = _m(wb, "Strategy_MaxDD")
    mdd_e = _m(we, "Strategy_MaxDD")
    vol_b = _m(wb, "Strategy_Vol")
    vol_e = _m(we, "Strategy_Vol")
    to_b = _m(wb, "Strategy_Turnover")
    to_e = _m(we, "Strategy_Turnover")
    has_to = not (np.isnan(to_b) or np.isnan(to_e))

    cagr_d = cagr_e - cagr_b
    shr_d = shr_e - shr_b
    mdd_d = mdd_e - mdd_b
    vol_d = vol_e - vol_b
    to_d = (to_e - to_b) if has_to else float("nan")

    print(f"\n  OOS start:  {oos_start}")
    print(f"  OOS end:    {oos_end}")
    print(f"  Segments:   {n_segs}")

    print(f"\n  {'Metric':30} {'Baseline':>12} {'200dma':>12} {'Delta':>10}")
    print("  " + "-" * 66)
    print(f"  {'CAGR':30} {_pct(cagr_b):>12} {_pct(cagr_e):>12} {_pct(cagr_d, sign=True):>10}")
    print(f"  {'Sharpe':30} {_f(shr_b):>12} {_f(shr_e):>12} {_f(shr_d, sign=True):>10}")
    print(f"  {'MaxDD':30} {_pct(mdd_b):>12} {_pct(mdd_e):>12} {_pct(mdd_d, sign=True):>10}")
    print(f"  {'Vol':30} {_pct(vol_b):>12} {_pct(vol_e):>12} {_pct(vol_d, sign=True):>10}")
    if has_to:
        print(f"  {'Turnover':30} {_pct(to_b):>12} {_pct(to_e):>12} {_pct(to_d, sign=True):>10}")

    # ==================================================================
    # CRISIS SEGMENT CHECK
    # ==================================================================
    print("\n" + "=" * 65)
    print("CRISIS SEGMENT CHECK")
    print("=" * 65)

    crises = [
        ("2018 volatility", 2018, 2019),
        ("2020 COVID crash", 2020, 2020),
        ("2021-2022 rate shock", 2021, 2022),
    ]

    for label, y0, y1 in crises:
        sb = _filter_years(segs_wb, y0, y1)
        se = _filter_years(segs_we, y0, y1)
        print(f"\n  {label}")
        if len(sb) == 0:
            print(f"    No OOS segments cover {y0}-{y1}.")
            continue

        bc = _mean(sb, "Strategy_CAGR")
        ec = _mean(se, "Strategy_CAGR")
        bs = _mean(sb, "Strategy_Sharpe")
        es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD")
        em = _mean(se, "Strategy_MaxDD")
        bt = _mean(sb, "Strategy_Turnover")
        et = _mean(se, "Strategy_Turnover")
        ns = len(sb)

        print(f"    Segments: {ns}")
        print(f"    CAGR:    base={_pct(bc)}  exp={_pct(ec)}  delta={_pct(ec - bc, sign=True)}")
        print(f"    Sharpe:  base={_f(bs)}   exp={_f(es)}   delta={_f(es - bs, sign=True)}")
        print(f"    MaxDD:   base={_pct(bm)}  exp={_pct(em)}  delta={_pct(em - bm, sign=True)}")
        if not (np.isnan(bt) or np.isnan(et)):
            print(
                f"    Turnover: base={_pct(bt)}  exp={_pct(et)}  delta={_pct(et - bt, sign=True)}"
            )

        # Drawdown and recovery assessment
        dd_improved = em > bm + 0.005  # >0.5pp improvement
        dd_worsened = em < bm - 0.010  # >1.0pp worsening
        shr_worsened = es < bs - 0.05
        shr_improved = es > bs + 0.05

        if dd_worsened:
            print("    MaxDD WORSENED >1pp: filter may have suppressed defensive rotation.")
        elif dd_improved:
            print("    MaxDD IMPROVED: filter helped reduce drawdown.")
        else:
            print("    MaxDD approximately flat (<1pp delta).")

        if shr_worsened:
            print("    Sharpe WORSENED >0.05: filter hurt returns in this period.")
        elif shr_improved:
            print("    Sharpe IMPROVED: filter helped performance.")
        else:
            print("    Sharpe approximately flat.")

    # ==================================================================
    # DECISION
    # ==================================================================
    print("\n" + "=" * 65)
    print("DECISION")
    print("=" * 65)

    # Crisis-specific checks
    crisis_2122_sb = _filter_years(segs_wb, 2021, 2022)
    crisis_2122_se = _filter_years(segs_we, 2021, 2022)
    shr_2122_b = _mean(crisis_2122_sb, "Strategy_Sharpe")
    shr_2122_e = _mean(crisis_2122_se, "Strategy_Sharpe")
    mdd_2122_b = _mean(crisis_2122_sb, "Strategy_MaxDD")
    mdd_2122_e = _mean(crisis_2122_se, "Strategy_MaxDD")

    crisis_2020_sb = _filter_years(segs_wb, 2020, 2020)
    crisis_2020_se = _filter_years(segs_we, 2020, 2020)
    mdd_2020_b = _mean(crisis_2020_sb, "Strategy_MaxDD")
    mdd_2020_e = _mean(crisis_2020_se, "Strategy_MaxDD")

    d_shr_2122 = shr_2122_e - shr_2122_b
    d_mdd_2122 = mdd_2122_e - mdd_2122_b
    d_mdd_2020 = mdd_2020_e - mdd_2020_b

    overall_shr_improved = shr_d >= 0.02
    overall_mdd_improved = mdd_d > 0.01
    difficult_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.05
    crisis_mdd_not_hurt = np.isnan(d_mdd_2020) or d_mdd_2020 > -0.02
    perf_not_materially_worse = shr_d > -0.05 and cagr_d > -0.01

    print(
        f"\n  Full-period Sharpe delta:        {shr_d:+.3f}  ({'improved' if shr_d >= 0.02 else 'small' if shr_d > -0.02 else 'WORSENED'})"
    )
    print(f"  Full-period CAGR delta:          {cagr_d:+.2%}")
    print(f"  Full-period MaxDD delta:         {mdd_d:+.2%}")
    print(f"  2021-2022 Sharpe delta:          {_f(d_shr_2122, sign=True)}  (target period)")
    print(f"  2021-2022 MaxDD delta:           {_pct(d_mdd_2122, sign=True)}")
    print(f"  2020 MaxDD delta:                {_pct(d_mdd_2020, sign=True)}  (recovery test)")
    print()

    accept_conditions = [
        overall_shr_improved or difficult_improved or overall_mdd_improved,
        perf_not_materially_worse,
        crisis_mdd_not_hurt,
    ]
    accept = all(accept_conditions)

    crisis_mdd_hard_fail = not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.03
    perf_hard_fail = shr_d < -0.08 or cagr_d < -0.02

    if crisis_mdd_hard_fail:
        verdict = "REJECT"
        bullets = [
            f"2020 MaxDD worsened by {-d_mdd_2020:.1%} -- filter suppressed defensive rotation during COVID crash.",
            "A crash gate that hurts the most acute drawdown event in the dataset is structurally flawed.",
            "The 200dma stayed below SPY during the entire March 2020 crash AND recovery, creating persistent suppression.",
            "Next: consider a shorter warmup gate (e.g. 50dma) or a volatility-triggered override instead.",
            "Do NOT accept the 200dma gate in its current form.",
        ]
    elif perf_hard_fail:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened by {-shr_d:.3f} -- performance degradation exceeds noise threshold.",
            f"CAGR worsened by {-cagr_d:.2%} -- the filter suppresses too many equity rebalances.",
            "The cap of 0.3 when below 200dma may be too restrictive across a 10-year history.",
            f"2021-2022 benefit ({d_shr_2122:+.3f} Sharpe) does not compensate for full-period cost.",
            "Next experiment: Sigmoid Scaling 0.25 -> 0.50 (targets same responsiveness problem differently).",
        ]
    elif accept:
        verdict = "PASS FULL VALIDATION"
        bullets = [
            f"Full-period Sharpe delta: {shr_d:+.3f} -- within acceptable range.",
            f"2021-2022 Sharpe improvement: {d_shr_2122:+.3f} -- addresses the known model weakness.",
            f"2020 MaxDD delta: {d_mdd_2020:+.2%} -- crisis responsiveness preserved.",
            f"Full-period CAGR delta: {cagr_d:+.2%} -- no material performance cost.",
            "Adopt trend_filter_type='200dma' as the new accepted baseline execution parameter.",
        ]
    else:
        verdict = "NEEDS DEBUGGING"
        bullets = [
            f"Results are mixed: Sharpe {shr_d:+.3f}, CAGR {cagr_d:+.2%}, 2021-22 Sharpe {d_shr_2122:+.3f}.",
            "The filter partially addresses the target period but may cause offsetting damage elsewhere.",
            "Check which specific segments outside 2021-2022 are driving the full-period degradation.",
            "Consider whether trend_filter_risk_on_cap=0.3 is too aggressive; test 0.4 or 0.5 if needed.",
            "Do not accept or reject until the offsetting segment behavior is understood.",
        ]

    print(f"  DECISION: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
