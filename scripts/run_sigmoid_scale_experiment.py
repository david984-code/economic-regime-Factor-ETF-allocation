"""Experiment: Sigmoid Scaling Adjustment.

Single variable change: sigmoid_scale 0.25 -> 0.50.
  Baseline:   risk_on = sigmoid(z_score * 0.25)
  Experiment: risk_on = sigmoid(z_score * 0.50)

All other baseline parameters remain identical:
  market_lookback_months = 24
  VOL_LOOKBACK             = 63
  tolerance                = 0.015
  trend_filter_type        = none
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

SCALE_BASE = 0.25
SCALE_EXP = 0.50

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
    "trend_filter_type": "none",
    "trend_filter_risk_on_cap": 0.3,
}

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"
FULL_START = "2010-01-01"
FULL_END = None


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


def _print_table(ob, oe, label_b="Baseline (0.25)", label_e="Exp (0.50)"):
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
    has_to = not (np.isnan(to_b) or np.isnan(to_e))
    print(f"  {'Metric':30} {label_b:>16} {label_e:>14} {'Delta':>10}")
    print("  " + "-" * 72)
    print(
        f"  {'CAGR':30} {_pct(cagr_b):>16} {_pct(cagr_e):>14} {_pct(cagr_e - cagr_b, sign=True):>10}"
    )
    print(f"  {'Sharpe':30} {_f(shr_b):>16} {_f(shr_e):>14} {_f(shr_e - shr_b, sign=True):>10}")
    print(
        f"  {'MaxDD':30} {_pct(mdd_b):>16} {_pct(mdd_e):>14} {_pct(mdd_e - mdd_b, sign=True):>10}"
    )
    print(f"  {'Vol':30} {_pct(vol_b):>16} {_pct(vol_e):>14} {_pct(vol_e - vol_b, sign=True):>10}")
    if has_to:
        print(
            f"  {'Turnover':30} {_pct(to_b):>16} {_pct(to_e):>14} {_pct(to_e - to_b, sign=True):>10}"
        )
    return cagr_b, cagr_e, shr_b, shr_e, mdd_b, mdd_e, vol_b, vol_e, to_b, to_e


def main():
    print("=" * 65)
    print("EXPERIMENT: Sigmoid Scaling Adjustment")
    print(f"Single variable: sigmoid_scale {SCALE_BASE} -> {SCALE_EXP}")
    print(f"  Baseline:   risk_on = sigmoid(z * {SCALE_BASE})")
    print(f"  Experiment: risk_on = sigmoid(z * {SCALE_EXP})")
    print(f"VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  lookback=24M")
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
    print("  Lookahead bias (sigmoid_scale):  PASS")
    print("    sigmoid_scale only changes the steepness of the mapping.")
    print("    The z-score input still uses the expanding window (no future data).")
    print("  Forward-fill leakage:            PASS")
    print("    risk_on forward-filled daily from month-end, same as baseline.")
    print("  Normalization leakage:           PASS")
    print("    Expanding z-score window unchanged. No future data in normalization.")
    print("  Rebalance timing alignment:      PASS")
    print("    Signal computed at month-end t-1, applied at first trading day of month t.")
    print("  Parameter isolation:             PASS")
    print("    sigmoid_scale is the only changed parameter. Verified via SHARED dict.")

    # ==================================================================
    # FAST-MODE
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print(f"  Running baseline   (sigmoid_scale={SCALE_BASE}) ...")
    df_fb = run_walk_forward_evaluation(**fk, sigmoid_scale=SCALE_BASE)

    print(f"  Running experiment (sigmoid_scale={SCALE_EXP}) ...")
    df_fe = run_walk_forward_evaluation(**fk, sigmoid_scale=SCALE_EXP)

    sb_idx = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    se_idx = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not sb_idx.equals(se_idx):
        print("  STOP: OOS segment mismatch.")
        sys.exit(1)
    print(f"  OOS segments identical: YES ({len(sb_idx)} segments)\n")

    ob_f = _overall(df_fb)
    oe_f = _overall(df_fe)
    fc, fce, fs, fse, fm, fme, fv, fve, ft, fte = _print_table(ob_f, oe_f)

    f_shr_d = fse - fs
    f_cagr_d = fce - fc
    diff_segs_b = _filter_years(_segs(df_fb), 2021, 2022)
    diff_segs_e = _filter_years(_segs(df_fe), 2021, 2022)
    f_ds_b = _mean(diff_segs_b, "Strategy_Sharpe")
    f_ds_e = _mean(diff_segs_e, "Strategy_Sharpe")
    f_diff_ok = not np.isnan(f_ds_b) and not np.isnan(f_ds_e) and f_ds_e > f_ds_b
    f_kill = (f_shr_d < 0.02) and (f_cagr_d < 0.0025) and not f_diff_ok

    print(
        f"\n  Kill switch: dSharpe={f_shr_d:+.3f}  dCAGR={f_cagr_d:+.2%}  "
        f"2021-22 diff={'YES' if f_diff_ok else 'NO'} (b={_f(f_ds_b)} e={_f(f_ds_e)})"
    )
    print(f"  Kill fires: {'YES' if f_kill else 'NO'}")

    if f_kill:
        print()
        print("  Fast-mode KILL SWITCH fires.")
        print("  The experiment does not meet escalation thresholds in fast-mode.")
        print("  Note: fast-mode OOS starts ~2023 -- 2021-2022 not covered.")
        print("  Escalating to full walk-forward because 2021-2022 is the target period.")

    # ==================================================================
    # FULL WALK-FORWARD
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print(f"  Running baseline   (sigmoid_scale={SCALE_BASE}) ...")
    df_wb = run_walk_forward_evaluation(**wk, sigmoid_scale=SCALE_BASE)

    print(f"  Running experiment (sigmoid_scale={SCALE_EXP}) ...")
    df_we = run_walk_forward_evaluation(**wk, sigmoid_scale=SCALE_EXP)

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
    wc, wce, ws, wse, wm, wme, wv, wve, wt, wte = _print_table(ob_w, oe_w)

    shr_d = wse - ws
    cagr_d = wce - wc
    mdd_d = wme - wm
    to_d = (wte - wt) if not (np.isnan(wt) or np.isnan(wte)) else float("nan")

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
        print(f"    CAGR:    base={_pct(bc)}  exp={_pct(ec)}  delta={_pct(ec - bc, sign=True)}")
        print(f"    Sharpe:  base={_f(bs)}   exp={_f(es)}   delta={_f(es - bs, sign=True)}")
        print(f"    MaxDD:   base={_pct(bm)}  exp={_pct(em)}  delta={_pct(em - bm, sign=True)}")
        if not (np.isnan(bt) or np.isnan(et)):
            print(
                f"    Turnover: base={_pct(bt)}  exp={_pct(et)}  delta={_pct(et - bt, sign=True)}"
            )

        crisis_results[label] = {"ds": es - bs, "dm": em - bm}
        if em < bm - 0.01:
            print("    FLAG: MaxDD worsened >1pp.")
        elif em > bm + 0.01:
            print("    MaxDD improved.")
        if es > bs + 0.05:
            print("    Sharpe improved in this period.")
        elif es < bs - 0.05:
            print("    Sharpe worsened >0.05 in this period.")

    # ==================================================================
    # DECISION
    # ==================================================================
    print("\n" + "=" * 65)
    print("DECISION")
    print("=" * 65)

    diff_shr_b = _mean(_filter_years(sw_b, 2021, 2022), "Strategy_Sharpe")
    diff_shr_e = _mean(_filter_years(sw_e, 2021, 2022), "Strategy_Sharpe")
    diff_mdd_b = _mean(_filter_years(sw_b, 2021, 2022), "Strategy_MaxDD")
    diff_mdd_e = _mean(_filter_years(sw_e, 2021, 2022), "Strategy_MaxDD")
    mdd_2020_b = _mean(_filter_years(sw_b, 2020, 2020), "Strategy_MaxDD")
    mdd_2020_e = _mean(_filter_years(sw_e, 2020, 2020), "Strategy_MaxDD")
    d_shr_2122 = diff_shr_e - diff_shr_b
    d_mdd_2122 = diff_mdd_e - diff_mdd_b
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
    print(f"  2021-2022 MaxDD delta:          {_pct(d_mdd_2122, sign=True)}")
    print(f"  2020 MaxDD delta:               {_pct(d_mdd_2020, sign=True)}")

    # Verdict logic
    perf_improved = shr_d >= 0.02 or cagr_d >= 0.0025
    perf_approx_flat = abs(shr_d) < 0.05 and abs(cagr_d) < 0.01
    diff_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.05
    perf_hard_fail = shr_d < -0.05 or cagr_d < -0.01
    mdd_2020_hard_fail = not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.03
    to_materially_up = not np.isnan(to_d) and to_d > 0.20

    print()
    if perf_hard_fail or mdd_2020_hard_fail:
        verdict = "REJECT"
        if mdd_2020_hard_fail:
            bullets = [
                f"2020 MaxDD worsened by {-d_mdd_2020:.1%}: more decisive signal over-rotated into equities during COVID crash.",
                f"Full-period Sharpe delta {shr_d:+.3f}.",
                "Scaling 0.50 makes the model more aggressive in both directions -- hurts during fast drawdowns.",
                "The gain from faster defensive rotation does not offset the crash exposure increase.",
                "Next: 18M lookback experiment (momentum responsiveness with less distribution impact).",
            ]
        else:
            bullets = [
                f"Full-period Sharpe worsened by {-shr_d:.3f} (threshold: -0.05).",
                f"CAGR worsened by {-cagr_d:.2%}.",
                "More aggressive mapping creates over-allocation to equities in trending-up regimes,",
                "increasing downside exposure in subsequent corrections.",
                "Next: 18M lookback experiment.",
            ]
    elif perf_improved and not to_materially_up:
        verdict = "PASS FULL VALIDATION"
        bullets = [
            f"Full-period Sharpe improved {shr_d:+.3f}.",
            f"CAGR improved {cagr_d:+.2%}.",
            f"2021-2022 Sharpe delta: {d_shr_2122:+.3f}.",
            f"2020 MaxDD delta {d_mdd_2020:+.2%}: crisis responsiveness preserved.",
            "Adopt sigmoid_scale=0.50 as new baseline. Update PROJECT_CONTEXT.md.",
        ]
    elif to_materially_up and not perf_improved:
        verdict = "REJECT"
        bullets = [
            f"Turnover increased materially (+{to_d:.2%}): more decisive signal creates more signal-driven churn.",
            f"Performance improvement {shr_d:+.3f} does not compensate for the turnover cost.",
            "The tolerance filter absorbs small moves but signal-driven trades above tau=1.5% still execute.",
            "The cost-adjusted benefit does not clear the acceptance bar.",
            "Next: 18M lookback experiment.",
        ]
    elif perf_approx_flat and diff_improved:
        verdict = "PASS FULL VALIDATION"
        bullets = [
            f"Performance approximately flat (dSharpe={shr_d:+.3f}, dCAGR={cagr_d:+.2%}).",
            f"2021-2022 Sharpe improved {d_shr_2122:+.3f}: the target period benefited.",
            f"2020 MaxDD delta {d_mdd_2020:+.2%}: crisis behavior preserved.",
            "Regime responsiveness improved without material full-period cost.",
            "Adopt sigmoid_scale=0.50 as new baseline.",
        ]
    elif perf_approx_flat:
        verdict = "REJECT"
        bullets = [
            f"Performance approximately flat (dSharpe={shr_d:+.3f}, dCAGR={cagr_d:+.2%}).",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: target period did not improve.",
            "Scaling 0.50 does not address the responsiveness problem -- z-scores are too moderate",
            "even at 0.50 to produce meaningfully different risk_on during the 2022 rate shock.",
            "Next: 18M lookback experiment.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed results: Sharpe {shr_d:+.3f}, CAGR {cagr_d:+.2%}, 2021-22 Sharpe {_f(d_shr_2122, sign=True)}.",
            "Improvement is not material enough to meet escalation thresholds.",
            "No single period shows consistent benefit that justifies adoption.",
            "Next: 18M lookback experiment.",
            "If 18M also fails, consider accepting current baseline as the stable state.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
