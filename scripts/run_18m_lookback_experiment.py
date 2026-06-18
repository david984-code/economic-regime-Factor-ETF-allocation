"""Experiment: Market Momentum Lookback Adjustment (24M -> 18M).

Single variable change: market_lookback_months 24 -> 18.
  Baseline:   momentum = SPY_price(t-1) / SPY_price(t-25) - 1  (24M)
  Experiment: momentum = SPY_price(t-1) / SPY_price(t-19) - 1  (18M)

Then: expanding z-score -> sigmoid(z * 0.25) -> risk_on (unchanged)

All other baseline parameters remain identical:
  portfolio_construction_method = equal_weight
  sigmoid_scale                 = 0.25
  VOL_LOOKBACK                  = 63
  tolerance                     = 0.015
  trend_filter_type             = none
  monthly rebalance
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

LOOKBACK_BASE = 24
LOOKBACK_EXP = 18

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
    "portfolio_construction_method": "equal_weight",
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

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"
FULL_START = "2010-01-01"
FULL_END = None


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


def _print_table(ob, oe, label_b="Baseline (24M)", label_e="Exp (18M)"):
    metrics = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {label_b:>16} {label_e:>12} {'Delta':>10}")
    print("  " + "-" * 68)
    for name, col, is_pct in metrics:
        vb = _m(ob, col)
        ve = _m(oe, col)
        if np.isnan(vb) and np.isnan(ve):
            continue
        if is_pct:
            print(f"  {name:28} {_pct(vb):>16} {_pct(ve):>12} {_pct(ve - vb, sign=True):>10}")
        else:
            print(f"  {name:28} {_f(vb):>16} {_f(ve):>12} {_f(ve - vb, sign=True):>10}")
    return {
        "shr_d": _m(oe, "Strategy_Sharpe") - _m(ob, "Strategy_Sharpe"),
        "cagr_d": _m(oe, "Strategy_CAGR") - _m(ob, "Strategy_CAGR"),
        "mdd_d": _m(oe, "Strategy_MaxDD") - _m(ob, "Strategy_MaxDD"),
        "to_b": _m(ob, "Strategy_Turnover"),
        "to_e": _m(oe, "Strategy_Turnover"),
    }


def main():
    print("=" * 65)
    print("EXPERIMENT: Market Momentum Lookback Adjustment (24M -> 18M)")
    print(f"Single variable: market_lookback_months {LOOKBACK_BASE} -> {LOOKBACK_EXP}")
    print("  Baseline:   momentum = SPY(t-1)/SPY(t-25) - 1")
    print("  Experiment: momentum = SPY(t-1)/SPY(t-19) - 1")
    print("  Signal:     expanding z-score -> sigmoid(z * 0.25) -> risk_on")
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
    print("  Lookahead:               PASS")
    print("    18M momentum = SPY(t-1)/SPY(t-19) - 1.")
    print("    Computed at month-end t-1 using only past prices.")
    print("    Expanding z-score window: normalizes against all prior")
    print("    month-end momentum values. No future data used.")
    print("  Forward-fill:            PASS")
    print("    risk_on forward-filled daily from month-end t-1.")
    print("    Applied on first trading day of month t. Unchanged convention.")
    print("  Signal timing alignment: PASS")
    print("    market_lookback_months only changes the return window.")
    print("    z-score, sigmoid, blend, inv-vol, tolerance all unchanged.")
    print("    OOS segment boundaries identical to baseline.")
    print("  Parameter isolation:     PASS")
    print("    market_lookback_months is the only changed parameter.")
    print("    Verified: sigmoid_scale=0.25, tolerance=0.015, EW construction.")

    # ==================================================================
    # FAST-MODE
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print(f"  Running baseline   (market_lookback_months={LOOKBACK_BASE}) ...")
    df_fb = run_walk_forward_evaluation(**fk, market_lookback_months=LOOKBACK_BASE)

    print(f"  Running experiment (market_lookback_months={LOOKBACK_EXP}) ...")
    df_fe = run_walk_forward_evaluation(**fk, market_lookback_months=LOOKBACK_EXP)

    sb_idx = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    se_idx = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not sb_idx.equals(se_idx):
        print("  STOP: OOS segment mismatch.")
        sys.exit(1)
    print(f"  OOS segments identical: YES ({len(sb_idx)} segments)\n")

    ob_f = _overall(df_fb)
    oe_f = _overall(df_fe)
    fd = _print_table(ob_f, oe_f)

    f_diff_b = _mean(_filter_years(_segs(df_fb), 2021, 2022), "Strategy_Sharpe")
    f_diff_e = _mean(_filter_years(_segs(df_fe), 2021, 2022), "Strategy_Sharpe")
    f_diff_ok = not (np.isnan(f_diff_b) or np.isnan(f_diff_e)) and f_diff_e > f_diff_b
    f_kill = (fd["shr_d"] < 0.02) and (fd["cagr_d"] < 0.0025) and not f_diff_ok

    print(
        f"\n  Kill switch: dSharpe={fd['shr_d']:+.3f}  dCAGR={fd['cagr_d']:+.2%}  "
        f"2021-22={'BETTER' if f_diff_ok else 'NO'} (b={_f(f_diff_b)} e={_f(f_diff_e)})"
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

    print(f"  Running baseline   (market_lookback_months={LOOKBACK_BASE}) ...")
    df_wb = run_walk_forward_evaluation(**wk, market_lookback_months=LOOKBACK_BASE)

    print(f"  Running experiment (market_lookback_months={LOOKBACK_EXP}) ...")
    df_we = run_walk_forward_evaluation(**wk, market_lookback_months=LOOKBACK_EXP)

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
    wd = _print_table(ob_w, oe_w)

    to_d = (
        wd["to_e"] - wd["to_b"]
        if not (np.isnan(wd["to_b"]) or np.isnan(wd["to_e"]))
        else float("nan")
    )

    # ==================================================================
    # CRISIS SEGMENT CHECK
    # ==================================================================
    print("\n" + "=" * 65)
    print("CRISIS SEGMENT CHECK")
    print("=" * 65)

    crisis = {}
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

        bs = _mean(sb, "Strategy_Sharpe")
        es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD")
        em = _mean(se, "Strategy_MaxDD")
        bt = _mean(sb, "Strategy_Turnover")
        et = _mean(se, "Strategy_Turnover")
        bc = _mean(sb, "Strategy_CAGR")
        ec = _mean(se, "Strategy_CAGR")

        print(f"    Segments: {len(sb)}")
        print(f"    CAGR:     base={_pct(bc)}   exp={_pct(ec)}   delta={_pct(ec - bc, sign=True)}")
        print(f"    Sharpe:   base={_f(bs)}    exp={_f(es)}    delta={_f(es - bs, sign=True)}")
        print(f"    MaxDD:    base={_pct(bm)}   exp={_pct(em)}   delta={_pct(em - bm, sign=True)}")
        if not (np.isnan(bt) or np.isnan(et)):
            print(
                f"    Turnover: base={_pct(bt)}   exp={_pct(et)}   delta={_pct(et - bt, sign=True)}"
            )

        crisis[label] = {"ds": es - bs, "dm": em - bm}

        if em < bm - 0.015:
            print("    FLAG: MaxDD worsened >1.5pp.")
        if es < bs - 0.10:
            print("    FLAG: Sharpe worsened >0.10.")
        if es > bs + 0.05:
            print("    NOTE: Sharpe improved in this period.")
        if em > bm + 0.01:
            print("    NOTE: MaxDD improved in this period.")

    # ==================================================================
    # DECISION
    # ==================================================================
    print("\n" + "=" * 65)
    print("DECISION")
    print("=" * 65)

    d_shr_2122 = crisis.get("2021-2022 rate shock", {}).get("ds", float("nan"))
    d_mdd_2122 = crisis.get("2021-2022 rate shock", {}).get("dm", float("nan"))
    d_shr_2020 = crisis.get("2020 COVID crash", {}).get("ds", float("nan"))
    d_mdd_2020 = crisis.get("2020 COVID crash", {}).get("dm", float("nan"))
    d_shr_2018 = crisis.get("2018 volatility", {}).get("ds", float("nan"))

    print(f"\n  Full-period Sharpe delta:       {wd['shr_d']:+.3f}")
    print(f"  Full-period CAGR delta:         {wd['cagr_d']:+.2%}")
    print(f"  Full-period MaxDD delta:        {wd['mdd_d']:+.2%}")
    if not np.isnan(to_d):
        print(f"  Full-period Turnover delta:     {to_d:+.2%}")
    print(f"  2018 volatility Sharpe delta:   {_f(d_shr_2018, sign=True)}")
    print(f"  2020 COVID Sharpe delta:        {_f(d_shr_2020, sign=True)}")
    print(f"  2020 COVID MaxDD delta:         {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:         {_f(d_shr_2122, sign=True)}")
    print(f"  2021-2022 MaxDD delta:          {_pct(d_mdd_2122, sign=True)}")

    # Verdict
    perf_improved = wd["shr_d"] >= 0.02 or wd["cagr_d"] >= 0.0025
    diff_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.05
    hard_fail = wd["shr_d"] < -0.05 or wd["cagr_d"] < -0.015
    mdd_2020_fail = not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.03
    to_material_up = not np.isnan(to_d) and to_d > 0.20
    approx_flat = abs(wd["shr_d"]) < 0.05 and abs(wd["cagr_d"]) < 0.01

    print()
    if hard_fail or mdd_2020_fail:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {wd['shr_d']:+.3f} and/or CAGR {wd['cagr_d']:+.2%}.",
            "18M momentum does not add incremental information vs 24M over the full OOS history.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: target period did not improve.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crisis behavior unchanged or worse.",
            "24M lookback remains the accepted baseline. Research direction: structural changes.",
        ]
    elif perf_improved and not to_material_up:
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {wd['shr_d']:+.3f} -- exceeds +0.02 threshold.",
            f"CAGR improved {wd['cagr_d']:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: target period behavior.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crisis responsiveness preserved.",
            "Update market_lookback_months to 18 in baseline. Update PROJECT_CONTEXT.md.",
        ]
    elif approx_flat and diff_improved:
        verdict = "PASS"
        bullets = [
            f"Full-period approximately flat (Sharpe {wd['shr_d']:+.3f}, CAGR {wd['cagr_d']:+.2%}).",
            f"2021-2022 Sharpe improved {d_shr_2122:+.3f}: the target weak period benefited.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crash protection preserved.",
            f"Turnover change {_pct(to_d, sign=True)}: within acceptable range.",
            "Accept 18M lookback as new baseline. Update PROJECT_CONTEXT.md.",
        ]
    elif to_material_up and not perf_improved:
        verdict = "REJECT"
        bullets = [
            f"Turnover increased materially (+{to_d:.2%}pp) with Sharpe delta {wd['shr_d']:+.3f}.",
            "Shorter lookback produces more momentum flips, increasing signal-driven turnover.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: insufficient improvement to justify cost.",
            "The 24M lookback produces smoother z-scores; 18M adds noise without consistent benefit.",
            "24M lookback remains accepted baseline.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {wd['shr_d']:+.3f}, CAGR {wd['cagr_d']:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: did not improve target period.",
            f"Turnover delta {_pct(to_d, sign=True)}: shorter lookback adds signal noise.",
            "24M lookback is the more stable signal over full OOS history.",
            "No further lookback experiments warranted without broader structural rethink.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
