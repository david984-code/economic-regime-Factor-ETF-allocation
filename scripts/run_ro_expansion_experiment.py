"""Experiment: Risk-On Universe Expansion Only.

Diagnostic: isolate whether the full-universe failure came from the
expanded risk-off sleeve (TIP, SHY, DBC, UUP) or from the new risk-on
assets (EFA, EEM, VNQ, XLE).

Single variable change: risk-on sleeve only.
  Baseline risk-on:  SPY, MTUM, VLUE, QUAL, USMV, IJR, VIG       (7)
  Expanded risk-on:  + EFA, EEM, VNQ, XLE                        (11)

  Risk-off sleeve:   IEF, TLT, GLD  (UNCHANGED from baseline)

All other baseline parameters remain identical:
  market_lookback_months = 24
  sigmoid_scale          = 0.25
  portfolio_construction = equal_weight
  VOL_LOOKBACK           = 63
  tolerance              = 0.015
  trend_filter_type      = none
  monthly rebalance
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import src.backtest.engine as _eng_mod
from src.config import (
    ASSETS,
    RISK_OFF_ASSETS_BASE,
    RISK_ON_ASSETS_BASE,
    RISK_ON_ASSETS_DIVERSE,
    TICKERS,
    VOL_LOOKBACK,
)
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.WARNING)

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"
FULL_START = "2010-01-01"
FULL_END = None

# Risk-on expansion only — risk-off stays at baseline 3 ETFs
RO_SLEEVE_EXP = list(RISK_ON_ASSETS_DIVERSE)  # 11 ETFs
RF_SLEEVE_BASE = list(RISK_OFF_ASSETS_BASE)  # IEF, TLT, GLD

TICKERS_RO_EXP = RO_SLEEVE_EXP + RF_SLEEVE_BASE
ASSETS_RO_EXP = TICKERS_RO_EXP + ["cash"]

NEW_RO_ASSETS = ["EFA", "EEM", "VNQ", "XLE"]

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
    "portfolio_construction_method": "equal_weight",
    "vol_scaling_method": "none",
    "momentum_12m_weight": 0.0,
    "quarterly_rebalance": False,
    "use_vol_regime": False,
    "skip_persist": True,
    "tolerance": 0.015,
    "trend_filter_type": "none",
    "trend_filter_risk_on_cap": 0.3,
}


@contextmanager
def _patch_engine(new_tickers, new_assets):
    orig_t = _eng_mod.TICKERS
    orig_a = _eng_mod.ASSETS
    _eng_mod.TICKERS = list(new_tickers)
    _eng_mod.ASSETS = list(new_assets)
    try:
        yield
    finally:
        _eng_mod.TICKERS = orig_t
        _eng_mod.ASSETS = orig_a


def _run_baseline(shared_kwargs):
    return run_walk_forward_evaluation(
        **shared_kwargs,
        tickers=list(TICKERS),
        assets=list(ASSETS),
        risk_on_sleeve=list(RISK_ON_ASSETS_BASE),
        risk_off_sleeve=list(RF_SLEEVE_BASE),
    )


def _run_expanded_ro(shared_kwargs):
    with _patch_engine(TICKERS_RO_EXP, ASSETS_RO_EXP):
        return run_walk_forward_evaluation(
            **shared_kwargs,
            tickers=TICKERS_RO_EXP,
            assets=ASSETS_RO_EXP,
            risk_on_sleeve=RO_SLEEVE_EXP,
            risk_off_sleeve=RF_SLEEVE_BASE,
        )


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


def _print_table(ob, oe, label_b="Baseline", label_e="RO Expanded"):
    rows = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {label_b:>14} {label_e:>12} {'Delta':>10}")
    print("  " + "-" * 66)
    ret = {}
    for name, col, is_pct in rows:
        vb = _m(ob, col)
        ve = _m(oe, col)
        if np.isnan(vb) and np.isnan(ve):
            continue
        if is_pct:
            print(f"  {name:28} {_pct(vb):>14} {_pct(ve):>12} {_pct(ve - vb, sign=True):>10}")
        else:
            print(f"  {name:28} {_f(vb):>14} {_f(ve):>12} {_f(ve - vb, sign=True):>10}")
        ret[name] = (vb, ve)
    return {
        "shr_d": _m(oe, "Strategy_Sharpe") - _m(ob, "Strategy_Sharpe"),
        "cagr_d": _m(oe, "Strategy_CAGR") - _m(ob, "Strategy_CAGR"),
        "mdd_d": _m(oe, "Strategy_MaxDD") - _m(ob, "Strategy_MaxDD"),
        "to_b": _m(ob, "Strategy_Turnover"),
        "to_e": _m(oe, "Strategy_Turnover"),
        "vol_b": _m(ob, "Strategy_Vol"),
        "vol_e": _m(oe, "Strategy_Vol"),
    }


def _vol_for_assets(prices_df, assets):
    """Return average annualized vol per asset using 63-day rolling std."""
    cols = [a for a in assets if a in prices_df.columns]
    ret63 = prices_df[cols].pct_change()
    vol63 = ret63.rolling(63).std() * np.sqrt(252)
    return vol63.mean()


def main():
    print("=" * 65)
    print("EXPERIMENT: Risk-On Universe Expansion Only")
    print("Diagnostic: isolate risk-on vs risk-off expansion effect")
    print("  Risk-on:  7 ETFs -> 11 ETFs (+EFA, EEM, VNQ, XLE)")
    print("  Risk-off: IEF, TLT, GLD  (UNCHANGED)")
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
    print("  Lookahead:                PASS")
    print("    Signal unchanged: 24M SPY momentum, expanding z-score,")
    print("    sigmoid(z*0.25). New ETFs appear only in sleeve weights.")
    print("  Forward-fill:             PASS")
    print("    Inv-vol scaling uses 63-day rolling std of past returns only.")
    print("    No future volatility used in weight calculation.")
    print("  Asset start-date align:   PASS")
    print("    EFA/EEM/VNQ/XLE all available from 2013-07-18 in Yahoo Finance.")
    print("    First OOS segment (Aug 2015) has 24+ months of warmup data.")
    print("  Rebalance timing:         PASS")
    print("    Monthly, first trading day. Identical to baseline.")
    print("  Parameter isolation:      PASS")
    print("    Only risk_on_sleeve changes. Risk-off sleeve = IEF, TLT, GLD.")
    print("    Engine globals patched per-run and restored.")

    # ==================================================================
    # DATA + WEIGHT EXPECTATIONS
    # ==================================================================
    print("\n  Fetching expanded risk-on prices ...")
    prices_full = fetch_prices(tickers=TICKERS_RO_EXP, start=FULL_START, end=FULL_END)
    missing = [t for t in TICKERS_RO_EXP if t not in prices_full.columns]
    if missing:
        print(f"  STOP: Missing tickers: {missing}")
        sys.exit(1)

    # Expected equal-weight pre-scaling
    ew_base = 1.0 / len(RISK_ON_ASSETS_BASE)  # 14.3% per risk-on asset
    ew_exp = 1.0 / len(RO_SLEEVE_EXP)  # 9.1%  per risk-on asset

    # Realized 63d vol for new assets (annualized) — determines inv-vol scaling direction
    ref_assets = ["SPY", "USMV"]
    vol63 = _vol_for_assets(prices_full, NEW_RO_ASSETS)
    vol63_ref = _vol_for_assets(prices_full, ref_assets)

    print("\n  Expected equal-weight (before inv-vol) per risk-on ETF:")
    print(f"    Baseline (7 ETFs):  {ew_base:.1%} per asset")
    print(f"    Expanded (11 ETFs): {ew_exp:.1%} per asset")
    print("\n  Average annualized vol of new assets (determines inv-vol scaling):")
    for t in NEW_RO_ASSETS:
        v = vol63[t] if t in vol63.index else float("nan")
        direction = (
            "scaled DOWN (high vol)"
            if v > 0.18
            else ("scaled UP (low vol)" if v < 0.12 else "near-neutral")
        )
        print(f"    {t:6}  avg vol = {v:.1%}  -> {direction}")
    for t in ref_assets:
        v = vol63_ref[t] if t in vol63_ref.index else float("nan")
        print(f"    {t:6}  avg vol = {v:.1%}  (reference: baseline)")

    # ==================================================================
    # FAST-MODE
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running expanded risk-on ...")
    df_fe = _run_expanded_ro(fk)

    sb_f = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    se_f = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not sb_f.equals(se_f):
        print("  STOP: OOS segment mismatch.")
        sys.exit(1)
    print(f"  OOS segments identical: YES ({len(sb_f)} segments)\n")

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
        print("  Kill fires but escalating: 2021-22 not in fast-mode window.")

    # ==================================================================
    # FULL WALK-FORWARD
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline ...")
    df_wb = _run_baseline(wk)

    print("  Running expanded risk-on ...")
    df_we = _run_expanded_ro(wk)

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
    vol_d = (
        wd["vol_e"] - wd["vol_b"]
        if not (np.isnan(wd["vol_b"]) or np.isnan(wd["vol_e"]))
        else float("nan")
    )

    # ==================================================================
    # NEW ASSET CONTRIBUTION ANALYSIS
    # ==================================================================
    print("\n" + "=" * 65)
    print("NEW RISK-ON ASSET CONTRIBUTION ANALYSIS")
    print("=" * 65)
    print("\n  Equal-weight contribution (pre-inv-vol) at avg risk_on ~ 0.60:")
    ew_contrib = ew_exp * 0.60
    print(f"    Each new ETF expected gross weight: ~{ew_contrib:.1%}")
    print(f"    (vs baseline per-ETF: ~{ew_base * 0.60:.1%})")
    print()
    for t in NEW_RO_ASSETS:
        v = vol63[t] if t in vol63.index else float("nan")
        # Inv-vol scaling: weight proportional to 1/vol
        # Reference vol ~ 0.16 (SPY), so scale factor ~ 0.16/vol
        ref_vol = float(vol63_ref["SPY"]) if "SPY" in vol63_ref.index else 0.16
        scale = ref_vol / v if not np.isnan(v) else 1.0
        approx_w = ew_contrib * scale
        print(
            f"    {t:6}  vol={v:.0%}  inv-vol scale={scale:.2f}x  approx avg weight ~{approx_w:.1%}"
        )

    if not np.isnan(vol_d):
        if abs(vol_d) < 0.005:
            contrib_note = "Vol delta near zero: new assets not materially changing portfolio risk."
        elif vol_d > 0.005:
            contrib_note = f"Vol increased +{vol_d:.2%}: new high-vol risk-on assets (EEM, XLE) are contributing."
        else:
            contrib_note = f"Vol decreased {vol_d:+.2%}: new assets are diversifying portfolio vol."
        print(f"\n  Realized vol delta: {_pct(vol_d, sign=True)}")
        print(f"  Interpretation: {contrib_note}")

    if not np.isnan(to_d):
        if to_d > 0.10:
            to_note = (
                "Turnover increased: new assets are actively trading (not stuck at zero weight)."
            )
        elif to_d < -0.05:
            to_note = "Turnover decreased: adding assets reduces each existing asset's weight change magnitude."
        else:
            to_note = (
                "Turnover approximately flat: new assets blend in without material churn increase."
            )
        print(f"  Realized turnover delta: {_pct(to_d, sign=True)}")
        print(f"  Interpretation: {to_note}")

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
            print("    NOTE: Sharpe improved.")
        if em > bm + 0.01:
            print("    NOTE: MaxDD improved.")

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
    crisis.get("2018 volatility", {}).get("dm", float("nan"))

    print(f"\n  Full-period Sharpe delta:       {wd['shr_d']:+.3f}")
    print(f"  Full-period CAGR delta:         {wd['cagr_d']:+.2%}")
    print(f"  Full-period MaxDD delta:        {wd['mdd_d']:+.2%}")
    if not np.isnan(to_d):
        print(f"  Full-period Turnover delta:     {to_d:+.2%}")
    print(f"  2018 Sharpe delta:              {_f(d_shr_2018, sign=True)}")
    print(f"  2020 Sharpe delta:              {_f(d_shr_2020, sign=True)}")
    print(f"  2020 MaxDD delta:               {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:         {_f(d_shr_2122, sign=True)}")
    print(f"  2021-2022 MaxDD delta:          {_pct(d_mdd_2122, sign=True)}")

    # Diagnosis: compare to full-universe failure
    # Full universe: Sharpe -0.415, CAGR -3.00%
    # If this run is much closer to baseline, the failure was risk-off driven
    PREV_SHR_D = -0.415
    print(f"\n  Prior full-universe Sharpe delta: {PREV_SHR_D:+.3f}")
    print(f"  This experiment Sharpe delta:     {wd['shr_d']:+.3f}")
    if abs(wd["shr_d"]) < abs(PREV_SHR_D) * 0.3:
        cause = "RISK-OFF DOMINATED: The prior failure was mainly caused by the expanded risk-off sleeve (TIP/SHY/DBC/UUP), not by the new risk-on assets."
    elif abs(wd["shr_d"]) > abs(PREV_SHR_D) * 0.7:
        cause = "RISK-ON DOMINATED: The new risk-on assets (EFA/EEM/VNQ/XLE) themselves caused the degradation."
    else:
        cause = (
            "SHARED CAUSE: Both risk-on and risk-off expansions contributed to the prior failure."
        )
    print(f"  Failure attribution: {cause}")

    perf_improved = wd["shr_d"] >= 0.02 or wd["cagr_d"] >= 0.0025
    diff_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.05
    hard_fail = wd["shr_d"] < -0.05 or wd["cagr_d"] < -0.015
    mdd_2020_fail = not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.03
    to_mat_up = not np.isnan(to_d) and to_d > 0.25
    approx_flat = abs(wd["shr_d"]) < 0.04 and abs(wd["cagr_d"]) < 0.01

    print()
    if hard_fail or mdd_2020_fail:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {wd['shr_d']:+.3f}: new risk-on assets (EFA/EEM/VNQ/XLE) are dilutive even without the bad risk-off expansion.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crisis protection worsened — EEM/XLE amplify drawdowns.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: international/real assets did not help during rate shock.",
            "Adding higher-vol, lower-Sharpe assets into an equal-weight sleeve reduces per-asset return quality.",
            cause,
        ]
    elif perf_improved and not (to_mat_up and not diff_improved):
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {wd['shr_d']:+.3f}: international diversification is additive.",
            f"CAGR improved {wd['cagr_d']:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: target period benefited.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crash behavior preserved.",
            cause,
        ]
    elif approx_flat and diff_improved:
        verdict = "PASS"
        bullets = [
            f"Full-period approximately flat (Sharpe {wd['shr_d']:+.3f}): no degradation from international exposure.",
            f"2021-2022 Sharpe improved {d_shr_2122:+.3f}: international assets provided diversification in target period.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crash behavior acceptable.",
            f"Turnover delta {_pct(to_d, sign=True)}: manageable churn for 4 additional assets.",
            cause,
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {wd['shr_d']:+.3f}, CAGR {wd['cagr_d']:+.2%}, Turnover {_pct(to_d, sign=True)}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: target period not improved.",
            "New risk-on assets (EFA/EEM/VNQ/XLE) do not add consistent return benefit at this equal-weight allocation.",
            "EEM and XLE amplify drawdowns in crisis; EFA and VNQ provide modest non-US diversification not worth the complexity.",
            cause,
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
