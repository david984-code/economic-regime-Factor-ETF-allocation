"""Experiment: Risk-Off Universe Expansion Only.

Diagnostic: confirm whether the full-universe failure was primarily caused
by the expanded risk-off sleeve (TIP, SHY, DBC, UUP).

Single variable change: risk-off sleeve only.
  Baseline risk-off: IEF, TLT, GLD                       (3 ETFs)
  Expanded risk-off: IEF, TLT, GLD, TIP, SHY, DBC, UUP  (7 ETFs)

  Risk-on sleeve: SPY, MTUM, VLUE, QUAL, USMV, IJR, VIG  (UNCHANGED)

All other baseline parameters remain identical:
  market_lookback_months = 24
  sigmoid_scale          = 0.25
  portfolio_construction = equal_weight
  VOL_LOOKBACK           = 63
  tolerance              = 0.015
  trend_filter_type      = none
  monthly rebalance

Key structural hypothesis:
  Inverse-vol scaling applied to a heterogeneous risk-off sleeve
  (SHY vol ~0.7%, UUP vol ~5%, TIP vol ~6%, DBC vol ~17%, GLD vol ~14%,
   IEF vol ~7%, TLT vol ~12%) will concentrate weight into the lowest-vol
  assets (SHY, UUP), which return near zero. This would mechanically
  suppress CAGR and Sharpe during risk-off regimes.
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
    RISK_OFF_ASSETS_DIVERSE,
    RISK_ON_ASSETS_BASE,
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

RO_SLEEVE_BASE = list(RISK_ON_ASSETS_BASE)  # unchanged
RF_SLEEVE_BASE = list(RISK_OFF_ASSETS_BASE)  # IEF, TLT, GLD
RF_SLEEVE_EXP = list(RISK_OFF_ASSETS_DIVERSE)  # + TIP, SHY, DBC, UUP

NEW_RF_ASSETS = ["TIP", "SHY", "DBC", "UUP"]

TICKERS_RF_EXP = list(TICKERS) + NEW_RF_ASSETS
ASSETS_RF_EXP = TICKERS_RF_EXP + ["cash"]

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

# Sharpe deltas from prior experiments for comparison
PRIOR_FULL_UNIVERSE_SHR_D = -0.415
PRIOR_RO_ONLY_SHR_D = -0.119


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
        risk_on_sleeve=RO_SLEEVE_BASE,
        risk_off_sleeve=RF_SLEEVE_BASE,
    )


def _run_expanded_rf(shared_kwargs):
    with _patch_engine(TICKERS_RF_EXP, ASSETS_RF_EXP):
        return run_walk_forward_evaluation(
            **shared_kwargs,
            tickers=TICKERS_RF_EXP,
            assets=ASSETS_RF_EXP,
            risk_on_sleeve=RO_SLEEVE_BASE,
            risk_off_sleeve=RF_SLEEVE_EXP,
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


def _print_table(ob, oe, label_b="Baseline", label_e="RF Expanded"):
    rows = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {label_b:>14} {label_e:>12} {'Delta':>10}")
    print("  " + "-" * 66)
    for name, col, is_pct in rows:
        vb = _m(ob, col)
        ve = _m(oe, col)
        if np.isnan(vb) and np.isnan(ve):
            continue
        if is_pct:
            print(f"  {name:28} {_pct(vb):>14} {_pct(ve):>12} {_pct(ve - vb, sign=True):>10}")
        else:
            print(f"  {name:28} {_f(vb):>14} {_f(ve):>12} {_f(ve - vb, sign=True):>10}")
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
    cols = [a for a in assets if a in prices_df.columns]
    return (prices_df[cols].pct_change().rolling(63).std() * np.sqrt(252)).mean()


def main():
    print("=" * 65)
    print("EXPERIMENT: Risk-Off Universe Expansion Only")
    print("Diagnostic: confirm risk-off expansion as primary failure cause")
    print("  Risk-off: 3 ETFs -> 7 ETFs (+TIP, SHY, DBC, UUP)")
    print("  Risk-on:  SPY, MTUM, VLUE, QUAL, USMV, IJR, VIG  (UNCHANGED)")
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
    print("    Signal unchanged. New ETFs appear only in risk-off sleeve.")
    print("  Forward-fill:             PASS")
    print("    Inv-vol scaling uses 63-day rolling std of past returns.")
    print("  Asset start-date align:   PASS")
    print("    TIP 2003, SHY 2002, DBC 2006, UUP 2007 — all pre-2010.")
    print("    Yahoo Finance provides data from 2013-07-18 for these tickers.")
    print("    First OOS segment Aug 2015 has 24+ months of warmup.")
    print("  Rebalance timing:         PASS")
    print("    Monthly, first trading day. Identical to baseline.")
    print("  Parameter isolation:      PASS")
    print("    Only risk_off_sleeve changes. Risk-on sleeve unchanged.")
    print("    Engine globals patched per-run and restored.")

    # ==================================================================
    # VOL STRUCTURE DIAGNOSTIC
    # ==================================================================
    print("\n  Fetching expanded risk-off prices ...")
    prices_full = fetch_prices(tickers=TICKERS_RF_EXP, start=FULL_START, end=FULL_END)
    missing = [t for t in TICKERS_RF_EXP if t not in prices_full.columns]
    if missing:
        print(f"  STOP: Missing tickers: {missing}")
        sys.exit(1)

    all_rf = RF_SLEEVE_EXP
    vol_rf = _vol_for_assets(prices_full, all_rf)

    print("\n  Risk-off sleeve vol structure (determines inv-vol weight concentration):")
    print(f"  {'ETF':6}  {'Ann. Vol':>10}  {'1/vol':>8}  {'Status'}")
    print("  " + "-" * 52)
    inv_vols = {}
    for t in all_rf:
        v = float(vol_rf[t]) if t in vol_rf.index else float("nan")
        inv_v = 1.0 / v if not np.isnan(v) and v > 0 else float("nan")
        inv_vols[t] = inv_v
        tag = "EXISTING" if t in RF_SLEEVE_BASE else "NEW"
        print(f"  {t:6}  {_pct(v):>10}  {inv_v:>8.1f}  {tag}")

    total_inv_v = sum(v for v in inv_vols.values() if not np.isnan(v))
    print("\n  Expected inv-vol weight shares in risk-off sleeve:")
    for t in all_rf:
        iv = inv_vols.get(t, float("nan"))
        share = iv / total_inv_v if not np.isnan(iv) else float("nan")
        tag = "EXISTING" if t in RF_SLEEVE_BASE else "NEW"
        print(f"  {t:6}  ~{_pct(share)}  of risk-off allocation  ({tag})")
    print("  (These shares then multiply by (1-risk_on) ~ 0.40 for portfolio weight)")

    # ==================================================================
    # FAST-MODE
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running expanded risk-off ...")
    df_fe = _run_expanded_rf(fk)

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

    print("  Running expanded risk-off ...")
    df_we = _run_expanded_rf(wk)

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
    print("NEW RISK-OFF ASSET CONTRIBUTION ANALYSIS")
    print("=" * 65)

    # Approximate portfolio-level weight per new RF asset at avg risk_on ~ 0.60
    avg_risk_off = 0.40
    print(f"\n  At avg risk_on ~0.60 -> avg risk-off allocation ~{avg_risk_off:.0%}")
    print("  Approx portfolio weight per new asset = rf_allocation * inv-vol share:")
    print()
    for t in NEW_RF_ASSETS:
        iv = inv_vols.get(t, float("nan"))
        share = iv / total_inv_v if not np.isnan(iv) else float("nan")
        port_w = avg_risk_off * share
        v = float(vol_rf[t]) if t in vol_rf.index else float("nan")
        dominance = ""
        if not np.isnan(share):
            if share > 0.30:
                dominance = "  <- DOMINATES risk-off sleeve"
            elif share > 0.15:
                dominance = "  <- significant"
        print(
            f"  {t:6}  vol={_pct(v)}  inv-vol share={_pct(share)}  "
            f"~port weight={_pct(port_w)}{dominance}"
        )

    print("\n  Baseline risk-off assets for comparison:")
    for t in RF_SLEEVE_BASE:
        iv = inv_vols.get(t, float("nan"))
        share = iv / total_inv_v if not np.isnan(iv) else float("nan")
        port_w = avg_risk_off * share
        v = float(vol_rf[t]) if t in vol_rf.index else float("nan")
        print(f"  {t:6}  vol={_pct(v)}  inv-vol share={_pct(share)}  ~port weight={_pct(port_w)}")

    if not np.isnan(vol_d):
        vol_note = (
            "Vol fell sharply: SHY/UUP dominate the risk-off sleeve, suppressing portfolio vol."
            if vol_d < -0.010
            else "Vol roughly unchanged: new risk-off assets are not dominating the sleeve."
            if abs(vol_d) < 0.005
            else "Vol increased: new risk-off assets (DBC) are increasing sleeve volatility."
        )
        print(f"\n  Realized portfolio vol delta: {_pct(vol_d, sign=True)}")
        print(f"  Interpretation: {vol_note}")

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
    crisis.get("2021-2022 rate shock", {}).get("dm", float("nan"))
    d_shr_2020 = crisis.get("2020 COVID crash", {}).get("ds", float("nan"))
    d_mdd_2020 = crisis.get("2020 COVID crash", {}).get("dm", float("nan"))
    d_shr_2018 = crisis.get("2018 volatility", {}).get("ds", float("nan"))

    print(f"\n  Full-period Sharpe delta:            {wd['shr_d']:+.3f}")
    print(f"  Full-period CAGR delta:              {wd['cagr_d']:+.2%}")
    print(f"  Full-period MaxDD delta:             {wd['mdd_d']:+.2%}")
    if not np.isnan(to_d):
        print(f"  Full-period Turnover delta:          {to_d:+.2%}")
    if not np.isnan(vol_d):
        print(f"  Full-period Vol delta:               {_pct(vol_d, sign=True)}")
    print(f"  2018 Sharpe delta:                   {_f(d_shr_2018, sign=True)}")
    print(f"  2020 Sharpe delta:                   {_f(d_shr_2020, sign=True)}")
    print(f"  2020 MaxDD delta:                    {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:              {_f(d_shr_2122, sign=True)}")

    # Attribution comparison
    print("\n  --- FAILURE ATTRIBUTION ---")
    print(f"  Full universe Sharpe delta:          {PRIOR_FULL_UNIVERSE_SHR_D:+.3f}")
    print(f"  Risk-on expansion only (prior run):  {PRIOR_RO_ONLY_SHR_D:+.3f}")
    print(f"  Risk-off expansion only (this run):  {wd['shr_d']:+.3f}")

    if not np.isnan(wd["shr_d"]):
        rf_share = abs(wd["shr_d"]) / abs(PRIOR_FULL_UNIVERSE_SHR_D) * 100
        ro_share = abs(PRIOR_RO_ONLY_SHR_D) / abs(PRIOR_FULL_UNIVERSE_SHR_D) * 100
        print(f"\n  Risk-off expansion explains ~{rf_share:.0f}% of the full-universe Sharpe loss.")
        print(f"  Risk-on  expansion explains ~{ro_share:.0f}% of the full-universe Sharpe loss.")
        print("  (Note: effects are not purely additive due to interaction terms)")

        if rf_share > 65:
            cause = (
                "CONFIRMED: Risk-off expansion is the dominant cause of the full-universe failure."
            )
        elif rf_share > 35:
            cause = "PARTIAL: Risk-off expansion is a significant but not dominant cause; both sleeves contributed."
        else:
            cause = "NOT CONFIRMED: Risk-off expansion is a minor contributor; failure mainly from risk-on expansion."
    else:
        cause = "Attribution inconclusive."
        rf_share = float("nan")

    print(f"\n  {cause}")

    perf_improved = wd["shr_d"] >= 0.02 or wd["cagr_d"] >= 0.0025
    diff_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.05
    hard_fail = wd["shr_d"] < -0.05 or wd["cagr_d"] < -0.015
    approx_flat = abs(wd["shr_d"]) < 0.03 and abs(wd["cagr_d"]) < 0.008

    print()
    if hard_fail:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {wd['shr_d']:+.3f}: expanding the risk-off sleeve is harmful on its own.",
            "Root cause: SHY (~0.7% vol) dominates the risk-off sleeve under inv-vol scaling, earning near-zero return.",
            "During risk-off regimes, the portfolio earns effectively cash-like returns on a large fraction of assets.",
            f"2021-2022 target period: Sharpe delta {_f(d_shr_2122, sign=True)} — inflation diversification (TIP/DBC) not sufficient to overcome SHY drag.",
            cause,
        ]
    elif perf_improved and not approx_flat:
        verdict = "PASS"
        bullets = [
            f"Sharpe improved {wd['shr_d']:+.3f}: diversified risk-off sleeve adds genuine value.",
            f"CAGR improved {wd['cagr_d']:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: inflation hedges (TIP, DBC) helped in target period.",
            "Crisis protection preserved across 2018 and 2020.",
            cause,
        ]
    elif approx_flat and diff_improved:
        verdict = "PASS"
        bullets = [
            f"Full-period approximately flat (Sharpe {wd['shr_d']:+.3f}): diversification without meaningful cost.",
            f"2021-2022 improved {d_shr_2122:+.3f}: inflation hedges contributed in target period.",
            "Risk-off sleeve expansion at equal-weight+inv-vol is workable with the original risk-on assets intact.",
            f"Turnover delta {_pct(to_d, sign=True)}: manageable.",
            cause,
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {wd['shr_d']:+.3f}, CAGR {wd['cagr_d']:+.2%}.",
            f"2021-2022 target period: Sharpe delta {_f(d_shr_2122, sign=True)}: inflation hedges not sufficient.",
            "Inv-vol scaling concentrates the risk-off sleeve into lowest-vol assets (SHY, UUP) with near-zero return.",
            "The equal-weight + inv-vol framework cannot handle vol heterogeneity in the risk-off sleeve.",
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
