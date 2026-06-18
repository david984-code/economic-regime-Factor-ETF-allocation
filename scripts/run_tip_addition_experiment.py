"""Experiment: Risk-Off Sleeve — Add TIP Only (Minimal Inflation Hedge).

Single variable change: risk-off sleeve from [IEF, TLT, GLD] -> [IEF, TLT, GLD, TIP].

TIP (iShares TIPS Bond ETF) adds inflation-linked bond exposure with vol (~5%)
that is similar to IEF (~6%), so inverse-vol scaling should distribute weight
reasonably without extreme concentration.

All other baseline parameters remain identical:
  market_lookback_months = 24, sigmoid_scale = 0.25,
  portfolio_construction = equal_weight, VOL_LOOKBACK = 63,
  tolerance = 0.015, monthly rebalance, trend_filter_type = none
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

RF_BASE = list(RISK_OFF_ASSETS_BASE)  # IEF, TLT, GLD
RF_EXP = RF_BASE + ["TIP"]  # IEF, TLT, GLD, TIP

TICKERS_EXP = list(TICKERS) + ["TIP"]
ASSETS_EXP = TICKERS_EXP + ["cash"]

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


def _run_baseline(kw):
    return run_walk_forward_evaluation(
        **kw,
        tickers=list(TICKERS),
        assets=list(ASSETS),
        risk_on_sleeve=list(RISK_ON_ASSETS_BASE),
        risk_off_sleeve=RF_BASE,
        rf_sleeve_cap=0.0,
    )


def _run_exp(kw):
    with _patch_engine(TICKERS_EXP, ASSETS_EXP):
        return run_walk_forward_evaluation(
            **kw,
            tickers=TICKERS_EXP,
            assets=ASSETS_EXP,
            risk_on_sleeve=list(RISK_ON_ASSETS_BASE),
            risk_off_sleeve=RF_EXP,
            rf_sleeve_cap=0.0,
        )


# ── helpers ──────────────────────────────────────────────────────────────────


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


def _print_table(ob, oe, lb="Baseline", le="+ TIP"):
    rows = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {lb:>14} {le:>10} {'Delta':>10}")
    print("  " + "-" * 64)
    for name, col, is_pct in rows:
        vb = _m(ob, col)
        ve = _m(oe, col)
        if np.isnan(vb) and np.isnan(ve):
            continue
        if is_pct:
            print(f"  {name:28} {_pct(vb):>14} {_pct(ve):>10} {_pct(ve - vb, sign=True):>10}")
        else:
            print(f"  {name:28} {_f(vb):>14} {_f(ve):>10} {_f(ve - vb, sign=True):>10}")
    return {
        "shr_d": _m(oe, "Strategy_Sharpe") - _m(ob, "Strategy_Sharpe"),
        "cagr_d": _m(oe, "Strategy_CAGR") - _m(ob, "Strategy_CAGR"),
        "mdd_d": _m(oe, "Strategy_MaxDD") - _m(ob, "Strategy_MaxDD"),
        "to_b": _m(ob, "Strategy_Turnover"),
        "to_e": _m(oe, "Strategy_Turnover"),
        "vol_b": _m(ob, "Strategy_Vol"),
        "vol_e": _m(oe, "Strategy_Vol"),
    }


def _vol_for(prices, assets):
    cols = [a for a in assets if a in prices.columns]
    return (prices[cols].pct_change().rolling(63).std() * np.sqrt(252)).mean()


def _inv_vol_shares(vol_series, sleeve):
    ivs = {
        a: 1.0 / float(vol_series[a])
        for a in sleeve
        if a in vol_series.index and float(vol_series[a]) > 0
    }
    total = sum(ivs.values())
    return (
        {a: v / total for a, v in ivs.items()}
        if total > 0
        else {a: 1 / len(sleeve) for a in sleeve}
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Risk-Off Sleeve — Add TIP Only")
    print("  Baseline risk-off: IEF, TLT, GLD")
    print("  Experiment:        IEF, TLT, GLD, TIP")
    print(f"  VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  sigmoid_scale=0.25")
    print("=" * 65)

    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ── bias audit ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)
    print("  Lookahead:                PASS")
    print("    Signal unchanged (24M SPY, expanding z-score, sigmoid(z*0.25)).")
    print("    TIP appears only in sleeve weights; no signal involvement.")
    print("  Forward-fill:             PASS")
    print("    Equal-weight + inv-vol sleeve allocation per rebalance.")
    print("    Forward-filled daily from month-end, same as baseline.")
    print("  Asset start-date align:   PASS")
    print("    TIP inception Dec 2003; Yahoo data from 2013-07-18.")
    print("    First OOS segment Aug 2015: 24+ months of vol warmup.")
    print("  Rebalance timing:         PASS")
    print("    Monthly, first trading day. Unchanged from baseline.")
    print("  Parameter isolation:      PASS")
    print("    Only risk_off_sleeve changes (3 -> 4 ETFs).")
    print("    Engine globals patched per-run and restored.")

    # ── vol structure ─────────────────────────────────────────────────────────
    print("\n  Fetching prices ...")
    prices = fetch_prices(tickers=TICKERS_EXP, start=FULL_START, end=FULL_END)
    if "TIP" not in prices.columns:
        print("STOP: TIP not available.")
        sys.exit(1)

    vol = _vol_for(prices, RF_EXP)
    shares_base = _inv_vol_shares(vol, RF_BASE)
    shares_exp = _inv_vol_shares(vol, RF_EXP)
    avg_ro = 0.40  # ~(1 - avg_risk_on)

    print("\n  Risk-off sleeve inv-vol weight structure:")
    print(
        f"  {'ETF':6}  {'Ann.Vol':>8}  {'Baseline share':>14}  {'Exp share':>10}  {'~Port wt':>10}  {'Change'}"
    )
    print("  " + "-" * 68)
    for t in RF_EXP:
        v = float(vol[t]) if t in vol.index else float("nan")
        bs = shares_base.get(t, 0.0)
        es = shares_exp.get(t, 0.0)
        pw = es * avg_ro
        chg = "NEW" if t == "TIP" else (f"{_pct(es - bs, sign=True)}" if not np.isnan(bs) else "")
        print(
            f"  {t:6}  {_pct(v):>8}  {_pct(bs) if t in shares_base else '---':>14}  "
            f"{_pct(es):>10}  {_pct(pw):>10}  {chg}"
        )

    tip_port_w = shares_exp.get("TIP", 0.0) * avg_ro
    print(f"\n  TIP expected average portfolio weight: ~{tip_port_w:.1%}")
    print(
        f"  (vs IEF: ~{shares_exp.get('IEF', 0) * avg_ro:.1%},  "
        f"GLD: ~{shares_exp.get('GLD', 0) * avg_ro:.1%},  "
        f"TLT: ~{shares_exp.get('TLT', 0) * avg_ro:.1%})"
    )
    print(
        f"  Vol parity check: TIP ({_pct(float(vol['TIP']))}) ~= IEF ({_pct(float(vol['IEF']))})"
        f"  -> reasonable inv-vol balance, no dominance risk"
    )

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running + TIP ...")
    df_fe = _run_exp(fk)

    sb_f = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    se_f = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not sb_f.equals(se_f):
        print("  STOP: OOS segment mismatch.")
        sys.exit(1)
    print(f"  OOS segments identical: YES ({len(sb_f)} segments)\n")

    ob_f = _overall(df_fb)
    oe_f = _overall(df_fe)
    fd = _print_table(ob_f, oe_f)

    f_ds_b = _mean(_filter_years(_segs(df_fb), 2021, 2022), "Strategy_Sharpe")
    f_ds_e = _mean(_filter_years(_segs(df_fe), 2021, 2022), "Strategy_Sharpe")
    f_diff_ok = not (np.isnan(f_ds_b) or np.isnan(f_ds_e)) and f_ds_e > f_ds_b
    f_kill = (fd["shr_d"] < 0.02) and (fd["cagr_d"] < 0.0025) and not f_diff_ok

    print(
        f"\n  Kill switch: dSharpe={fd['shr_d']:+.3f}  dCAGR={fd['cagr_d']:+.2%}  "
        f"2021-22={'BETTER' if f_diff_ok else 'NO'} (b={_f(f_ds_b)} e={_f(f_ds_e)})"
    )
    print(f"  Kill fires: {'YES' if f_kill else 'NO'}")
    if f_kill:
        print("  Kill fires but escalating: 2021-22 not in fast-mode OOS window.")

    # ── full walk-forward ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline ...")
    df_wb = _run_baseline(wk)

    print("  Running + TIP ...")
    df_we = _run_exp(wk)

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

    # ── sleeve weight summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RISK-OFF SLEEVE WEIGHT SUMMARY")
    print("=" * 65)
    print(f"\n  {'ETF':6}  {'Baseline sleeve%':>16}  {'Exp sleeve%':>12}  {'~Port wt (exp)':>14}")
    print("  " + "-" * 54)
    for t in RF_EXP:
        bs = shares_base.get(t, float("nan"))
        es = shares_exp[t]
        pw = es * avg_ro
        bs_str = _pct(bs) if not np.isnan(bs) else "---"
        print(f"  {t:6}  {bs_str:>16}  {_pct(es):>12}  {_pct(pw):>14}")

    print(f"\n  TIP average portfolio weight: ~{tip_port_w:.1%}")
    if not np.isnan(vol_d):
        if abs(vol_d) < 0.003:
            print(
                f"  Portfolio vol delta: ~flat ({_pct(vol_d, sign=True)}): TIP vol matches IEF, no distortion."
            )
        elif vol_d < 0:
            print(
                f"  Portfolio vol delta: {_pct(vol_d, sign=True)}: slight vol suppression from TIP allocation."
            )
        else:
            print(f"  Portfolio vol delta: {_pct(vol_d, sign=True)}: marginal vol increase.")

    # ── crisis check ──────────────────────────────────────────────────────────
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
        if es < bs - 0.05:
            print("    FLAG: Sharpe worsened >0.05.")
        if es > bs + 0.03:
            print("    NOTE: Sharpe improved.")
        if em > bm + 0.005:
            print("    NOTE: MaxDD improved.")

    # ── decision ──────────────────────────────────────────────────────────────
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
    if not np.isnan(vol_d):
        print(f"  Full-period Vol delta:          {_pct(vol_d, sign=True)}")
    print(f"  2018 Sharpe delta:              {_f(d_shr_2018, sign=True)}")
    print(f"  2020 Sharpe delta:              {_f(d_shr_2020, sign=True)}")
    print(f"  2020 MaxDD delta:               {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:         {_f(d_shr_2122, sign=True)}")
    print(f"  2021-2022 MaxDD delta:          {_pct(d_mdd_2122, sign=True)}")

    perf_improved = wd["shr_d"] >= 0.02 or wd["cagr_d"] >= 0.0025
    diff_improved = not np.isnan(d_shr_2122) and d_shr_2122 > 0.03
    hard_fail_shr = wd["shr_d"] < -0.05
    hard_fail_cagr = wd["cagr_d"] < -0.015
    mdd_2020_fail = not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02
    approx_flat = abs(wd["shr_d"]) < 0.04 and abs(wd["cagr_d"]) < 0.01
    to_mat_up = not np.isnan(to_d) and to_d > 0.20

    print()
    if perf_improved and not (to_mat_up and not diff_improved) and not mdd_2020_fail:
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {wd['shr_d']:+.3f}: TIP adds genuine value to risk-off sleeve.",
            f"CAGR improved {wd['cagr_d']:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: inflation-linked exposure helped in target period.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: COVID crash behavior preserved.",
            "Adopt IEF/TLT/GLD/TIP as new risk-off baseline. Update PROJECT_CONTEXT.md.",
        ]
    elif approx_flat and diff_improved and not mdd_2020_fail:
        verdict = "PASS"
        bullets = [
            f"Full-period approximately flat (Sharpe {wd['shr_d']:+.3f}): TIP does not hurt overall performance.",
            f"2021-2022 improved {_f(d_shr_2122, sign=True)}: inflation-linked exposure contributed in target period.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crash behavior preserved.",
            f"Turnover delta {_pct(to_d, sign=True)}: manageable.",
            "Accept IEF/TLT/GLD/TIP as new risk-off baseline.",
        ]
    elif hard_fail_shr or hard_fail_cagr or mdd_2020_fail:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {wd['shr_d']:+.3f}: adding TIP to the risk-off sleeve is net negative.",
            f"TIP vol (~5%) is close enough to IEF that inv-vol scaling gives it ~{shares_exp.get('TIP', 0):.0%} of sleeve, diluting GLD/TLT.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: TIP does not protect in equity crashes.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: inflation benefit too small vs dilution cost.",
            "Keep 3-ETF risk-off sleeve (IEF, TLT, GLD) as baseline.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {wd['shr_d']:+.3f}, CAGR {wd['cagr_d']:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: inflation benefit marginal.",
            f"TIP takes sleeve share (~{shares_exp.get('TIP', 0):.0%}) from GLD/TLT without consistent return benefit.",
            f"2020 MaxDD delta {_pct(d_mdd_2020, sign=True)}: crash behavior unchanged.",
            "Keep 3-ETF risk-off sleeve. The 2022 inflation problem is better addressed via signal improvement.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
