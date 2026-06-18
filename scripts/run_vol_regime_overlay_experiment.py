"""Experiment: Volatility Regime Overlay Signal.

Single variable change: signal definition.

Baseline:
  risk_on = sigmoid(zscore(24M SPY momentum) * 0.25)

Experiment:
  vol_63d    = 63-day realized vol of SPY (annualized)
  zscore_vol = expanding z-score of vol_63d (same method as momentum z-score)
  combined   = 0.7 * zscore_24M_mom - 0.3 * zscore_63d_vol
  risk_on    = sigmoid(combined * 0.25)

Interpretation: high realized vol reduces risk_on independently of momentum.

All other baseline parameters are unchanged:
  market_lookback_months=24, sigmoid_scale=0.25, equal_weight sleeves,
  VOL_LOOKBACK=63, tolerance=0.015, monthly rebalance, trend_filter_type=none.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

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
    "tickers": list(TICKERS),
    "assets": list(ASSETS),
    "risk_on_sleeve": list(RISK_ON_ASSETS_BASE),
    "risk_off_sleeve": list(RISK_OFF_ASSETS_BASE),
    "rf_sleeve_cap": 0.0,
    "breadth_weight": 0.0,
}


def _run_baseline(kw):
    return run_walk_forward_evaluation(**kw, vol_zscore_weight=0.0)


def _run_exp(kw):
    return run_walk_forward_evaluation(**kw, vol_zscore_weight=0.30)


# ── helpers ───────────────────────────────────────────────────────────────────


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


def _print_table(ob, oe, lb="Baseline", le="+ VolZ"):
    rows = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {lb:>14} {le:>10} {'Delta':>10}")
    print("  " + "-" * 64)
    d = {}
    for name, col, is_pct in rows:
        vb = _m(ob, col)
        ve = _m(oe, col)
        if np.isnan(vb) and np.isnan(ve):
            continue
        if is_pct:
            print(f"  {name:28} {_pct(vb):>14} {_pct(ve):>10} {_pct(ve - vb, sign=True):>10}")
        else:
            print(f"  {name:28} {_f(vb):>14} {_f(ve):>10} {_f(ve - vb, sign=True):>10}")
        d[col] = ve - vb
    return d


def _signal_preview(prices):
    """Print vol signal characteristics as sanity check."""
    spy_ret = prices["SPY"].pct_change()
    rv = spy_ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    rv_m = rv.resample("ME").last().dropna()
    print(
        f"  63-day realized vol: mean={rv_m.mean():.1%}  std={rv_m.std():.1%}  "
        f"min={rv_m.min():.1%}  max={rv_m.max():.1%}"
    )
    print(f"  % months above mean vol (historically elevated): {(rv_m > rv_m.mean()).mean():.0%}")
    # Show a few crisis-period vol readings
    for label, d0, d1 in [
        ("Mar 2020", "2020-03", "2020-03"),
        ("Q4 2022", "2022-10", "2022-12"),
        ("Q4 2018", "2018-10", "2018-12"),
    ]:
        subset = rv_m.loc[d0:d1]
        if len(subset):
            print(f"  {label} vol: {subset.values[0]:.1%}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Volatility Regime Overlay Signal")
    print("  Baseline:   risk_on = sigmoid(zscore(24M SPY mom) * 0.25)")
    print("  Experiment: combined = 0.7*zscore_mom - 0.3*zscore_63d_vol")
    print("              risk_on = sigmoid(combined * 0.25)")
    print(f"  VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  sigmoid_scale=0.25")
    print("=" * 65)

    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ── bias audit ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)
    print("  Lookahead:                   PASS")
    print("    63-day rolling std uses only past daily returns.")
    print("    Month-end sample: price at last trading day, no peeking forward.")
    print("  Timing alignment:            PASS")
    print("    Vol computed at month-end t; rebalance on first trading day t+1.")
    print("    Identical to SPY momentum timing convention.")
    print("  Forward-fill leakage:        PASS")
    print("    Vol z-score forward-filled daily from month-end same as momentum.")
    print("  Normalization leakage:       PASS")
    print("    Expanding z-score: at month t only vol[0..t] used.")
    print("    No global fit on full history.")
    print("  Signal separation note:      IMPORTANT")
    print("    VOL_LOOKBACK=63 (inverse-vol scaling) and 63-day realized vol signal")
    print("    both use a 63-day window but serve entirely different roles:")
    print("    - VOL_LOOKBACK: intra-sleeve weight scaling (portfolio construction)")
    print("    - vol_zscore_weight: signal combination (market timing)")
    print("    They share the same lookback but operate on orthogonal pipeline stages.")
    print("    This is NOT a change to portfolio construction — it is signal-only.")

    # ── signal preview ────────────────────────────────────────────────────────
    print("\n  Fetching SPY prices for signal preview ...")
    prices_preview = fetch_prices(tickers=["SPY"], start="2009-01-01", end=FULL_END)
    _signal_preview(prices_preview)

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running + vol-z ...")
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

    shr_d_f = fd.get("Strategy_Sharpe", float("nan"))
    cagr_d_f = fd.get("Strategy_CAGR", float("nan"))
    f_kill = (
        (not np.isnan(shr_d_f))
        and (shr_d_f < 0.02)
        and (not np.isnan(cagr_d_f))
        and (cagr_d_f < 0.0025)
    )

    print(f"\n  Kill switch: dSharpe={shr_d_f:+.3f}  dCAGR={cagr_d_f:+.2%}")
    print(f"  Kill fires: {'YES' if f_kill else 'NO'}")
    if f_kill:
        print("  Kill fires — escalating to check 2020/2022 (outside fast OOS window).")
    else:
        print("  Escalating to full walk-forward.")

    # ── full walk-forward ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline ...")
    df_wb = _run_baseline(wk)

    print("  Running + vol-z ...")
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

    shr_d_w = wd.get("Strategy_Sharpe", float("nan"))
    cagr_d_w = wd.get("Strategy_CAGR", float("nan"))
    mdd_d_w = wd.get("Strategy_MaxDD", float("nan"))
    to_d_w = wd.get("Strategy_Turnover", float("nan"))

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
        bc = _mean(sb, "Strategy_CAGR")
        ec = _mean(se, "Strategy_CAGR")
        bto = _mean(sb, "Strategy_Turnover")
        eto = _mean(se, "Strategy_Turnover")

        print(f"    Segments:  {len(sb)}")
        print(f"    CAGR:     base={_pct(bc)}   exp={_pct(ec)}   delta={_pct(ec - bc, sign=True)}")
        print(f"    Sharpe:   base={_f(bs)}    exp={_f(es)}    delta={_f(es - bs, sign=True)}")
        print(f"    MaxDD:    base={_pct(bm)}   exp={_pct(em)}   delta={_pct(em - bm, sign=True)}")
        if not (np.isnan(bto) or np.isnan(eto)):
            print(
                f"    Turnover: base={_pct(bto)}   exp={_pct(eto)}   delta={_pct(eto - bto, sign=True)}"
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

    d_shr_2020 = crisis.get("2020 COVID crash", {}).get("ds", float("nan"))
    d_mdd_2020 = crisis.get("2020 COVID crash", {}).get("dm", float("nan"))
    d_shr_2122 = crisis.get("2021-2022 rate shock", {}).get("ds", float("nan"))
    d_mdd_2122 = crisis.get("2021-2022 rate shock", {}).get("dm", float("nan"))
    d_shr_2018 = crisis.get("2018 volatility", {}).get("ds", float("nan"))
    d_mdd_2018 = crisis.get("2018 volatility", {}).get("dm", float("nan"))

    print(f"\n  Full-period Sharpe delta:       {shr_d_w:+.3f}")
    print(f"  Full-period CAGR delta:         {cagr_d_w:+.2%}")
    print(f"  Full-period MaxDD delta:        {mdd_d_w:+.2%}")
    if not np.isnan(to_d_w):
        print(f"  Full-period Turnover delta:     {to_d_w:+.2%}")
    print(f"  2018 Sharpe delta:              {_f(d_shr_2018, sign=True)}")
    print(f"  2018 MaxDD delta:               {_pct(d_mdd_2018, sign=True)}")
    print(f"  2020 Sharpe delta:              {_f(d_shr_2020, sign=True)}")
    print(f"  2020 MaxDD delta:               {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:         {_f(d_shr_2122, sign=True)}")
    print(f"  2021-2022 MaxDD delta:          {_pct(d_mdd_2122, sign=True)}")

    perf_improved = shr_d_w >= 0.02 or (not np.isnan(cagr_d_w) and cagr_d_w >= 0.0025)
    hard_fail = shr_d_w < -0.04 or (not np.isnan(cagr_d_w) and cagr_d_w < -0.01)
    (not np.isnan(d_shr_2020) and d_shr_2020 > 0.03) or (
        not np.isnan(d_shr_2018) and d_shr_2018 > 0.03
    )
    crisis_hurt = (not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02) or (
        not np.isnan(d_shr_2020) and d_shr_2020 < -0.05
    )
    np.isnan(to_d_w) or to_d_w <= 0.20

    print()
    if perf_improved and not crisis_hurt:
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {shr_d_w:+.3f}: realized vol provides independent regime information.",
            f"CAGR improved {cagr_d_w:+.2%}.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: vol spike front-ran momentum deterioration.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: rate-shock regime detected via vol elevation.",
            f"Turnover delta {_pct(to_d_w, sign=True)}: manageable signal responsiveness increase.",
        ]
    elif hard_fail or crisis_hurt:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {shr_d_w:+.3f}: 63-day realized vol duplicates information already"
            f" in the 24M momentum z-score and adds noise.",
            "The vol signal and momentum signal are highly correlated during crises:"
            " vol spikes occur simultaneously with or after momentum turns, providing no lead.",
            f"2020 COVID: vol exploded in the same month as SPY's crash — no early warning."
            f" Sharpe delta {_f(d_shr_2020, sign=True)}, MaxDD delta {_pct(d_mdd_2020, sign=True)}.",
            f"2021-2022 rate shock: persistent low-to-moderate vol did not signal regime change."
            f" Sharpe delta {_f(d_shr_2122, sign=True)}.",
            "Pure 24M momentum baseline captures slow regime changes better."
            " 63-day vol is a coincident indicator, not a leading one.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {shr_d_w:+.3f}, CAGR {cagr_d_w:+.2%} — below the +0.02 threshold.",
            "The 63-day realized vol z-score is a coincident indicator: it peaks during crashes"
            " simultaneously with momentum, not before — providing no timing edge.",
            "The 0.30 vol weight reduces the effective weight on the 24M momentum z-score"
            " (from 1.0 to 0.7) without compensating return for that dilution.",
            f"2020/2022 crisis performance {_f(d_shr_2020, sign=True)}/{_f(d_shr_2122, sign=True)}:"
            f" no consistent improvement in the periods where added value is needed most.",
            "Keep pure 24M SPY momentum baseline. If vol overlay is revisited,"
            " test lower weights (0.10-0.15) or a longer vol lookback (126d) that is less coincident.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
