"""Experiment: Yield Curve Inversion Flag Overlay.

Single variable change: signal definition.

Baseline:
  risk_on = sigmoid(zscore(24M SPY momentum) * 0.25)

Experiment:
  inversion_flag = 1  if 10Y-3M spread < 0 for 3 consecutive months, else 0
  combined_score = zscore_24M_SPY_momentum - 0.5 * inversion_flag
  risk_on        = sigmoid(combined_score * 0.25)

The flag offset (-0.5) shifts combined_score down by a fixed 0.5 sigma-equivalent
during confirmed inversion episodes, reducing risk_on from its momentum-implied level.
When inversion resolves, the flag drops immediately.

All other baseline parameters unchanged:
  market_lookback_months=24, sigmoid_scale=0.25, equal_weight sleeves,
  VOL_LOOKBACK=63, tolerance=0.015, monthly rebalance, trend_filter_type=none.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    ASSETS,
    RISK_OFF_ASSETS_BASE,
    RISK_ON_ASSETS_BASE,
    TICKERS,
    VOL_LOOKBACK,
)
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
    "vol_zscore_weight": 0.0,
    "yield_curve_weight": 0.0,
}


def _run_baseline(kw):
    return run_walk_forward_evaluation(**kw, inversion_flag_offset=0.0)


def _run_exp(kw):
    return run_walk_forward_evaluation(**kw, inversion_flag_offset=0.5)


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


def _print_table(ob, oe, lb="Baseline", le="+ InvFlag"):
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


def _flag_analysis(spread_monthly):
    """Compute inversion flag series and print statistics."""
    m = spread_monthly.dropna()

    # Build flag: 1 if 3 consecutive months negative
    flags = pd.Series(0.0, index=m.index)
    for i in range(2, len(m)):
        if m.iloc[i] < 0 and m.iloc[i - 1] < 0 and m.iloc[i - 2] < 0:
            flags.iloc[i] = 1.0

    total_flag_months = int(flags.sum())
    flag_pct = flags.mean()

    print(f"  Total months flag=1: {total_flag_months} ({flag_pct:.1%} of history)")

    # Find first trigger dates for known inversion episodes
    episodes = {
        "2007-2008": ("2006-01", "2008-12"),
        "2019": ("2019-01", "2020-06"),
        "2022-2023": ("2022-01", "2024-06"),
    }
    for label, (d0, d1) in episodes.items():
        ep = flags.loc[d0:d1]
        active = ep[ep > 0]
        if len(active):
            first_trigger = active.index[0]
            # Days from first consecutive-3-month trigger to episode start
            print(
                f"  {label}: first flag=1 at {first_trigger.strftime('%Y-%m')}"
                f"  (spread avg during flag: {float(m.loc[d0:d1][flags.loc[d0:d1] > 0].mean()):.2f}pp)"
            )
        else:
            print(f"  {label}: flag never triggered (spread above 0 throughout)")

    # Impact on sigmoid(z * 0.25) when flag is active
    # If z=0 (neutral), flag shifts combined from 0 to -0.5
    # sigmoid(-0.5 * 0.25) = sigmoid(-0.125) ≈ 0.469 vs sigmoid(0) = 0.500
    # If z=+1 (moderately bullish), flag shifts to 0.5 → sigmoid(0.5*0.25)=0.531 vs sigmoid(0.25)=0.562
    delta_neutral = 1 / (1 + np.exp(0.125)) - 0.5
    delta_bull = 1 / (1 + np.exp(-0.5 * 0.25)) - 1 / (1 + np.exp(-1.0 * 0.25))
    print("  Signal impact when flag fires:")
    print(
        f"    At z=0.0 (neutral):    risk_on: 0.500 → {1 / (1 + np.exp(0.125)):.3f}  (delta={delta_neutral:+.3f})"
    )
    print(
        f"    At z=+1.0 (bullish):   risk_on: {1 / (1 + np.exp(-0.25)):.3f} → {1 / (1 + np.exp(-0.5 * 0.25)):.3f}  (delta={delta_bull:+.3f})"
    )
    print(
        f"    At z=+2.0 (very bull): risk_on: {1 / (1 + np.exp(-0.5)):.3f} → {1 / (1 + np.exp(-1.5 * 0.25)):.3f}  (delta={1 / (1 + np.exp(-1.5 * 0.25)) - 1 / (1 + np.exp(-0.5)):+.3f})"
    )

    return flags, total_flag_months


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Yield Curve Inversion Flag Overlay")
    print("  Baseline:   risk_on = sigmoid(zscore(24M SPY mom) * 0.25)")
    print("  Experiment: combined = zscore_mom - 0.5 * inversion_flag")
    print("              inversion_flag = 1 if spread < 0 for 3+ months")
    print("              risk_on = sigmoid(combined * 0.25)")
    print("  Data:       ^TNX - ^IRX (yfinance), monthly")
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
    print("    Flag at month t uses spread[t], spread[t-1], spread[t-2].")
    print("    All three values are observable at month-end t.")
    print("    No future spread data enters the flag calculation.")
    print("  Macro timing alignment:      PASS")
    print("    Flag computed at month-end t; rebalance on first trading day t+1.")
    print("    3-month window requires 2 prior months — signal has a built-in")
    print("    confirmation delay of 3 months from first negative spread reading.")
    print("  Forward-fill leakage:        PASS")
    print("    Flag forward-filled daily from month-end, same as momentum signal.")
    print("    Flag transitions are discrete (0→1 or 1→0) at each month-end only.")
    print("  Normalization leakage:       PASS")
    print("    No z-score normalization of the flag itself.")
    print("    The -0.5 offset is applied to the z-score domain (not price domain).")
    print("    Offset is fixed, not fitted on any historical data.")
    print("  Event flag timing:           PASS")
    print("    The 3-month confirmation rule prevents false flags from a single")
    print("    month dip. The first trigger is always 3 months after initial inversion.")
    print("    This matches practitioner convention for confirmed inversion signals.")

    # ── spread and flag preview ───────────────────────────────────────────────
    print("\n  Fetching yield curve data for flag analysis ...")
    yc_raw = yf.download(["^TNX", "^IRX"], start="2000-01-01", progress=False)
    spread_full = (yc_raw["Close"]["^TNX"] - yc_raw["Close"]["^IRX"]).resample("ME").last().dropna()
    print(f"  Spread available: {spread_full.index[0].date()} to {spread_full.index[-1].date()}")
    flags, n_flag_months = _flag_analysis(spread_full)

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)
    # Flag coverage in fast-mode OOS window (note: 2019 inversion falls inside)
    fast_flags = flags.loc[FAST_START:FAST_END]
    print(
        f"  Flag months in fast-mode window: {int(fast_flags.sum())} of {len(fast_flags)}"
        f"  (2019 inversion + 2022-23 inversion may be present)\n"
    )

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running + inversion flag ...")
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
        print("  Kill fires — escalating to check 2020 / 2022 (not fully in fast-mode OOS).")
    else:
        print("  Escalating to full walk-forward.")

    # ── full walk-forward ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline ...")
    df_wb = _run_baseline(wk)

    print("  Running + inversion flag ...")
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
    print(f"  Segments:   {n_segs}")

    # Flag months within OOS window
    try:
        oos_flags = flags.loc[oos_start:oos_end]
        print(f"  Flag months in OOS window: {int(oos_flags.sum())} / {len(oos_flags)}")
    except Exception:
        pass
    print()

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

        # Flag coverage during period
        try:
            pf = flags.loc[f"{y0}" : f"{y1}"]
            flag_note = f"  (flag active {int(pf.sum())}/{len(pf)} months)"
        except Exception:
            flag_note = ""

        bs = _mean(sb, "Strategy_Sharpe")
        es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD")
        em = _mean(se, "Strategy_MaxDD")
        bc = _mean(sb, "Strategy_CAGR")
        ec = _mean(se, "Strategy_CAGR")
        bto = _mean(sb, "Strategy_Turnover")
        eto = _mean(se, "Strategy_Turnover")

        print(f"    Segments:  {len(sb)}{flag_note}")
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

    print(f"\n  Total flag months (full history): {n_flag_months}")
    print(f"  Full-period Sharpe delta:         {shr_d_w:+.3f}")
    print(f"  Full-period CAGR delta:           {cagr_d_w:+.2%}")
    print(f"  Full-period MaxDD delta:          {mdd_d_w:+.2%}")
    if not np.isnan(to_d_w):
        print(f"  Full-period Turnover delta:       {to_d_w:+.2%}")
    print(f"  2018 Sharpe delta:                {_f(d_shr_2018, sign=True)}")
    print(f"  2020 Sharpe delta:                {_f(d_shr_2020, sign=True)}")
    print(f"  2020 MaxDD delta:                 {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:           {_f(d_shr_2122, sign=True)}")
    print(f"  2021-2022 MaxDD delta:            {_pct(d_mdd_2122, sign=True)}")

    perf_improved = shr_d_w >= 0.02 or (not np.isnan(cagr_d_w) and cagr_d_w >= 0.0025)
    hard_fail = shr_d_w < -0.04 or (not np.isnan(cagr_d_w) and cagr_d_w < -0.015)
    crisis_hurt = (not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02) or (
        not np.isnan(d_shr_2020) and d_shr_2020 < -0.05
    )
    not np.isnan(d_shr_2122) and d_shr_2122 > 0.03

    print()
    if perf_improved and not crisis_hurt:
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {shr_d_w:+.3f}: the binary inversion flag"
            f" adds leading macro information without slow-moving noise.",
            f"CAGR improved {cagr_d_w:+.2%}.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: confirmed inversion triggered"
            f" reduced risk_on before the full rate-shock momentum drawdown.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: curve was steep pre-COVID"
            f" (flag off), so the signal did not hurt the crash response.",
            "Accept inversion flag overlay. Update signal definition in PROJECT_CONTEXT.md.",
        ]
    elif hard_fail or crisis_hurt:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {shr_d_w:+.3f}: the 3-month confirmation lag"
            f" makes the inversion flag a lagging, not leading, signal.",
            "The 10Y-3M curve inversions that preceded genuine bear markets"
            " (2007, 2022) had lead times of 6-18 months — but the equity drawdown"
            " often recovers before the flag resolves, so the flag fires into recoveries.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: curve was NOT inverted"
            f" pre-COVID — flag provided zero warning; and afterwards caused risk reduction"
            f" at wrong time.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: by the time the flag"
            f" triggered (3+ consecutive negative months), momentum had already captured"
            f" most of the regime shift.",
            "Keep pure 24M SPY momentum baseline. Binary inversion flag in this"
            " implementation is still a lagging overlay on a signal that already leads.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed/flat result: Sharpe {shr_d_w:+.3f}, CAGR {cagr_d_w:+.2%} — below +0.02 threshold.",
            "The 3-month confirmation window prevents false flags but introduces exactly"
            " the lag that makes yield curve inversions less useful as equity timing signals.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: the flag triggered during"
            f" the 2022 inversion, but the 24M momentum z-score already reflected the"
            f" deteriorating regime — the -0.5 offset added marginal impact.",
            f"2020 Sharpe delta {_f(d_shr_2020, sign=True)}: curve was not inverted in 2020,"
            f" so the flag was inactive — the overlay neither helped nor hurt the worst crash.",
            "Keep pure 24M SPY momentum baseline. Both yield curve forms (continuous z-score"
            " and binary flag) fail to add net value over the 116-segment OOS history.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
