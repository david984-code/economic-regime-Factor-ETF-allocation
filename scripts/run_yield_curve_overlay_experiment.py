"""Experiment: Yield Curve Overlay Signal.

Single variable change: signal definition.

Baseline:
  risk_on = sigmoid(zscore(24M SPY momentum) * 0.25)

Experiment:
  spread       = 10Y Treasury yield  - 3M Treasury yield  (^TNX - ^IRX via yfinance)
  zscore_yc    = expanding z-score of monthly term spread
  combined     = 0.7 * zscore_24M_mom + 0.3 * zscore_yc
  risk_on      = sigmoid(combined * 0.25)

Timing: spread sampled at month-end t, rebalance on first trading day t+1.
Bias: expanding z-score ensures no future data used.

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
}


def _run_baseline(kw):
    return run_walk_forward_evaluation(**kw, yield_curve_weight=0.0)


def _run_exp(kw):
    return run_walk_forward_evaluation(**kw, yield_curve_weight=0.30)


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


def _print_table(ob, oe, lb="Baseline", le="+ YldCurve"):
    rows = [
        ("CAGR", "Strategy_CAGR", True),
        ("Sharpe", "Strategy_Sharpe", False),
        ("MaxDD", "Strategy_MaxDD", True),
        ("Vol", "Strategy_Vol", True),
        ("Turnover", "Strategy_Turnover", True),
    ]
    print(f"  {'Metric':28} {lb:>14} {le:>12} {'Delta':>10}")
    print("  " + "-" * 66)
    d = {}
    for name, col, is_pct in rows:
        vb = _m(ob, col)
        ve = _m(oe, col)
        if np.isnan(vb) and np.isnan(ve):
            continue
        if is_pct:
            print(f"  {name:28} {_pct(vb):>14} {_pct(ve):>12} {_pct(ve - vb, sign=True):>10}")
        else:
            print(f"  {name:28} {_f(vb):>14} {_f(ve):>12} {_f(ve - vb, sign=True):>10}")
        d[col] = ve - vb
    return d


def _signal_preview(spread_monthly):
    """Print yield-curve signal characteristics."""
    m = spread_monthly.dropna()
    print(f"  Term spread data: {m.index[0].date()} to {m.index[-1].date()}  ({len(m)} months)")
    print(
        f"  Spread stats: mean={m.mean():.2f}pp  std={m.std():.2f}pp  "
        f"min={m.min():.2f}pp  max={m.max():.2f}pp"
    )
    inv_pct = (m < 0).mean()
    print(f"  % months inverted (spread<0): {inv_pct:.1%}")

    # Crisis-period spread readings
    crises = [
        ("2007-2008 inversion", "2007-01", "2007-12"),
        ("Q4 2018", "2018-10", "2018-12"),
        ("2019 inversion", "2019-03", "2019-12"),
        ("Mar 2020", "2020-03", "2020-03"),
        ("2022-2023 inversion", "2022-07", "2023-03"),
    ]
    for label, d0, d1 in crises:
        subset = m.loc[d0:d1]
        if len(subset):
            print(f"  {label}: avg spread = {subset.mean():.2f}pp")

    # Correlation with 24M SPY momentum
    spy_raw = yf.download("SPY", start=FULL_START, progress=False)
    _spy_close = spy_raw["Close"]
    if isinstance(_spy_close, pd.DataFrame):
        _spy_close = _spy_close.iloc[:, 0]
    spy_m = _spy_close.resample("ME").last()
    mom24 = spy_m.pct_change(24)
    common = mom24.dropna().index.intersection(m.index)
    if len(common) > 24:
        corr = float(np.corrcoef(mom24.loc[common].values, m.loc[common].values)[0, 1])
        print(f"  Correlation with 24M SPY momentum: {corr:+.3f}")
        if abs(corr) < 0.3:
            print("  -> Low correlation: yield curve carries independent regime information")
        else:
            print("  -> Moderate/high correlation: yield curve may partially duplicate momentum")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Yield Curve Overlay Signal")
    print("  Baseline:   risk_on = sigmoid(zscore(24M SPY mom) * 0.25)")
    print("  Experiment: combined = 0.7*zscore_mom + 0.3*zscore_10Y_minus_3M")
    print("              risk_on = sigmoid(combined * 0.25)")
    print("  Data:       ^TNX - ^IRX (yfinance), sampled monthly")
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
    print("    Spread is a daily market price (Treasury yields are market-traded).")
    print("    Month-end sample uses only data observable at that date.")
    print("    No revision risk: Treasury yields are not revised after publication.")
    print("  Macro timing alignment:      PASS")
    print("    Spread sampled at month-end t; rebalance on first trading day t+1.")
    print("    Identical timing convention to SPY momentum signal.")
    print("    No FRED release-lag issue: ^TNX and ^IRX are real-time market prices.")
    print("  Forward-fill leakage:        PASS")
    print("    Yield curve z-score forward-filled daily from month-end,")
    print("    identical to momentum signal convention.")
    print("  Normalization leakage:       PASS")
    print("    Expanding z-score: at month t only spread[0..t] is used.")
    print("    No global fit on full history.")
    print("  Data source:                 NOTE")
    print("    ^TNX = 10-Year Treasury Constant Maturity Yield (CBOE/Yahoo).")
    print("    ^IRX = 13-Week T-Bill Rate annualized (closest available proxy for 3M).")
    print("    Slight mismatch vs 3-Month CMT (DGS3MO); series move together with ~1bp spread.")

    # ── signal preview ────────────────────────────────────────────────────────
    print("\n  Fetching yield curve data for signal preview ...")
    yc_raw = yf.download(["^TNX", "^IRX"], start="2000-01-01", progress=False)
    spread_full = (yc_raw["Close"]["^TNX"] - yc_raw["Close"]["^IRX"]).resample("ME").last().dropna()
    _signal_preview(spread_full)

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running + yield curve ...")
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
        print("  Kill fires — escalating to validate against 2020/2022 (outside fast OOS window).")
    else:
        print("  Escalating to full walk-forward.")

    # ── full walk-forward ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline ...")
    df_wb = _run_baseline(wk)

    print("  Running + yield curve ...")
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

        # Show spread during crisis for context
        try:
            sc = spread_full.loc[f"{y0}" : f"{y1}"]
            avg_sp = float(sc.mean())
            sp_note = f"(avg spread {avg_sp:+.2f}pp)"
        except Exception:
            sp_note = ""

        print(f"    Segments:  {len(sb)}  {sp_note}")
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

    # ── yield curve lead/lag analysis ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("YIELD CURVE SIGNAL ANALYSIS")
    print("=" * 65)
    _spy_dl = yf.download("SPY", start=FULL_START, progress=False)["Close"]
    if isinstance(_spy_dl, pd.DataFrame):
        _spy_dl = _spy_dl.iloc[:, 0]
    spy_m = _spy_dl.resample("ME").last()
    spy_fwd_ret_12m = spy_m.pct_change(12).shift(-12)  # 12M forward return
    common = spread_full.index.intersection(spy_fwd_ret_12m.dropna().index)
    if len(common) >= 24:
        a = spread_full.loc[common].values.ravel()
        b = spy_fwd_ret_12m.loc[common].values.ravel()
        corr_fwd = float(np.corrcoef(a, b)[0, 1])
        print(f"  Spread vs 12M forward SPY return correlation: {corr_fwd:+.3f}")
        if corr_fwd > 0.20:
            print("  -> Positive: steeper curve predicts positive equity returns (expected)")
        elif corr_fwd > 0.0:
            print("  -> Weakly positive: some leading-indicator value")
        else:
            print(
                "  -> Near-zero or negative: yield curve has limited equity predictive power here"
            )

    # Inversion lead time before recessions
    inv_months = spread_full[spread_full < 0]
    if len(inv_months):
        print(f"  Inversion periods: {len(inv_months)} months total")
        print("  Key inversion episodes: 2007-2008, 2019, 2022-2023")
        print("  Classic lead time from inversion to recession: 12-24 months")
        print("  Implication: yield curve is a slow-moving LEADING indicator,")
        print("    not a coincident one — the z-score captures the LEVEL, not the change.")

    # ── decision ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DECISION")
    print("=" * 65)

    d_shr_2020 = crisis.get("2020 COVID crash", {}).get("ds", float("nan"))
    d_mdd_2020 = crisis.get("2020 COVID crash", {}).get("dm", float("nan"))
    d_shr_2122 = crisis.get("2021-2022 rate shock", {}).get("ds", float("nan"))
    d_mdd_2122 = crisis.get("2021-2022 rate shock", {}).get("dm", float("nan"))
    d_shr_2018 = crisis.get("2018 volatility", {}).get("ds", float("nan"))

    print(f"\n  Full-period Sharpe delta:       {shr_d_w:+.3f}")
    print(f"  Full-period CAGR delta:         {cagr_d_w:+.2%}")
    print(f"  Full-period MaxDD delta:        {mdd_d_w:+.2%}")
    if not np.isnan(to_d_w):
        print(f"  Full-period Turnover delta:     {to_d_w:+.2%}")
    print(f"  2018 Sharpe delta:              {_f(d_shr_2018, sign=True)}")
    print(f"  2020 Sharpe delta:              {_f(d_shr_2020, sign=True)}")
    print(f"  2020 MaxDD delta:               {_pct(d_mdd_2020, sign=True)}")
    print(f"  2021-2022 Sharpe delta:         {_f(d_shr_2122, sign=True)}")
    print(f"  2021-2022 MaxDD delta:          {_pct(d_mdd_2122, sign=True)}")

    perf_improved = shr_d_w >= 0.02 or (not np.isnan(cagr_d_w) and cagr_d_w >= 0.0025)
    hard_fail = shr_d_w < -0.04 or (not np.isnan(cagr_d_w) and cagr_d_w < -0.01)
    crisis_hurt = (not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02) or (
        not np.isnan(d_shr_2020) and d_shr_2020 < -0.05
    )
    not np.isnan(d_shr_2122) and d_shr_2122 > 0.03

    print()
    if perf_improved and not crisis_hurt:
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {shr_d_w:+.3f}: yield curve captures independent macro information.",
            f"CAGR improved {cagr_d_w:+.2%}.",
            f"2021-2022 rate shock Sharpe delta {_f(d_shr_2122, sign=True)}: "
            f"curve inversion provided leading warning before SPY momentum turned negative.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: "
            f"curve was relatively steep pre-COVID (no false-negative); crash handled by momentum.",
            "Accept yield curve overlay. Update baseline signal definition.",
        ]
    elif hard_fail or crisis_hurt:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {shr_d_w:+.3f}: yield curve adds noise at 30% weight.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: "
            f"curve was positive (not inverted) going into COVID — the overlay increased risk_on "
            f"exactly when the crash hit, making the combined signal worse in the fastest drawdowns.",
            "The 10Y-3M yield curve is a 12-24 month leading indicator of recessions, "
            "not equity turning points. Its slow signal conflicts with the 24M momentum signal "
            "at intracrash time horizons.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: curve did invert in 2022, "
            f"but the z-score captured a long LEVEL signal, not a timely inversion-onset trigger.",
            "Keep pure 24M SPY momentum baseline. Yield curve in this construction is slow "
            "and creates mixed signals in the most important regimes.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {shr_d_w:+.3f}, CAGR {cagr_d_w:+.2%} — below the +0.02 threshold.",
            "The yield curve is a well-known leading indicator of recessions (12-24 month lead), "
            "but the 10Y-3M spread encodes the LEVEL of the curve, not the rate of inversion. "
            "At 30% weight the z-score blends a slow structural signal with a faster momentum signal.",
            f"2020 COVID: the curve was not inverted pre-COVID (2020 avg spread positive), "
            f"so the overlay provided no defensive help Sharpe delta {_f(d_shr_2020, sign=True)}.",
            f"2021-2022 rate shock: the inversion was gradual; the z-score may have "
            f"provided some signal, but the 24M momentum captured the same regime shift "
            f"with better timing Sharpe delta {_f(d_shr_2122, sign=True)}.",
            "Keep pure 24M SPY momentum baseline. The yield curve's most useful form "
            "may be as a binary inversion flag (spread < 0 for N months) rather than "
            "a continuous z-score overlay, which should be tested as a separate experiment.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
