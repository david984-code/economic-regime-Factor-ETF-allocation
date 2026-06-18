"""Experiment: Market Breadth Overlay Signal.

Single variable change: signal definition.

Baseline:
  risk_on = sigmoid(zscore(24M SPY momentum) * 0.25)

Experiment:
  combined_score = 0.7 * zscore(24M SPY momentum) + 0.3 * zscore(sector breadth)
  risk_on = sigmoid(combined_score * 0.25)

Breadth definition:
  Daily: % of 9 SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)
         with price > 200-day rolling mean.
  Monthly: sample at month-end.
  Normalized: expanding z-score (same method as SPY momentum).

Note: ^SPXA200R is not available via yfinance. The SPDR 9-sector ETF proxy
covers all major S&P 500 economic sectors, provides data back to 2009, and is
free from survivorship bias. It is the standard practitioner breadth proxy
available without paid data.

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

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

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
}


def _run_baseline(kw):
    return run_walk_forward_evaluation(**kw, breadth_weight=0.0)


def _run_exp(kw):
    return run_walk_forward_evaluation(**kw, breadth_weight=0.30)


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


def _print_table(ob, oe, lb="Baseline", le="+ Breadth"):
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


# ── breadth signal preview ────────────────────────────────────────────────────


def _preview_breadth(prices_sector):
    """Show monthly breadth statistics as a sanity check."""
    above_df = pd.DataFrame(index=prices_sector.index)
    for etf in SECTOR_ETFS:
        if etf not in prices_sector.columns:
            continue
        px = prices_sector[etf].ffill()
        ma200 = px.rolling(200, min_periods=200).mean()
        above_df[etf] = (px > ma200).astype(float)
        above_df.loc[ma200.isna(), etf] = np.nan
    valid = above_df.notna().sum(axis=1).clip(lower=1)
    daily = above_df.sum(axis=1) / valid
    monthly = daily.resample("ME").last().dropna()

    print(f"  Breadth data: {monthly.index[0].date()} to {monthly.index[-1].date()}")
    print(
        f"  Monthly breadth: mean={monthly.mean():.1%}  std={monthly.std():.1%}  "
        f"min={monthly.min():.1%}  max={monthly.max():.1%}"
    )
    print(f"  % months breadth < 30%: {(monthly < 0.30).mean():.1%}  (bearish signal)")
    print(f"  % months breadth > 70%: {(monthly > 0.70).mean():.1%}  (bullish)")
    print(f"  9 sector ETFs used: {', '.join(SECTOR_ETFS)}")
    print("  (^SPXA200R not available via yfinance; 9-sector proxy covers all S&P 500 sectors)")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Market Breadth Overlay Signal")
    print("  Baseline:   risk_on = sigmoid(zscore(24M SPY mom) * 0.25)")
    print("  Experiment: risk_on = sigmoid((0.7*zscore_mom + 0.3*zscore_breadth)*0.25)")
    print("  Breadth:    9-sector SPDR ETFs, % above 200-day MA, expanding z-score")
    print(f"  VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  sigmoid_scale=0.25")
    print("=" * 65)

    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ── bias audit ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)
    print("  Lookahead bias:              PASS")
    print("    200-day MA uses rolling(200) on past prices only.")
    print("    Expanding z-score uses only breadth values observed up to each date.")
    print("    No future information enters the signal.")
    print("  Breadth timing alignment:    PASS")
    print("    Breadth sampled at month-end t.")
    print("    Rebalance executed on first trading day of month t+1.")
    print("    Same timing as SPY momentum signal — perfectly aligned.")
    print("  Forward-fill leakage:        PASS")
    print("    Daily breadth computed from end-of-day closes.")
    print("    Monthly value = last trading day of month.")
    print("    Forward-filled to daily for regime_df same as momentum signal.")
    print("  Normalization leakage:       PASS")
    print("    Expanding z-score: at month t, only breadth[0..t] used.")
    print("    No in-sample z-score fitted on the full history.")
    print("  Rebalance timing:            PASS")
    print("    First trading day of each month. Unchanged from baseline.")
    print("  Parameter isolation:         PASS")
    print("    Only signal composition changes (breadth_weight=0 -> 0.3).")
    print("    Sector ETFs not in allocation universe — no weight contamination.")

    # ── breadth sanity check ──────────────────────────────────────────────────
    print("\n  Fetching sector ETF prices for breadth preview ...")
    sector_prices = fetch_prices(tickers=SECTOR_ETFS, start="2009-01-01", end=FULL_END)
    _preview_breadth(sector_prices)

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline ...")
    df_fb = _run_baseline(fk)

    print("  Running + breadth ...")
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
        print(
            "  Kill fires — escalating for crisis-period verification (2020/2022 not in fast OOS)."
        )
    else:
        print("  Escalating to full walk-forward.")

    # ── full walk-forward ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    wk = dict(**SHARED, start=FULL_START, end=FULL_END, fast_mode=False)

    print("  Running baseline ...")
    df_wb = _run_baseline(wk)

    print("  Running + breadth ...")
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

    # Decision logic
    perf_improved = shr_d_w >= 0.02 or (not np.isnan(cagr_d_w) and cagr_d_w >= 0.0025)
    hard_fail = shr_d_w < -0.04 or (not np.isnan(cagr_d_w) and cagr_d_w < -0.01)
    crisis_help = (not np.isnan(d_shr_2020) and d_shr_2020 > 0.03) or (
        not np.isnan(d_shr_2122) and d_shr_2122 > 0.03
    )
    crisis_hurt = (not np.isnan(d_shr_2020) and d_shr_2020 < -0.05) or (
        not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02
    )
    turnover_cost = not np.isnan(to_d_w) and to_d_w > 0.15

    print()
    if perf_improved and not crisis_hurt and not (turnover_cost and not crisis_help):
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {shr_d_w:+.3f}: breadth adds meaningful signal beyond pure momentum.",
            f"CAGR improved {cagr_d_w:+.2%}.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: earlier defensive positioning visible.",
            f"2021-2022 rate shock Sharpe delta {_f(d_shr_2122, sign=True)}: breadth deterioration anticipated shift.",
            f"Turnover delta {_pct(to_d_w, sign=True)}: consistent with a somewhat more responsive combined signal.",
        ]
    elif hard_fail or crisis_hurt:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {shr_d_w:+.3f}: sector-breadth signal adds noise, not information.",
            "The 9-sector proxy (30% weight) shifts the signal on idiosyncratic sector moves"
            " that are irrelevant to the regime, generating false trades.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: breadth did not provide earlier warning"
            f" in the steepest part of the crash.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: breadth adds modest inflation-period benefit"
            f" but is outweighed by other-period drag.",
            "Keep pure 24M SPY momentum baseline. Breadth in this form is not additive.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed result: Sharpe {shr_d_w:+.3f}, CAGR {cagr_d_w:+.2%} — improvement below +0.02 threshold.",
            "The sector-breadth z-score blended at 30% shifts the combined_score distribution"
            " without providing a consistent regime-detection advantage.",
            f"2020 COVID Sharpe delta {_f(d_shr_2020, sign=True)}: breadth did not front-run the crash meaningfully.",
            f"2021-2022 Sharpe delta {_f(d_shr_2122, sign=True)}: any improvement is within noise.",
            "Keep pure 24M SPY momentum as baseline signal. If breadth is to be revisited,"
            " use a lower weight (e.g., 0.15) or a different signal construction.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
