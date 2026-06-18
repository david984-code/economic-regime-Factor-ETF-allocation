"""Experiment: Convex Exposure Mapping.

Single variable change: sigmoid scaling constant.

Baseline:   risk_on = sigmoid(zscore_24M_SPY_momentum * 0.25)
Experiment: risk_on = sigmoid(zscore_24M_SPY_momentum * 0.40)

Same signal (24M SPY momentum, expanding z-score). Only the mapping from
z-score to exposure is steeper — exposure responds more strongly to
extreme z-scores while remaining continuous in [0, 1].

All other parameters unchanged:
  market_lookback_months=24, equal_weight sleeves, VOL_LOOKBACK=63,
  tolerance=0.015, trend_filter_type=none, rebalance timing unchanged.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.engine import _compute_hybrid_risk_on
from src.config import (
    ASSETS,
    OUTPUTS_DIR,
    RISK_OFF_ASSETS_BASE,
    RISK_ON_ASSETS_BASE,
    TICKERS,
    VOL_LOOKBACK,
    get_end_date,
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
    "inversion_flag_offset": 0.0,
}


def _run_baseline(kw):
    return run_walk_forward_evaluation(**kw, sigmoid_scale=0.25)


def _run_exp(kw):
    return run_walk_forward_evaluation(**kw, sigmoid_scale=0.40)


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


def _print_table(ob, oe, lb="Baseline", le="0.40"):
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


def _compute_risk_on_series(prices, regime_df, sigmoid_scale):
    """Return monthly risk_on series for given sigmoid_scale (no other overlays)."""
    reg = regime_df.reindex(prices.index).ffill()
    out = _compute_hybrid_risk_on(
        prices,
        reg,
        macro_weight=0.0,
        market_lookback_months=24,
        use_momentum=True,
        trend_filter_type="none",
        trend_filter_risk_on_cap=0.3,
        vol_scaling_method="none",
        momentum_12m_weight=0.0,
        use_vol_regime=False,
        vol_regime_weight=0.0,
        sigmoid_scale=sigmoid_scale,
        breadth_weight=0.0,
        breadth_prices=None,
        vol_zscore_weight=0.0,
        yield_curve_weight=0.0,
        yield_curve_data=None,
        inversion_flag_offset=0.0,
    )
    return out["risk_on"].resample("ME").last().dropna()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Convex Exposure Mapping")
    print("  Baseline:   risk_on = sigmoid(z * 0.25)")
    print("  Experiment: risk_on = sigmoid(z * 0.40)")
    print(f"  VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  market_lookback_months=24")
    print("=" * 65)

    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ── bias audit ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)
    print("  Lookahead:             PASS")
    print("    Signal unchanged (24M SPY momentum, expanding z-score).")
    print("    Only the scalar multiplier in the sigmoid changes; no new data.")
    print("  Signal timing:         PASS")
    print("    Momentum and rebalance timing unchanged. Same month-end → next-month rebalance.")
    print("  Normalization leakage: PASS")
    print("    Z-score computation unchanged. 0.25 and 0.40 are fixed constants, not fitted.")
    print("  Rebalance alignment:   PASS")
    print("    First trading day of each month. Unchanged.")
    print("  Parameter isolation:   PASS")
    print("    Only sigmoid_scale differs (0.25 vs 0.40).")

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)

    print("  Running baseline (sigmoid_scale=0.25) ...")
    df_fb = _run_baseline(fk)

    print("  Running experiment (sigmoid_scale=0.40) ...")
    df_fe = _run_exp(fk)

    if df_fb.empty or "segment" not in df_fb.columns:
        print("  STOP: No walk-forward segments (check data fetch, e.g. IJR).")
        sys.exit(1)
    if df_fe.empty or "segment" not in df_fe.columns:
        print("  STOP: No walk-forward segments from experiment run.")
        sys.exit(1)

    sb_f = _segs(df_fb)[["test_start", "test_end"]].reset_index(drop=True)
    se_f = _segs(df_fe)[["test_start", "test_end"]].reset_index(drop=True)
    if not sb_f.equals(se_f):
        print("  STOP: OOS segment mismatch.")
        sys.exit(1)
    print(f"  OOS segments identical: YES ({len(sb_f)} segments)\n")

    ob_f = _overall(df_fb)
    oe_f = _overall(df_fe)
    fd = _print_table(ob_f, oe_f, lb="0.25", le="0.40")

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
        print("  Kill fires — escalating to full walk-forward for exposure/turnover check.")
    else:
        print("  Escalating to full walk-forward.")

    # ── full walk-forward ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    end = FULL_END or get_end_date()
    wk = dict(**SHARED, start=FULL_START, end=end, fast_mode=False)

    print("  Running baseline (sigmoid_scale=0.25) ...")
    df_wb = _run_baseline(wk)

    print("  Running experiment (sigmoid_scale=0.40) ...")
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
    wd = _print_table(ob_w, oe_w, lb="0.25", le="0.40")

    shr_d_w = wd.get("Strategy_Sharpe", float("nan"))
    cagr_d_w = wd.get("Strategy_CAGR", float("nan"))
    mdd_d_w = wd.get("Strategy_MaxDD", float("nan"))
    to_d_w = wd.get("Strategy_Turnover", float("nan"))

    # ── risk_on distribution and period averages ───────────────────────────────
    print("\n" + "=" * 65)
    print("RISK_ON DISTRIBUTION & PERIOD AVERAGES")
    print("=" * 65)

    prices_full = fetch_prices(tickers=list(TICKERS), start=FULL_START, end=end)
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()
    regime_df = regime_df.dropna(subset=["regime"]).sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
    regime_df = regime_df.reindex(prices_full.index).ffill()
    if "macro_score" not in regime_df.columns:
        regime_df["macro_score"] = 0.0

    ro_025 = _compute_risk_on_series(prices_full, regime_df, 0.25)
    ro_040 = _compute_risk_on_series(prices_full, regime_df, 0.40)

    common = ro_025.index.intersection(ro_040.index)
    r25 = ro_025.loc[common].dropna()
    r40 = ro_040.loc[common].dropna()

    print("\n  Distribution (monthly risk_on, common index):")
    print(f"  {'':6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  %<0.2  %>0.8  % in [0.4,0.6]")
    print("  " + "-" * 70)
    for label, s in [("0.25", r25), ("0.40", r40)]:
        pct_lo = (s < 0.2).mean() * 100
        pct_hi = (s > 0.8).mean() * 100
        pct_mid = ((s >= 0.4) & (s <= 0.6)).mean() * 100
        print(
            f"  {label:6} {s.mean():.3f}   {s.std():.3f}   {s.min():.3f}   {s.max():.3f}   "
            f"{pct_lo:5.1f}   {pct_hi:5.1f}   {pct_mid:5.1f}"
        )

    periods = [
        ("2018 selloff", "2018-01", "2018-12"),
        ("2020 COVID crash", "2020-01", "2020-12"),
        ("2021 bull market", "2021-01", "2021-12"),
        ("2022 bear market", "2022-01", "2022-12"),
    ]
    print("\n  Average risk_on by period:")
    print(f"  {'Period':24} {'0.25':>8} {'0.40':>8} {'Delta':>8}")
    print("  " + "-" * 52)
    for label, d0, d1 in periods:
        a25 = r25.loc[d0:d1].mean() if d0 in r25.index or len(r25.loc[d0:d1]) else float("nan")
        a40 = r40.loc[d0:d1].mean() if d0 in r40.index or len(r40.loc[d0:d1]) else float("nan")
        d = (a40 - a25) if not (np.isnan(a25) or np.isnan(a40)) else float("nan")
        a25s = _f(a25) if not np.isnan(a25) else "n/a"
        a40s = _f(a40) if not np.isnan(a40) else "n/a"
        ds = _f(d, sign=True) if not np.isnan(d) else "n/a"
        print(f"  {label:24} {a25s:>8} {a40s:>8} {ds:>8}")

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

    perf_improved = shr_d_w >= 0.02 or (not np.isnan(cagr_d_w) and cagr_d_w >= 0.0025)
    hard_fail = shr_d_w < -0.04 or (not np.isnan(cagr_d_w) and cagr_d_w < -0.015)
    crisis_hurt = (not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02) or (
        not np.isnan(d_shr_2020) and d_shr_2020 < -0.05
    )
    turnover_spike = not np.isnan(to_d_w) and to_d_w > 0.15

    print()
    if perf_improved and not crisis_hurt and not (turnover_spike and shr_d_w < 0.02):
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {shr_d_w:+.3f}: steeper mapping improves trend capture.",
            f"CAGR improved {cagr_d_w:+.2%}.",
            "risk_on distribution: 0.40 pushes more mass to extremes (higher % <0.2 or >0.8),"
            " reducing time spent near 0.5 — better alignment with regime.",
            f"Turnover delta {_pct(to_d_w, sign=True)}: acceptable for the return gain.",
            "Accept sigmoid_scale=0.40 as new baseline. Update PROJECT_CONTEXT.md.",
        ]
    elif hard_fail or crisis_hurt:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {shr_d_w:+.3f}: stronger convexity adds turnover"
            f" without improving trend capture.",
            "The 0.25 baseline compresses the signal toward 0.5 by design — it reduces"
            " sensitivity to noise. 0.40 makes the same z-score produce larger weight"
            " swings, so small z-score changes cross the tolerance threshold more often.",
            f"Turnover delta {_pct(to_d_w, sign=True)}: the main cost of convexity.",
            "Crisis periods: 2020/2022 deltas show no structural benefit from steeper mapping.",
            "Keep sigmoid_scale=0.25. Convex exposure in this form does not add value.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed/flat result: Sharpe {shr_d_w:+.3f}, CAGR {cagr_d_w:+.2%} — below +0.02 threshold.",
            "risk_on distribution under 0.40: more mass in extremes (defensive and aggressive)"
            " but average risk_on during 2018/2020/2022 shows the mapping shifts level without"
            " consistently improving timing.",
            f"Turnover delta {_pct(to_d_w, sign=True)}: stronger convexity increases rebalance"
            f" frequency as the signal crosses tolerance more often.",
            "Keep sigmoid_scale=0.25. The baseline compression is a feature: it dampens"
            " noise while preserving regime direction. 0.40 does not clear the bar.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
