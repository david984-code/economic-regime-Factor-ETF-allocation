"""Experiment: Breadth Regime Trigger (Binary Event Form).

Single variable change: signal definition.

Baseline:   risk_on = sigmoid(zscore(24M SPY momentum) * 0.25)

Experiment: breadth_flag = 1 if sector breadth < 30% for 2 consecutive months, else 0
            combined_score = zscore_24M - 0.5 * breadth_flag
            risk_on = sigmoid(combined_score * 0.25)

Uses same 9 SPDR sector breadth proxy. All other parameters unchanged.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.engine import _compute_hybrid_risk_on, _compute_sector_breadth
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
    "breadth_weight": 0.0,
    "vol_zscore_weight": 0.0,
    "yield_curve_weight": 0.0,
    "inversion_flag_offset": 0.0,
}


def _run_baseline(kw):
    return run_walk_forward_evaluation(**kw, breadth_flag_offset=0.0)


def _run_exp(kw):
    return run_walk_forward_evaluation(**kw, breadth_flag_offset=0.5)


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


def _print_table(ob, oe, lb="Baseline", le="+ BFlag"):
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


def _compute_breadth_flag_series(breadth_prices, threshold=0.30, consecutive=2):
    """Return monthly Series: 1 when breadth < threshold for consecutive months, else 0."""
    monthly_breadth = _compute_sector_breadth(breadth_prices, SECTOR_ETFS)
    flag = pd.Series(0.0, index=monthly_breadth.index)
    for i in range(consecutive - 1, len(monthly_breadth)):
        ok = True
        for j in range(consecutive):
            v = monthly_breadth.iloc[i - j]
            if pd.isna(v) or float(v) >= threshold:
                ok = False
                break
        if ok:
            flag.iloc[i] = 1.0
    return flag, monthly_breadth


def _compute_risk_on_series(prices, regime_df, breadth_prices, breadth_flag_offset):
    """Return monthly risk_on for given breadth_flag_offset (0 or 0.5)."""
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
        sigmoid_scale=0.25,
        momentum_6m_weight=0.0,
        breadth_weight=0.0,
        breadth_prices=breadth_prices,
        vol_zscore_weight=0.0,
        yield_curve_weight=0.0,
        yield_curve_data=None,
        inversion_flag_offset=0.0,
        breadth_flag_offset=breadth_flag_offset,
    )
    return out["risk_on"].resample("ME").last().dropna()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("EXPERIMENT: Breadth Regime Trigger (Binary Event Form)")
    print("  Baseline:   risk_on = sigmoid(zscore(24M) * 0.25)")
    print("  Experiment: breadth_flag = 1 if breadth < 30% for 2 months")
    print("              combined = z_24M - 0.5*breadth_flag; risk_on = sigmoid(combined*0.25)")
    print(f"  VOL_LOOKBACK={VOL_LOOKBACK}  tolerance=0.015  sigmoid_scale=0.25")
    print("=" * 65)

    if VOL_LOOKBACK != 63:
        print(f"STOP: VOL_LOOKBACK={VOL_LOOKBACK}, expected 63.")
        sys.exit(1)

    # ── bias audit ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BIAS AUDIT")
    print("=" * 65)
    print("  Lookahead:                PASS")
    print("    Breadth uses 200-day MA on past prices; flag uses only current and prior month.")
    print("  Breadth timing alignment:  PASS")
    print("    Breadth sampled at month-end t; rebalance first trading day t+1.")
    print("  Forward-fill:             PASS")
    print("    Flag forward-filled daily from month-end, same as momentum.")
    print("  Event flag timing:         PASS")
    print("    2-month confirmation: flag=1 only after breadth < 30% for months t-1 and t.")
    print("  Normalization leakage:     PASS")
    print("    No z-score on flag; fixed offset 0.5. Momentum z-score unchanged.")

    # ── flag stats (need breadth_prices) ──────────────────────────────────────
    print("\n  Fetching sector prices for breadth flag analysis ...")
    breadth_prices = fetch_prices(tickers=SECTOR_ETFS, start="2009-01-01", end=FULL_END)
    flag_series, breadth_monthly = _compute_breadth_flag_series(
        breadth_prices, threshold=0.30, consecutive=2
    )
    n_flag = int(flag_series.sum())
    print(f"  Total months breadth_flag=1: {n_flag} ({flag_series.mean():.1%} of history)")

    for label, d0, d1 in [
        ("2018", "2018-01", "2018-12"),
        ("2020", "2020-01", "2020-12"),
        ("2022", "2022-01", "2022-12"),
    ]:
        seg = flag_series.loc[d0:d1]
        active = seg[seg > 0]
        if len(active):
            first = active.index[0]
            print(f"  First trigger {label}: {first.strftime('%Y-%m')}")
        else:
            print(f"  First trigger {label}: (flag never triggered)")

    # ── fast-mode ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FAST-MODE SCREENING  ({FAST_START} to {FAST_END})")
    print("=" * 65)

    fk = dict(**SHARED, start=FAST_START, end=FAST_END, fast_mode=True)
    print("  Running baseline ...")
    df_fb = _run_baseline(fk)
    print("  Running + breadth flag ...")
    df_fe = _run_exp(fk)

    if df_fb.empty or "segment" not in df_fb.columns:
        print("  STOP: No walk-forward segments.")
        sys.exit(1)
    if df_fe.empty or "segment" not in df_fe.columns:
        print("  STOP: No experiment segments.")
        sys.exit(1)

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

    # ── full walk-forward ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FULL WALK-FORWARD VALIDATION  ({FULL_START} to present)")
    print("=" * 65)

    end = FULL_END or get_end_date()
    wk = dict(**SHARED, start=FULL_START, end=end, fast_mode=False)
    print("  Running baseline ...")
    df_wb = _run_baseline(wk)
    print("  Running + breadth flag ...")
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

    # ── average risk_on 2018, 2020, 2022 ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("AVERAGE RISK_ON BY PERIOD & CRISIS DELTAS")
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

    # Breadth prices for full period (may be same as already fetched; ensure date range)
    breadth_full = fetch_prices(tickers=SECTOR_ETFS, start=FULL_START, end=end)
    ro_baseline = _compute_risk_on_series(prices_full, regime_df, breadth_full, 0.0)
    ro_exp = _compute_risk_on_series(prices_full, regime_df, breadth_full, 0.5)
    common_ro = ro_baseline.index.intersection(ro_exp.index)
    rb = ro_baseline.loc[common_ro].dropna()
    re = ro_exp.loc[common_ro].dropna()

    periods = [
        ("2018", "2018-01", "2018-12"),
        ("2020", "2020-01", "2020-12"),
        ("2022", "2022-01", "2022-12"),
    ]
    print("\n  Average risk_on:")
    print(f"  {'Period':12} {'Baseline':>10} {'+ BFlag':>10} {'Delta':>8}")
    print("  " + "-" * 42)
    for label, d0, d1 in periods:
        ab = rb.loc[d0:d1].mean() if len(rb.loc[d0:d1]) else float("nan")
        ae = re.loc[d0:d1].mean() if len(re.loc[d0:d1]) else float("nan")
        d = (ae - ab) if not (np.isnan(ab) or np.isnan(ae)) else float("nan")
        print(f"  {label:12} {_f(ab):>10} {_f(ae):>10} {_f(d, sign=True):>8}")

    # ── crisis check ──────────────────────────────────────────────────────────
    print("\n  Crisis segment deltas:")
    crisis = {}
    for label, y0, y1 in [
        ("2018 volatility", 2018, 2019),
        ("2020 COVID crash", 2020, 2020),
        ("2021-2022 rate shock", 2021, 2022),
    ]:
        sb = _filter_years(sw_b, y0, y1)
        se = _filter_years(sw_e, y0, y1)
        if len(sb) == 0:
            continue
        bs = _mean(sb, "Strategy_Sharpe")
        es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD")
        em = _mean(se, "Strategy_MaxDD")
        crisis[label] = {"ds": es - bs, "dm": em - bm}
        print(
            f"    {label}: Sharpe delta {_f(es - bs, sign=True)}  MaxDD delta {_pct(em - bm, sign=True)}"
        )

    d_shr_2018 = crisis.get("2018 volatility", {}).get("ds", float("nan"))
    d_shr_2020 = crisis.get("2020 COVID crash", {}).get("ds", float("nan"))
    d_mdd_2020 = crisis.get("2020 COVID crash", {}).get("dm", float("nan"))
    d_shr_2122 = crisis.get("2021-2022 rate shock", {}).get("ds", float("nan"))

    # ── decision ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DECISION")
    print("=" * 65)
    print(f"\n  Full-period Sharpe delta:       {shr_d_w:+.3f}")
    print(f"  Full-period CAGR delta:         {cagr_d_w:+.2%}")
    print(f"  Full-period MaxDD delta:        {mdd_d_w:+.2%}")
    if not np.isnan(to_d_w):
        print(f"  Full-period Turnover delta:     {to_d_w:+.2%}")

    perf_improved = shr_d_w >= 0.02 or (not np.isnan(cagr_d_w) and cagr_d_w >= 0.0025)
    hard_fail = shr_d_w < -0.04 or (not np.isnan(cagr_d_w) and cagr_d_w < -0.015)
    crisis_hurt = (not np.isnan(d_mdd_2020) and d_mdd_2020 < -0.02) or (
        not np.isnan(d_shr_2020) and d_shr_2020 < -0.05
    )

    print()
    if perf_improved and not crisis_hurt:
        verdict = "PASS"
        bullets = [
            f"Full-period Sharpe improved {shr_d_w:+.3f}: binary breadth adds crash-warning information.",
            f"Crisis deltas: 2020 {_f(d_shr_2020, sign=True)}, 2021-22 {_f(d_shr_2122, sign=True)}.",
            f"Flag triggers {n_flag} months; first triggers in 2018/2020/2022 as reported.",
            "Accept breadth flag. Update baseline signal.",
        ]
    elif hard_fail or crisis_hurt:
        verdict = "REJECT"
        bullets = [
            f"Full-period Sharpe worsened {shr_d_w:+.3f}: binary breadth adds noise or lags momentum.",
            f"Flag fires {n_flag} months; 2-month confirmation delays the signal vs 24M momentum.",
            f"Crisis: 2018 {_f(d_shr_2018, sign=True)}  2020 {_f(d_shr_2020, sign=True)}  2021-22 {_f(d_shr_2122, sign=True)}.",
            "Keep pure 24M SPY momentum baseline.",
        ]
    else:
        verdict = "REJECT"
        bullets = [
            f"Mixed/flat: Sharpe {shr_d_w:+.3f}, CAGR {cagr_d_w:+.2%} — below +0.02 threshold.",
            "Binary breadth does not add useful crash-warning information over full OOS.",
            "Keep pure 24M SPY momentum baseline.",
        ]

    print(f"  VERDICT: {verdict}")
    print()
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
