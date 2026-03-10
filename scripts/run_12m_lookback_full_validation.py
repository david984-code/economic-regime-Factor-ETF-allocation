"""Full Walk-Forward Validation: 12M vs 24M Continuous Mapped Momentum.

Single variable change: market_lookback_months 24 -> 12.
Covers entire available dataset (2010-01-01 to present).
OOS segments start ~2015 (after 60-month training minimum).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation
from src.features.transforms import sigmoid

logging.basicConfig(level=logging.WARNING)

FULL_START = "2010-01-01"
FULL_END = None  # use today

SHARED_KWARGS = dict(
    start=FULL_START,
    end=FULL_END,
    min_train_months=60,
    test_months=12,
    expanding=True,
    use_stagflation_override=False,
    use_stagflation_risk_on_cap=False,
    use_regime_smoothing=False,
    use_hybrid_signal=True,
    hybrid_macro_weight=0.0,
    use_momentum=True,
    trend_filter_type="none",
    vol_scaling_method="none",
    portfolio_construction_method="equal_weight",
    momentum_12m_weight=0.0,
    quarterly_rebalance=False,
    fast_mode=False,
    skip_persist=True,
    use_vol_regime=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sortino(rets: pd.Series, rf_daily: float = 0.0, periods_per_year: int = 252) -> float:
    excess = (rets - rf_daily).dropna()
    if len(excess) < 5:
        return float("nan")
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = downside.std() * np.sqrt(periods_per_year)
    ann_excess = excess.mean() * periods_per_year
    return ann_excess / downside_std if downside_std > 0 else float("nan")


def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segments_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["segment"] != "OVERALL"].copy()


def _compute_risk_on(spy_monthly: pd.Series, lookback: int) -> tuple:
    """Reproduce exact risk_on pipeline: raw momentum -> expanding z-score -> sigmoid(z*0.25)."""
    n = len(spy_monthly)
    raw = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback:
            raw[i] = spy_monthly.iloc[i] / spy_monthly.iloc[i - lookback] - 1
    raw_s = pd.Series(raw, index=spy_monthly.index)

    min_history = max(lookback, 12)
    z = raw_s.copy()
    for i in range(n):
        trailing = raw_s.iloc[:i + 1].dropna()
        if len(trailing) >= min_history:
            z.iloc[i] = (raw_s.iloc[i] - trailing.mean()) / trailing.std()
        else:
            z.iloc[i] = 0.0

    risk_on = sigmoid(z * 0.25)
    return risk_on, raw_s, z


def _filter_segments(segs: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Keep segments whose test period overlaps [year_start, year_end]."""
    def overlaps(row):
        try:
            ts = pd.Period(row["test_start"], freq="M").year
            te = pd.Period(row["test_end"], freq="M").year
            return ts <= year_end and te >= year_start
        except Exception:
            return False
    mask = segs.apply(overlaps, axis=1)
    return segs[mask]


def _mean_metric(segs: pd.DataFrame, col: str) -> float:
    if col in segs.columns:
        return segs[col].dropna().mean()
    return float("nan")


def _get_month(series: pd.Series, month_str: str) -> float:
    ts = pd.Timestamp(month_str + "-01") + pd.offsets.MonthEnd(0)
    return series.get(ts, float("nan"))


def _fmt(v: float, pct: bool = True, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    if pct:
        return f"{v:+.2%}" if sign else f"{v:.2%}"
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def _fmt_f(v: float, decimals: int = 3, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    fmt = f"{{:+.{decimals}f}}" if sign else f"{{:.{decimals}f}}"
    return fmt.format(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("FULL WALK-FORWARD VALIDATION: 12M vs 24M Continuous Mapped Momentum")
    print(f"Full dataset start: {FULL_START}  |  OOS coverage starts ~2015")
    print("Single variable: market_lookback_months 24 -> 12")
    print("=" * 72)

    # -- Load prices for diagnostics ----------------------------------------
    print("\nLoading price data for diagnostics...")
    prices = fetch_prices(start=FULL_START, end=FULL_END)
    spy_monthly = prices["SPY"].resample("ME").last()
    spy_daily_ret = prices["SPY"].pct_change().dropna()

    ro_24, raw_24, z_24 = _compute_risk_on(spy_monthly, 24)
    ro_12, raw_12, z_12 = _compute_risk_on(spy_monthly, 12)

    # Constant to detect default (z=0 -> sigmoid(0)=0.5)
    neutral_val = float(sigmoid(0.0))

    def _first_non_neutral(ro: pd.Series) -> str:
        v = ro[ro != neutral_val]
        return v.index[0].strftime("%Y-%m") if len(v) else "none"

    print(f"\nPRE-RUN VERIFICATION")
    print("-" * 44)
    print(f"  First valid risk_on (24M): {_first_non_neutral(ro_24)}")
    print(f"  First valid risk_on (12M): {_first_non_neutral(ro_12)}")
    print(f"  Price data range: {spy_monthly.index[0].strftime('%Y-%m')} to {spy_monthly.index[-1].strftime('%Y-%m')}")
    print(f"  Parameter diff: ONLY market_lookback_months (24 vs 12)")

    # -- Run walk-forward validation -----------------------------------------
    print("\nRunning BASELINE (market_lookback_months=24)...")
    df_base = run_walk_forward_evaluation(**SHARED_KWARGS, market_lookback_months=24)

    print("Running EXPERIMENT (market_lookback_months=12)...")
    df_exp = run_walk_forward_evaluation(**SHARED_KWARGS, market_lookback_months=12)

    if df_base.empty or df_exp.empty:
        print("ERROR: one or both runs returned empty results.")
        sys.exit(1)

    segs_b = _segments_only(df_base)
    segs_e = _segments_only(df_exp)

    # Verify identical segments
    idx_b = segs_b[["test_start", "test_end"]].reset_index(drop=True)
    idx_e = segs_e[["test_start", "test_end"]].reset_index(drop=True)
    seg_match = idx_b.equals(idx_e)
    print(f"\nOOS segments identical: {'YES' if seg_match else 'NO -- STOP'}")
    if not seg_match:
        print("Segment mismatch detected. Cannot compare. Exiting.")
        sys.exit(1)

    print(f"OOS segment count: {len(segs_b)}")
    if len(segs_b) > 0:
        ts0 = segs_b["test_start"].iloc[0]
        ts1 = segs_b["test_end"].iloc[-1]
        print(f"OOS coverage: {ts0} to {ts1}")

    ob = _overall(df_base)
    oe = _overall(df_exp)

    # Build a combined daily return series for Sortino (concat OOS test returns)
    # Since we only have segment-level metrics from walk-forward, compute Sortino
    # from segment-level data as mean of segment Sortinos.
    def _seg_sortino_mean(segs: pd.DataFrame) -> float:
        # Approximate: use segment CAGR and Vol to infer downside. Not available.
        # Instead compute from overall CAGR/Sharpe via approximation (Sharpe * sqrt(252)/sqrt(252))
        # This is not rigorous -- we'll flag it.
        return float("nan")

    # Pull scalar metrics
    def _m(row, col):
        v = row.get(col, float("nan"))
        return float(v) if not isinstance(v, float) else v

    cagr_b   = _m(ob, "Strategy_CAGR")
    cagr_e   = _m(oe, "Strategy_CAGR")
    sharpe_b = _m(ob, "Strategy_Sharpe")
    sharpe_e = _m(oe, "Strategy_Sharpe")
    maxdd_b  = _m(ob, "Strategy_MaxDD")
    maxdd_e  = _m(oe, "Strategy_MaxDD")
    vol_b    = _m(ob, "Strategy_Vol")
    vol_e    = _m(oe, "Strategy_Vol")
    to_b     = _m(ob, "Strategy_Turnover")
    to_e     = _m(oe, "Strategy_Turnover")

    cagr_d   = cagr_e - cagr_b
    sharpe_d = sharpe_e - sharpe_b
    maxdd_d  = maxdd_e - maxdd_b
    vol_d    = vol_e - vol_b
    to_d     = to_e - to_b if not (np.isnan(to_b) or np.isnan(to_e)) else float("nan")
    to_ratio = to_e / to_b if (not np.isnan(to_b) and to_b > 0) else float("nan")

    # =========================================================================
    print("\n" + "=" * 72)
    print("1. FULL-PERIOD PERFORMANCE (OOS average across all walk-forward folds)")
    print("=" * 72)
    print(f"  {'Metric':30} {'Baseline (24M)':>14} {'Exp (12M)':>12} {'Delta':>10}")
    print("  " + "-" * 68)
    print(f"  {'CAGR':30} {_fmt(cagr_b):>14} {_fmt(cagr_e):>12} {_fmt(cagr_d, sign=True):>10}")
    print(f"  {'Sharpe':30} {_fmt_f(sharpe_b):>14} {_fmt_f(sharpe_e):>12} {_fmt_f(sharpe_d, sign=True):>10}")
    print(f"  {'MaxDD':30} {_fmt(maxdd_b):>14} {_fmt(maxdd_e):>12} {_fmt(maxdd_d, sign=True):>10}")
    print(f"  {'Vol':30} {_fmt(vol_b):>14} {_fmt(vol_e):>12} {_fmt(vol_d, sign=True):>10}")
    if not np.isnan(to_b):
        print(f"  {'Turnover':30} {_fmt(to_b):>14} {_fmt(to_e):>12} {_fmt(to_d, sign=True):>10}")
    else:
        print(f"  {'Turnover':30} {'n/a':>14} {'n/a':>12} {'n/a':>10}")
    print(f"  {'Sortino':30} {'[see note]':>14} {'[see note]':>12} {'n/a':>10}")
    print()
    print("  Note: Sortino not in walk-forward output; Sharpe is the primary risk-adj metric.")
    print("  Note: Metrics are mean of per-segment OOS results (standard walk-forward convention).")

    # =========================================================================
    print("\n" + "=" * 72)
    print("2. SUBPERIOD ANALYSIS (mean of OOS segments overlapping each window)")
    print("=" * 72)

    subperiods = [
        ("2000-2007", 2000, 2007, "pre-ETF universe: no OOS coverage expected"),
        ("2008-2012", 2008, 2012, "pre-OOS: training period only"),
        ("2013-2017", 2013, 2017, "partial OOS (first segments ~2015)"),
        ("2018-2020", 2018, 2020, "full OOS coverage"),
        ("2021-2022", 2021, 2022, "full OOS coverage"),
        ("2023-present", 2023, 2030, "full OOS coverage"),
    ]

    print(f"  {'Subperiod':15} {'B_CAGR':>8} {'E_CAGR':>8} {'dCAGR':>7} {'B_Sharpe':>9} {'E_Sharpe':>9} {'dSharpe':>8} {'B_MaxDD':>8} {'E_MaxDD':>8} {'Segs':>5} {'Note'}")
    print("  " + "-" * 120)

    for name, y0, y1, note in subperiods:
        sb = _filter_segments(segs_b, y0, y1)
        se = _filter_segments(segs_e, y0, y1)
        n_b = len(sb)
        if n_b == 0:
            print(f"  {name:15} {'--':>8} {'--':>8} {'--':>7} {'--':>9} {'--':>9} {'--':>8} {'--':>8} {'--':>8} {0:>5}  {note}")
            continue
        bc = _mean_metric(sb, "Strategy_CAGR")
        ec = _mean_metric(se, "Strategy_CAGR")
        bs = _mean_metric(sb, "Strategy_Sharpe")
        es = _mean_metric(se, "Strategy_Sharpe")
        bdd = _mean_metric(sb, "Strategy_MaxDD")
        edd = _mean_metric(se, "Strategy_MaxDD")
        dc = ec - bc
        ds = es - bs
        print(f"  {name:15} {_fmt(bc):>8} {_fmt(ec):>8} {_fmt(dc,sign=True):>7} "
              f"{_fmt_f(bs):>9} {_fmt_f(es):>9} {_fmt_f(ds,sign=True):>8} "
              f"{_fmt(bdd):>8} {_fmt(edd):>8} {n_b:>5}  {note}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("3. CRISIS DIAGNOSTICS (month-end risk_on from diagnostic computation)")
    print("=" * 72)

    crisis_events = [
        {
            "name": "2008 Global Financial Crisis",
            "peak": "2007-10",
            "trough": "2009-03",
            "key_months": ["2007-10", "2008-03", "2008-09", "2008-10", "2008-11", "2009-03", "2009-06"],
            "note": "In training period -- signal behavior shown, no OOS P&L available",
        },
        {
            "name": "2011 Euro Debt Crisis",
            "peak": "2011-04",
            "trough": "2011-10",
            "key_months": ["2011-04", "2011-07", "2011-08", "2011-10", "2011-12"],
            "note": "In training period",
        },
        {
            "name": "2018 Q4 Crash",
            "peak": "2018-09",
            "trough": "2018-12",
            "key_months": ["2018-08", "2018-09", "2018-10", "2018-11", "2018-12", "2019-01"],
            "note": "May be OOS depending on segment structure",
        },
        {
            "name": "2020 COVID Crash",
            "peak": "2020-02",
            "trough": "2020-03",
            "key_months": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-06"],
            "note": "OOS in full validation",
        },
        {
            "name": "2022 Rate Shock",
            "peak": "2022-01",
            "trough": "2022-10",
            "key_months": ["2021-12", "2022-01", "2022-03", "2022-06", "2022-09", "2022-10", "2022-12"],
            "note": "OOS in full validation",
        },
    ]

    for ev in crisis_events:
        print(f"\n  {ev['name']}  [{ev['note']}]")
        print(f"  {'Month':10} {'24M risk_on':>12} {'12M risk_on':>12} {'Delta':>10} {'Signal direction'}")
        print("  " + "-" * 62)
        for m in ev["key_months"]:
            r24 = _get_month(ro_24, m)
            r12 = _get_month(ro_12, m)
            d   = r12 - r24 if not (np.isnan(r24) or np.isnan(r12)) else float("nan")
            r24s = _fmt_f(r24) if not np.isnan(r24) else "n/a"
            r12s = _fmt_f(r12) if not np.isnan(r12) else "n/a"
            ds   = _fmt_f(d, sign=True) if not np.isnan(d) else "n/a"
            if np.isnan(d):
                direction = "no data"
            elif abs(d) < 0.01:
                direction = "models agree"
            elif d < -0.02:
                direction = "12M MORE defensive"
            elif d > 0.02:
                direction = "12M MORE aggressive"
            else:
                direction = "minor diff"
            print(f"  {m:10} {r24s:>12} {r12s:>12} {ds:>10}  {direction}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("4. SIGNAL DISTRIBUTION COMPARISON (monthly, full diagnostic window)")
    print("=" * 72)
    r24c = ro_24.dropna()
    r12c = ro_12.dropna()
    spy_ret_m = spy_monthly.pct_change().dropna()

    print(f"  {'Metric':40} {'Baseline (24M)':>14} {'Exp (12M)':>14}")
    print("  " + "-" * 70)
    print(f"  {'Mean risk_on':40} {r24c.mean():>14.3f} {r12c.mean():>14.3f}")
    print(f"  {'Std risk_on':40} {r24c.std():>14.3f} {r12c.std():>14.3f}")
    print(f"  {'Min risk_on':40} {r24c.min():>14.3f} {r12c.min():>14.3f}")
    print(f"  {'Max risk_on':40} {r24c.max():>14.3f} {r12c.max():>14.3f}")
    print(f"  {'% months risk_on < 0.35 (defensive)':40} {(r24c < 0.35).mean():>14.1%} {(r12c < 0.35).mean():>14.1%}")
    print(f"  {'% months risk_on < 0.40':40} {(r24c < 0.40).mean():>14.1%} {(r12c < 0.40).mean():>14.1%}")
    print(f"  {'% months risk_on > 0.60':40} {(r24c > 0.60).mean():>14.1%} {(r12c > 0.60).mean():>14.1%}")
    print(f"  {'% months risk_on > 0.65 (aggressive)':40} {(r24c > 0.65).mean():>14.1%} {(r12c > 0.65).mean():>14.1%}")
    neutral_lo, neutral_hi = 0.45, 0.55
    b_neutral = ((r24c >= neutral_lo) & (r24c <= neutral_hi)).mean()
    e_neutral = ((r12c >= neutral_lo) & (r12c <= neutral_hi)).mean()
    print(f"  {'% months neutral [0.45, 0.55]':40} {b_neutral:>14.1%} {e_neutral:>14.1%}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("5. REGIME RESPONSIVENESS (z-score sign crossings and timing)")
    print("=" * 72)

    def _crossings_by_year(z: pd.Series) -> dict:
        clean = z.dropna()
        sign  = np.sign(clean)
        cross = (sign != sign.shift(1)).astype(int)
        cross.iloc[0] = 0
        return cross.groupby(cross.index.year).sum().to_dict()

    cx_24 = _crossings_by_year(z_24)
    cx_12 = _crossings_by_year(z_12)
    all_yrs = sorted(set(cx_24) | set(cx_12))

    print(f"  Z-score sign crossings per year (proxy for regime flips):")
    print(f"  {'Year':6} {'24M':>8} {'12M':>8} {'Delta':>8}")
    print("  " + "-" * 32)
    tot_24 = tot_12 = 0
    for y in all_yrs:
        c24 = cx_24.get(y, 0)
        c12 = cx_12.get(y, 0)
        tot_24 += c24
        tot_12 += c12
        print(f"  {y:6} {c24:>8} {c12:>8} {c12 - c24:>+8}")
    print("  " + "-" * 32)
    print(f"  {'Total':6} {tot_24:>8} {tot_12:>8} {tot_12 - tot_24:>+8}")

    ratio_cx = tot_12 / tot_24 if tot_24 > 0 else float("nan")
    print(f"\n  12M crossing frequency: {ratio_cx:.1f}x baseline (24M)")

    # Average time to cross (lag analysis):
    # For major risk-off events, find when each signal first went below 0
    def _first_z_cross_below_zero(z: pd.Series, after: str, before: str) -> str:
        window = z.loc[after:before]
        neg = window[window < 0]
        return neg.index[0].strftime("%Y-%m") if len(neg) > 0 else "never"

    print(f"\n  First z-score < 0 (risk-off lean) by crisis:")
    print(f"  {'Crisis':25} {'24M':>10} {'12M':>10} {'12M leads?'}")
    print("  " + "-" * 52)
    crisis_lag = [
        ("2018 Q4", "2018-09", "2019-06"),
        ("2020 COVID", "2020-01", "2020-06"),
        ("2022 Rate Shock", "2022-01", "2023-01"),
    ]
    for cname, after, before in crisis_lag:
        d24 = _first_z_cross_below_zero(z_24, after, before)
        d12 = _first_z_cross_below_zero(z_12, after, before)
        if d24 == "never" and d12 == "never":
            lead = "neither crossed"
        elif d12 == "never":
            lead = "12M never crossed"
        elif d24 == "never":
            lead = "12M crossed, 24M did not"
        else:
            t24 = pd.Timestamp(d24 + "-01")
            t12 = pd.Timestamp(d12 + "-01")
            diff_m = (t24.year - t12.year) * 12 + (t24.month - t12.month)
            if diff_m > 0:
                lead = f"12M leads by {diff_m}m"
            elif diff_m < 0:
                lead = f"12M lags by {-diff_m}m"
            else:
                lead = "same month"
        print(f"  {cname:25} {d24:>10} {d12:>10}  {lead}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("6. TURNOVER ANALYSIS")
    print("=" * 72)

    if not np.isnan(to_b):
        print(f"  Full-period annualized turnover:")
        print(f"    Baseline (24M):   {to_b:.2%}")
        print(f"    Experiment (12M): {to_e:.2%}")
        print(f"    Delta:            {to_d:+.2%}")
        print(f"    Ratio:            {to_ratio:.2f}x")

    # Per-segment turnover
    if "Strategy_Turnover" in segs_b.columns:
        print(f"\n  Per-segment turnover:")
        print(f"  {'Segment':10} {'Test Period':22} {'Baseline TO':>12} {'Exp TO':>10} {'Delta':>8}")
        print("  " + "-" * 66)
        for (_, rb), (_, re) in zip(segs_b.iterrows(), segs_e.iterrows()):
            period = f"{rb['test_start']} to {rb['test_end']}"
            tb = rb.get("Strategy_Turnover", float("nan"))
            te = re.get("Strategy_Turnover", float("nan"))
            td = te - tb if not (np.isnan(tb) or np.isnan(te)) else float("nan")
            print(f"  {int(rb['segment']):>10} {period:22} {_fmt(tb):>12} {_fmt(te):>10} {_fmt(td, sign=True):>8}")

    # Is turnover increase concentrated?
    # Proxy: check if z-score crossings cluster in high-vol periods
    vix_proxy = spy_daily_ret.rolling(21).std() * np.sqrt(252)
    vix_m = vix_proxy.resample("ME").last()
    high_vol_months = vix_m[vix_m > vix_m.quantile(0.75)].index

    def _crossing_dates(z: pd.Series) -> list:
        clean = z.dropna()
        sign  = np.sign(clean)
        cross_mask = sign != sign.shift(1)
        cross_mask.iloc[0] = False
        return list(cross_mask[cross_mask].index)

    cx_dates_24 = _crossing_dates(z_24)
    cx_dates_12 = _crossing_dates(z_12)
    cx12_in_hv = sum(1 for d in cx_dates_12 if d in high_vol_months)
    pct_in_hv = cx12_in_hv / len(cx_dates_12) if cx_dates_12 else float("nan")

    print(f"\n  Turnover concentration analysis:")
    print(f"    12M total z-crossings: {len(cx_dates_12)}")
    print(f"    12M crossings in high-vol months (top quartile realized vol): {cx12_in_hv} ({pct_in_hv:.0%})")
    if pct_in_hv > 0.6:
        print(f"    -> Turnover is concentrated in volatile regimes (expected)")
    else:
        print(f"    -> Turnover is spread across market regimes")

    # =========================================================================
    print("\n" + "=" * 72)
    print("7. ROBUSTNESS SUMMARY")
    print("=" * 72)

    # Count subperiods where 12M beats 24M
    periods_with_data = []
    for name, y0, y1, _ in subperiods:
        sb = _filter_segments(segs_b, y0, y1)
        se = _filter_segments(segs_e, y0, y1)
        if len(sb) == 0:
            continue
        bs = _mean_metric(sb, "Strategy_Sharpe")
        es = _mean_metric(se, "Strategy_Sharpe")
        beat = es > bs
        periods_with_data.append((name, beat, es - bs))

    n_beat = sum(1 for _, b, _ in periods_with_data if b)
    n_total = len(periods_with_data)

    print(f"  Subperiods with OOS data: {n_total}")
    print(f"  12M beats 24M (Sharpe) in: {n_beat} / {n_total} subperiods")
    print()
    for name, beat, ds in periods_with_data:
        icon = "+" if beat else "-"
        print(f"    [{icon}] {name:15}  Sharpe delta = {ds:+.3f}")

    print()
    # Assess recency concentration
    recent_periods = [n for n, b, d in periods_with_data if b and ("2023" in n or "2018" in n)]
    older_periods  = [n for n, b, d in periods_with_data if b and "2013" in n]

    if n_beat == n_total:
        robustness = "IMPROVEMENT IS BROAD: 12M outperforms 24M across all available OOS subperiods."
    elif n_beat >= n_total * 0.67:
        robustness = f"IMPROVEMENT IS MOSTLY BROAD: 12M wins in {n_beat}/{n_total} subperiods. Check if losses are in high-stress periods."
    elif n_beat == n_total - 1:
        robustness = "MOSTLY BROAD with one subperiod exception. Identify which period and why."
    else:
        robustness = f"IMPROVEMENT IS CONCENTRATED: 12M only wins in {n_beat}/{n_total} subperiods. Likely recency bias."

    print(f"  Assessment: {robustness}")

    crisis_check = []
    for ev_name, after, before in crisis_lag:
        d12 = _first_z_cross_below_zero(z_12, after, before)
        d24 = _first_z_cross_below_zero(z_24, after, before)
        if d12 != "never" and d24 == "never":
            crisis_check.append(f"  12M responded defensively in {ev_name} while 24M did not")
        elif d12 != "never" and d24 != "never":
            t24 = pd.Timestamp(d24 + "-01")
            t12 = pd.Timestamp(d12 + "-01")
            lag = (t24.year - t12.year) * 12 + (t24.month - t12.month)
            if lag > 0:
                crisis_check.append(f"  12M turned defensive {lag}m earlier in {ev_name}")
            elif lag < 0:
                crisis_check.append(f"  12M turned defensive {-lag}m LATER in {ev_name} (worse)")

    if crisis_check:
        print("\n  Crisis responsiveness:")
        for c in crisis_check:
            print(c)
    else:
        print("\n  Crisis responsiveness: 12M and 24M responded similarly in all tested crises.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("8. FINAL VERDICT")
    print("=" * 72)

    # Kill switch
    ks_sharpe = sharpe_d < 0.02
    ks_cagr   = cagr_d < 0.0025

    difficult_years = {2020, 2022}
    def _in_diff(row):
        try:
            y0 = pd.Period(row["test_start"], freq="M").year
            y1 = pd.Period(row["test_end"], freq="M").year
            return any(y in difficult_years for y in range(y0, y1 + 1))
        except Exception:
            return False

    mask_b = segs_b.apply(_in_diff, axis=1)
    mask_e = segs_e.apply(_in_diff, axis=1)
    ds_b_diff = segs_b.loc[mask_b, "Strategy_Sharpe"].mean() if mask_b.any() else float("nan")
    ds_e_diff = segs_e.loc[mask_e, "Strategy_Sharpe"].mean() if mask_e.any() else float("nan")
    difficult_improved = (not np.isnan(ds_e_diff) and not np.isnan(ds_b_diff) and ds_e_diff > ds_b_diff)

    high_churn_crossing = (tot_24 > 0 and tot_12 > 1.5 * tot_24)
    high_churn_to = (not np.isnan(to_ratio) and to_ratio > 1.5)
    high_churn = high_churn_crossing or high_churn_to
    sharpe_thresh = 0.04 if high_churn else 0.02

    esc_sharpe = sharpe_d >= sharpe_thresh
    esc_cagr   = cagr_d >= 0.0025
    esc_maxdd  = maxdd_d > 0.01
    esc_diff   = difficult_improved
    passes = esc_sharpe or esc_cagr or esc_maxdd or esc_diff

    print(f"  Screening thresholds:")
    print(f"    High churn flag:       {'YES (threshold = +0.04)' if high_churn else 'NO  (threshold = +0.02)'}")
    print(f"    Sharpe delta:          {sharpe_d:+.3f}  (need >= {sharpe_thresh:.2f})  {'PASS' if esc_sharpe else 'FAIL'}")
    print(f"    CAGR delta:            {cagr_d:+.2%}  (need >= +0.25%)  {'PASS' if esc_cagr else 'FAIL'}")
    print(f"    MaxDD delta:           {maxdd_d:+.2%}  (need > +1.0%)  {'PASS' if esc_maxdd else 'FAIL'}")
    b_diff_str = f"{ds_b_diff:.3f}" if not np.isnan(ds_b_diff) else "n/a"
    e_diff_str = f"{ds_e_diff:.3f}" if not np.isnan(ds_e_diff) else "n/a"
    print(f"    Difficult-period Sharpe: baseline={b_diff_str}, exp={e_diff_str}  {'PASS' if esc_diff else 'FAIL'}")
    print()
    print(f"  Subperiod breadth:  {n_beat}/{n_total} subperiods won by 12M")
    print(f"  Turnover ratio:     {to_ratio:.2f}x baseline" if not np.isnan(to_ratio) else "  Turnover ratio: n/a")
    print()

    if not passes:
        verdict = "REJECT -- does not meet screening thresholds in full walk-forward"
    elif n_beat < n_total / 2:
        verdict = "CONDITIONAL ACCEPT -- passes thresholds but improvement is concentrated. Do not replace baseline without further investigation."
    elif not difficult_improved and not np.isnan(ds_b_diff):
        verdict = "CONDITIONAL ACCEPT -- passes CAGR/Sharpe thresholds but does NOT improve difficult-period performance. Consider accepting only if turnover cost is acceptable."
    else:
        verdict = "ACCEPT -- passes full walk-forward thresholds with broad subperiod improvement"

    print(f"  FULL WALK-FORWARD VERDICT: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
