"""Experiment: 12M vs 24M continuous mapped momentum.

Single variable change: market_lookback_months 24 -> 12.
All other parameters, pipeline steps, and conventions are identical.
Fast-mode window: 2018-01-01 to 2024-12-31.
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

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"

SHARED_KWARGS = dict(
    start=FAST_START,
    end=FAST_END,
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
    fast_mode=True,
    skip_persist=True,
    use_vol_regime=False,
)


def _compute_risk_on(spy_monthly: pd.Series, lookback: int) -> tuple:
    """Reproduce the exact risk_on pipeline (hybrid_macro_weight=0.0).

    Pipeline:
      raw_momentum = spy[i] / spy[i-lookback] - 1   (use_momentum=True)
      min_history  = max(lookback, 12)
      z_score      = expanding zscore of raw_momentum (at least min_history obs)
      risk_on      = sigmoid(z_score * 0.25)
    """
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


def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _difficult_sharpe(df: pd.DataFrame) -> float:
    segs = df[df["segment"] != "OVERALL"].copy()
    difficult = {2020, 2022}

    def _in_diff(row):
        try:
            y0 = pd.Period(row["test_start"], freq="M").year
            y1 = pd.Period(row["test_end"], freq="M").year
            return any(y in difficult for y in range(y0, y1 + 1))
        except Exception:
            return False

    mask = segs.apply(_in_diff, axis=1)
    return segs.loc[mask, "Strategy_Sharpe"].mean() if mask.any() else float("nan")


def main():
    print("=" * 70)
    print("EXPERIMENT: 12M vs 24M Continuous Mapped Momentum (fast-mode)")
    print(f"Window: {FAST_START} to {FAST_END}")
    print("Single variable: market_lookback_months 24 -> 12")
    print("=" * 70)

    # -- Pre-run: load prices for diagnostics ---------------------------------
    prices = fetch_prices(start=FAST_START, end=FAST_END)
    spy_monthly = prices["SPY"].resample("ME").last()

    ro_24, raw_24, z_24 = _compute_risk_on(spy_monthly, 24)
    ro_12, raw_12, z_12 = _compute_risk_on(spy_monthly, 12)

    # First non-zero z-score date (i.e., first month with >= min_history obs)
    def _first_valid(ro: pd.Series) -> str:
        v = ro[ro != sigmoid(0.0 * 0.25)]  # sigmoid(0) = 0.5 when z=0
        return v.index[0].strftime("%Y-%m") if len(v) else "none"

    print("\nPRE-RUN VERIFICATION")
    print("-" * 40)
    print(f"  First valid risk_on date (24M): {_first_valid(ro_24)}")
    print(f"  First valid risk_on date (12M): {_first_valid(ro_12)}")
    print(f"  Parameter diff: ONLY market_lookback_months (24 vs 12)")

    # -- Run baseline ---------------------------------------------------------
    print("\nRunning BASELINE (market_lookback_months=24)...")
    df_base = run_walk_forward_evaluation(**SHARED_KWARGS, market_lookback_months=24)

    # -- Run experiment -------------------------------------------------------
    print("Running EXPERIMENT (market_lookback_months=12)...")
    df_exp = run_walk_forward_evaluation(**SHARED_KWARGS, market_lookback_months=12)

    if df_base.empty or df_exp.empty:
        print("ERROR: one or both runs returned empty results.")
        sys.exit(1)

    # Verify identical OOS segments
    segs_b = df_base[df_base["segment"] != "OVERALL"][["test_start", "test_end"]].reset_index(drop=True)
    segs_e = df_exp[df_exp["segment"] != "OVERALL"][["test_start", "test_end"]].reset_index(drop=True)
    seg_match = segs_b.equals(segs_e)
    print(f"\nOOS segments identical: {'YES' if seg_match else 'NO -- STOP'}")
    if not seg_match:
        print("Segment mismatch - cannot compare. Exiting.")
        sys.exit(1)
    print(f"OOS segment count: {len(segs_b)}")

    b = _overall(df_base)
    e = _overall(df_exp)

    cagr_b   = b["Strategy_CAGR"]
    cagr_e   = e["Strategy_CAGR"]
    sharpe_b = b["Strategy_Sharpe"]
    sharpe_e = e["Strategy_Sharpe"]
    maxdd_b  = b["Strategy_MaxDD"]
    maxdd_e  = e["Strategy_MaxDD"]
    vol_b    = b["Strategy_Vol"]
    vol_e    = e["Strategy_Vol"]
    to_col   = "Strategy_Turnover"
    has_to   = (to_col in b.index and not np.isnan(b[to_col]))
    to_b     = b[to_col] if has_to else float("nan")
    to_e     = e[to_col] if has_to else float("nan")

    cagr_d   = cagr_e   - cagr_b
    sharpe_d = sharpe_e - sharpe_b
    maxdd_d  = maxdd_e  - maxdd_b
    vol_d    = vol_e    - vol_b
    to_d     = to_e - to_b if has_to else float("nan")
    to_ratio = to_e / to_b if (has_to and to_b > 0) else float("nan")

    # =========================================================================
    print("\n" + "=" * 70)
    print("1. METRICS vs BASELINE")
    print("=" * 70)
    print(f"{'':30} {'Baseline (24M)':>14} {'Exp (12M)':>12} {'Delta':>10}")
    print("-" * 68)
    print(f"{'CAGR':30} {cagr_b:>14.2%} {cagr_e:>12.2%} {cagr_d:>+10.2%}")
    print(f"{'Sharpe':30} {sharpe_b:>14.3f} {sharpe_e:>12.3f} {sharpe_d:>+10.3f}")
    print(f"{'MaxDD':30} {maxdd_b:>14.2%} {maxdd_e:>12.2%} {maxdd_d:>+10.2%}")
    print(f"{'Vol':30} {vol_b:>14.2%} {vol_e:>12.2%} {vol_d:>+10.2%}")
    if has_to:
        print(f"{'Turnover':30} {to_b:>14.2%} {to_e:>12.2%} {to_d:>+10.2%}")
    else:
        print(f"{'Turnover':30} {'n/a':>14} {'n/a':>12} {'n/a':>10}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("2. KILL SWITCH EVALUATION")
    print("=" * 70)
    ks_sharpe = sharpe_d < 0.02
    ks_cagr   = cagr_d < 0.0025

    ds_b = _difficult_sharpe(df_base)
    ds_e = _difficult_sharpe(df_exp)
    difficult_improved = (not np.isnan(ds_e) and not np.isnan(ds_b) and ds_e > ds_b)
    ks_no_diff = not difficult_improved
    kill = ks_sharpe and ks_cagr and ks_no_diff

    print(f"  Sharpe delta < +0.02?         {'YES' if ks_sharpe else 'NO ':3}  (delta = {sharpe_d:+.3f})")
    print(f"  CAGR delta < +0.25%?          {'YES' if ks_cagr else 'NO ':3}  (delta = {cagr_d:+.2%})")
    diff_b_str = f"{ds_b:.3f}" if not np.isnan(ds_b) else "n/a"
    diff_e_str = f"{ds_e:.3f}" if not np.isnan(ds_e) else "n/a"
    print(f"  No difficult-period impr.?    {'YES' if ks_no_diff else 'NO ':3}  "
          f"(baseline = {diff_b_str}, exp = {diff_e_str})")
    print(f"  Kill switch fires?            {'YES -> REJECT' if kill else 'NO -> continue'}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ANNUAL RAW MOMENTUM ZERO-CROSSING COUNTS")
    print("=" * 70)

    def _zero_crossings(raw: pd.Series) -> dict:
        clean = raw.dropna()
        sign  = np.sign(clean)
        cross = (sign != sign.shift(1)).astype(int)
        cross.iloc[0] = 0
        return cross.groupby(cross.index.year).sum().to_dict()

    cx_24 = _zero_crossings(raw_24)
    cx_12 = _zero_crossings(raw_12)
    all_yrs = sorted(set(cx_24) | set(cx_12))
    print(f"  {'Year':6} {'24M crossings':>15} {'12M crossings':>15} {'Delta':>8}")
    print("  " + "-" * 46)
    tot_24 = tot_12 = 0
    for y in all_yrs:
        c24 = cx_24.get(y, 0)
        c12 = cx_12.get(y, 0)
        tot_24 += c24
        tot_12 += c12
        print(f"  {y:6} {c24:>15} {c12:>15} {c12 - c24:>+8}")
    print("  " + "-" * 46)
    print(f"  {'Total':6} {tot_24:>15} {tot_12:>15} {tot_12 - tot_24:>+8}")
    if tot_24 > 0 and tot_12 > 1.5 * tot_24:
        print(f"\n  HIGH CHURN FLAG: 12M crossings = {tot_12 / tot_24:.1f}x baseline")
        print("  Escalation threshold raised to Sharpe delta >= +0.04")

    # =========================================================================
    print("\n" + "=" * 70)
    print("4. RISK_ON DISTRIBUTION SUMMARY (monthly, full diagnostic window)")
    print("=" * 70)
    r24c = ro_24.dropna()
    r12c = ro_12.dropna()
    print(f"  {'Metric':35} {'Baseline (24M)':>14} {'Exp (12M)':>14}")
    print("  " + "-" * 65)
    print(f"  {'Mean risk_on':35} {r24c.mean():>14.3f} {r12c.mean():>14.3f}")
    print(f"  {'Std risk_on':35} {r24c.std():>14.3f} {r12c.std():>14.3f}")
    print(f"  {'Min risk_on':35} {r24c.min():>14.3f} {r12c.min():>14.3f}")
    print(f"  {'Max risk_on':35} {r24c.max():>14.3f} {r12c.max():>14.3f}")
    print(f"  {'% months risk_on < 0.40':35} {(r24c < 0.40).mean():>14.1%} {(r12c < 0.40).mean():>14.1%}")
    print(f"  {'% months risk_on > 0.60':35} {(r24c > 0.60).mean():>14.1%} {(r12c > 0.60).mean():>14.1%}")
    print(f"  {'% months risk_on in [0.45,0.55]':35} "
          f"{((r24c >= 0.45) & (r24c <= 0.55)).mean():>14.1%} "
          f"{((r12c >= 0.45) & (r12c <= 0.55)).mean():>14.1%}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("5. CRASH-PERIOD SIGNAL COMPARISON (month-end risk_on)")
    print("=" * 70)

    crash_blocks = {
        "2020 COVID drawdown (peak Feb 19, trough Mar 23)": [
            "2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05",
        ],
        "2022 rate shock (SPY -20% Jan-Oct)": [
            "2021-12", "2022-01", "2022-03", "2022-06", "2022-09", "2022-10", "2022-12",
        ],
        "2023 Oct-Nov signal event": [
            "2023-09", "2023-10", "2023-11", "2023-12", "2024-01",
        ],
    }

    def _get(series: pd.Series, month_str: str) -> float:
        ts = pd.Timestamp(month_str + "-01") + pd.offsets.MonthEnd(0)
        return series.get(ts, float("nan"))

    for period, months in crash_blocks.items():
        print(f"\n  {period}")
        print(f"  {'Month':10} {'24M risk_on':>12} {'12M risk_on':>12} {'Delta':>10}")
        print("  " + "-" * 48)
        for m in months:
            r24v = _get(ro_24, m)
            r12v = _get(ro_12, m)
            dv   = r12v - r24v if not (np.isnan(r24v) or np.isnan(r12v)) else float("nan")
            r24s = f"{r24v:.3f}" if not np.isnan(r24v) else "n/a"
            r12s = f"{r12v:.3f}" if not np.isnan(r12v) else "n/a"
            ds   = f"{dv:+.3f}" if not np.isnan(dv) else "n/a"
            print(f"  {m:10} {r24s:>12} {r12s:>12} {ds:>10}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("6. DIVERGENCE PERIODS (|risk_on_12M - risk_on_24M| > 0.05)")
    print("=" * 70)
    common = ro_24.index.intersection(ro_12.index)
    diff   = (ro_12.reindex(common) - ro_24.reindex(common)).abs()
    diverg = diff[diff > 0.05]

    if len(diverg) > 0:
        spy_ret = spy_monthly.pct_change()
        print(f"  {'Month':10} {'24M risk_on':>12} {'12M risk_on':>12} {'|Delta|':>10} {'SPY ret':>10}")
        print("  " + "-" * 58)
        for dt, dv in diverg.items():
            r24v = ro_24.loc[dt]
            r12v = ro_12.loc[dt]
            sr   = spy_ret.get(dt, float("nan"))
            srs  = f"{sr:+.1%}" if not np.isnan(sr) else "n/a"
            print(f"  {dt.strftime('%Y-%m'):10} {r24v:>12.3f} {r12v:>12.3f} {dv:>10.3f} {srs:>10}")
        print(f"\n  Total divergent months: {len(diverg)}")
        print(f"  Max divergence:         {diff.max():.3f} ({diff.idxmax().strftime('%Y-%m')})")
    else:
        print("  No months with |delta| > 0.05. Models tracked closely.")

    # =========================================================================
    print("\n" + "=" * 70)
    print("7. TURNOVER COMPARISON")
    print("=" * 70)
    if has_to:
        print(f"  Baseline turnover:   {to_b:.2%}")
        print(f"  Experiment turnover: {to_e:.2%}")
        print(f"  Delta:               {to_d:+.2%}")
        print(f"  Ratio:               {to_ratio:.2f}x")
        if to_ratio > 1.5:
            print("  HIGH CHURN: turnover > 1.5x baseline.")
            print("  Escalation threshold raised to Sharpe delta >= +0.04.")
    else:
        print("  Turnover not available in walk-forward output.")

    # =========================================================================
    print("\n" + "=" * 70)
    print("8. ESCALATION DECISION")
    print("=" * 70)
    high_churn = (not np.isnan(to_ratio) and to_ratio > 1.5)
    also_high_churn = (tot_24 > 0 and tot_12 > 1.5 * tot_24)
    sharpe_thresh = 0.04 if (high_churn or also_high_churn) else 0.02

    esc_sharpe = sharpe_d >= sharpe_thresh
    esc_cagr   = cagr_d >= 0.0025
    esc_maxdd  = maxdd_d > 0.01
    esc_diff   = difficult_improved
    escalate   = esc_sharpe or esc_cagr or esc_maxdd or esc_diff

    thr_note = "(raised: high churn)" if sharpe_thresh == 0.04 else ""
    print(f"  High churn flag active?          {'YES' if (high_churn or also_high_churn) else 'NO'}")
    print(f"  Sharpe threshold:                {sharpe_thresh:+.2f} {thr_note}")
    print(f"  Sharpe delta >= threshold?       {'YES' if esc_sharpe else 'NO '}  ({sharpe_d:+.3f})")
    print(f"  CAGR improves >= +0.25%?         {'YES' if esc_cagr else 'NO '}  ({cagr_d:+.2%})")
    print(f"  MaxDD improves > +1%?            {'YES' if esc_maxdd else 'NO '}  ({maxdd_d:+.2%})")
    print(f"  Difficult-period improvement?    {'YES' if esc_diff else 'NO '}")
    print()

    if kill:
        verdict = "FAST-MODE VERDICT: REJECTED (kill switch fired)"
    elif escalate:
        verdict = "FAST-MODE VERDICT: PASS -- escalate to full walk-forward validation"
    else:
        verdict = "FAST-MODE VERDICT: INSUFFICIENT -- does not meet escalation thresholds"

    print(f"  {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
