"""Experiment: Inverse-vol smoothing — VOL_LOOKBACK 63 -> 126.

Single variable change: VOL_LOOKBACK patched in-memory between runs.
No file changes. Both runs use identical all other parameters.
Fast-mode window: 2018-01-01 to 2024-12-31.

Engine semantics reminder:
  weights at each rebalance = blend(risk_on) then vol_scaled_weights_from_std
  vol_scaled_weights_from_std divides each asset by its rolling std then renormalizes
  rolling std window = VOL_LOOKBACK (63 baseline)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import src.backtest.engine as _eng
import src.config as _cfg
from src.allocation.vol_scaling import vol_scaled_weights_from_std
from src.backtest.engine import _blend_alloc, run_backtest_with_allocations
from src.config import (
    ASSETS,
    RISK_OFF_ASSETS_BASE,
    RISK_ON_ASSETS_BASE,
    TICKERS,
)
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation
from src.features.transforms import sigmoid

logging.basicConfig(level=logging.WARNING)

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"
LOOKBACK_MONTHS = 24
VOL_BASE = 63
VOL_EXP = 126

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
    market_lookback_months=LOOKBACK_MONTHS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segments_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["segment"] != "OVERALL"].copy()


def _m(row, col: str) -> float:
    v = row.get(col, float("nan"))
    return float(v) if v is not None else float("nan")


def _fmt_pct(v: float, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    return f"{v:+.2%}" if sign else f"{v:.2%}"


def _fmt_f(v: float, d: int = 3, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    return fmt.format(v)


def _difficult_sharpe(df: pd.DataFrame) -> float:
    segs = _segments_only(df)
    difficult = {2020, 2022}
    def _in(row):
        try:
            y0 = pd.Period(row["test_start"], freq="M").year
            y1 = pd.Period(row["test_end"], freq="M").year
            return any(y in difficult for y in range(y0, y1 + 1))
        except Exception:
            return False
    mask = segs.apply(_in, axis=1)
    return segs.loc[mask, "Strategy_Sharpe"].mean() if mask.any() else float("nan")


def _crash_segs(df: pd.DataFrame, year: int):
    segs = _segments_only(df)
    def _covers(row):
        try:
            y0 = pd.Period(row["test_start"], freq="M").year
            y1 = pd.Period(row["test_end"], freq="M").year
            return y0 <= year <= y1
        except Exception:
            return False
    mask = segs.apply(_covers, axis=1)
    return segs[mask]


def _compute_risk_on_me(spy_monthly: pd.Series, lookback: int) -> pd.Series:
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
    return sigmoid(z * 0.25)


def _weight_stability(prices: pd.DataFrame, vol_lookback: int,
                      risk_on_me: pd.Series) -> dict:
    """Compute per-asset monthly weight std under a given VOL_LOOKBACK.
    Reconstructs the exact target weights the engine would use.
    """
    returns_daily = prices[TICKERS].pct_change().iloc[1:]
    rolling_std = returns_daily.rolling(vol_lookback, min_periods=1).std()

    dates = prices.index
    months = pd.Series(dates).dt.to_period("M").values
    month_changed_mask = np.concatenate([[True], months[1:] != months[:-1]])
    rebalance_dates = list(dates[month_changed_mask])

    w_risk_on = {a: (1.0 / len(RISK_ON_ASSETS_BASE) if a in RISK_ON_ASSETS_BASE else 0.0)
                 for a in ASSETS}
    w_risk_off = {a: (1.0 / len(RISK_OFF_ASSETS_BASE) if a in RISK_OFF_ASSETS_BASE else 0.0)
                  for a in ASSETS}

    records = []
    for d in rebalance_dates:
        prior_ends = risk_on_me.index[risk_on_me.index < d]
        ro = float(risk_on_me.loc[prior_ends[-1]]) if len(prior_ends) > 0 else 0.5

        if d in rolling_std.index:
            row = rolling_std.loc[d]
        else:
            idx = rolling_std.index.asof(d)
            row = rolling_std.iloc[0] if pd.isna(idx) else rolling_std.loc[idx]
        std_d = {a: float(row[a]) if a in row.index and pd.notna(row[a]) else None
                 for a in TICKERS}

        blended = _blend_alloc(w_risk_off, w_risk_on, ro, ASSETS)
        final = vol_scaled_weights_from_std(blended, std_d, list(TICKERS))
        rec = {"date": d}
        for a in TICKERS:
            rec[a] = final.get(a, 0.0)
        records.append(rec)

    wdf = pd.DataFrame(records).set_index("date")
    result = {}
    risk_off_assets = ["IEF", "TLT", "GLD"]
    for a in risk_off_assets + ["SPY", "MTUM"]:
        if a in wdf.columns:
            monthly_std = wdf[a].std()
            monthly_mean = wdf[a].mean()
            monthly_range = wdf[a].max() - wdf[a].min()
            result[a] = {"mean": monthly_mean, "std": monthly_std, "range": monthly_range}
    return result


def main():
    print("=" * 72)
    print("EXPERIMENT: VOL_LOOKBACK 63 -> 126 (inverse-vol smoothing)")
    print(f"Fast-mode window: {FAST_START} to {FAST_END}")
    print("Single variable: VOL_LOOKBACK only (in-memory patch)")
    print("=" * 72)

    # -- Load prices for pre-run weight stability diagnostics ----------------
    prices = fetch_prices(start=FAST_START, end=FAST_END)
    spy_monthly = prices["SPY"].resample("ME").last()
    risk_on_me = _compute_risk_on_me(spy_monthly, LOOKBACK_MONTHS)

    # -- PRE-RUN VERIFICATION ------------------------------------------------
    print("\nPRE-RUN VERIFICATION")
    print("-" * 44)
    print(f"  Config VOL_LOOKBACK before runs: {_cfg.VOL_LOOKBACK}  [must be {VOL_BASE}]")
    print(f"  Engine VOL_LOOKBACK before runs: {_eng.VOL_LOOKBACK}  [must be {VOL_BASE}]")
    if _cfg.VOL_LOOKBACK != VOL_BASE or _eng.VOL_LOOKBACK != VOL_BASE:
        print("  STOP: VOL_LOOKBACK is not at baseline value. Aborting.")
        sys.exit(1)

    # List every parameter that will differ (must be exactly one)
    print(f"  Parameter diff: ONLY VOL_LOOKBACK ({VOL_BASE} -> {VOL_EXP})")
    print(f"  market_lookback_months: {LOOKBACK_MONTHS} in both runs")
    print(f"  signal pipeline: 24M SPY momentum -> expanding z-score -> sigmoid(z*0.25)")
    print(f"  rebalance: monthly, first trading day of each month, both runs")
    print(f"  blend: risk_on * inv_vol_scaled_risk_on + (1-risk_on) * inv_vol_scaled_risk_off")
    print(f"  cost model: |w_target(t) - w_target(t-1)| * COST_BPS, both runs")

    # -- BASELINE RUN --------------------------------------------------------
    print(f"\nRunning BASELINE (VOL_LOOKBACK={VOL_BASE})...")
    assert _eng.VOL_LOOKBACK == VOL_BASE
    df_base = run_walk_forward_evaluation(**SHARED_KWARGS)

    # -- PATCH ---------------------------------------------------------------
    _cfg.VOL_LOOKBACK = VOL_EXP
    _eng.VOL_LOOKBACK = VOL_EXP
    print(f"\nPatch applied: VOL_LOOKBACK -> {_eng.VOL_LOOKBACK}  [verified: {_eng.VOL_LOOKBACK}]")

    # -- EXPERIMENT RUN ------------------------------------------------------
    print(f"Running EXPERIMENT (VOL_LOOKBACK={VOL_EXP})...")
    assert _eng.VOL_LOOKBACK == VOL_EXP
    df_exp = run_walk_forward_evaluation(**SHARED_KWARGS)

    # -- RESTORE -------------------------------------------------------------
    _cfg.VOL_LOOKBACK = VOL_BASE
    _eng.VOL_LOOKBACK = VOL_BASE
    print(f"Restored: VOL_LOOKBACK -> {_eng.VOL_LOOKBACK}")

    if df_base.empty or df_exp.empty:
        print("ERROR: one or both runs returned empty results.")
        sys.exit(1)

    # Verify identical OOS segments
    segs_b = _segments_only(df_base)[["test_start", "test_end"]].reset_index(drop=True)
    segs_e = _segments_only(df_exp)[["test_start", "test_end"]].reset_index(drop=True)
    seg_match = segs_b.equals(segs_e)
    print(f"\nOOS segments identical: {'YES' if seg_match else 'NO -- STOP'}")
    if not seg_match:
        print("Segment mismatch. Cannot compare. Exiting.")
        sys.exit(1)
    print(f"OOS segment count: {len(segs_b)}")

    ob = _overall(df_base)
    oe = _overall(df_exp)

    cagr_b  = _m(ob, "Strategy_CAGR")
    cagr_e  = _m(oe, "Strategy_CAGR")
    shr_b   = _m(ob, "Strategy_Sharpe")
    shr_e   = _m(oe, "Strategy_Sharpe")
    mdd_b   = _m(ob, "Strategy_MaxDD")
    mdd_e   = _m(oe, "Strategy_MaxDD")
    vol_b   = _m(ob, "Strategy_Vol")
    vol_e   = _m(oe, "Strategy_Vol")
    to_b    = _m(ob, "Strategy_Turnover")
    to_e    = _m(oe, "Strategy_Turnover")
    has_to  = not (np.isnan(to_b) or np.isnan(to_e))

    cagr_d = cagr_e - cagr_b
    shr_d  = shr_e  - shr_b
    mdd_d  = mdd_e  - mdd_b
    vol_d  = vol_e  - vol_b
    to_d   = to_e - to_b if has_to else float("nan")
    to_ratio = to_e / to_b if (has_to and to_b > 0) else float("nan")

    # =========================================================================
    print("\n" + "=" * 72)
    print("2. METRICS vs BASELINE")
    print("=" * 72)
    print(f"  {'Metric':30} {'Baseline (63d)':>14} {'Exp (126d)':>12} {'Delta':>10}")
    print("  " + "-" * 68)
    print(f"  {'CAGR':30} {_fmt_pct(cagr_b):>14} {_fmt_pct(cagr_e):>12} {_fmt_pct(cagr_d, sign=True):>10}")
    print(f"  {'Sharpe':30} {_fmt_f(shr_b):>14} {_fmt_f(shr_e):>12} {_fmt_f(shr_d, sign=True):>10}")
    print(f"  {'MaxDD':30} {_fmt_pct(mdd_b):>14} {_fmt_pct(mdd_e):>12} {_fmt_pct(mdd_d, sign=True):>10}")
    print(f"  {'Vol':30} {_fmt_pct(vol_b):>14} {_fmt_pct(vol_e):>12} {_fmt_pct(vol_d, sign=True):>10}")
    if has_to:
        print(f"  {'Turnover':30} {_fmt_pct(to_b):>14} {_fmt_pct(to_e):>12} {_fmt_pct(to_d, sign=True):>10}")
    else:
        print(f"  {'Turnover':30} {'n/a':>14} {'n/a':>12} {'n/a':>10}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("3. KILL SWITCH EVALUATION")
    print("=" * 72)
    ks_sharpe = shr_d < 0.02
    ks_cagr   = cagr_d < 0.0025
    ds_b = _difficult_sharpe(df_base)
    ds_e = _difficult_sharpe(df_exp)
    diff_improved = (not np.isnan(ds_e) and not np.isnan(ds_b) and ds_e > ds_b)
    ks_no_diff = not diff_improved
    kill = ks_sharpe and ks_cagr and ks_no_diff

    print(f"  Sharpe delta < +0.02?             {'YES' if ks_sharpe else 'NO ':3}  ({shr_d:+.3f})")
    print(f"  CAGR delta < +0.25%?              {'YES' if ks_cagr else 'NO ':3}  ({cagr_d:+.2%})")
    ds_b_s = _fmt_f(ds_b) if not np.isnan(ds_b) else "n/a"
    ds_e_s = _fmt_f(ds_e) if not np.isnan(ds_e) else "n/a"
    print(f"  No difficult-period improvement?  {'YES' if ks_no_diff else 'NO ':3}  (baseline={ds_b_s}, exp={ds_e_s})")
    print(f"  Kill switch fires?                {'YES -> REJECT' if kill else 'NO -> continue'}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("4. TURNOVER ANALYSIS")
    print("=" * 72)
    if has_to:
        print(f"  Baseline turnover (63d):   {to_b:.2%}")
        print(f"  Experiment turnover (126d): {to_e:.2%}")
        print(f"  Absolute delta:             {to_d:+.2%}")
        print(f"  Ratio:                      {to_ratio:.2f}x")
        print()
        # Attribution context
        attr_vol_component = 0.9046  # from attribution diagnostic
        attr_total = 1.1325
        expected_reduction_lo = to_b * (attr_vol_component / attr_total) * 0.25  # 25% of vol component
        expected_reduction_hi = to_b * (attr_vol_component / attr_total) * 0.60  # 60% of vol component
        print(f"  Attribution context:")
        print(f"    Vol-rebalancing component (63d baseline, full history): 90.46% of 113.25% total")
        print(f"    Expected TO reduction range (25-60% of vol component): "
              f"{_fmt_pct(-expected_reduction_lo, sign=True)} to {_fmt_pct(-expected_reduction_hi, sign=True)}")
        if has_to and not np.isnan(to_d):
            if to_d < -expected_reduction_lo:
                print(f"    Result: reduction of {_fmt_pct(-to_d)} is WITHIN expected range -> vol smoothing effective")
            elif to_d > 0:
                print(f"    Result: turnover INCREASED -> unexpected. Smoothing may be disrupting netting.")
            else:
                print(f"    Result: reduction of {_fmt_pct(-to_d)} is BELOW expected range -> smoothing insufficient at 126d")
        if to_ratio > 1.5:
            print(f"    HIGH CHURN FLAG: experiment TO > 1.5x baseline. Escalation threshold = +0.04 Sharpe.")
    else:
        print("  Turnover not available in walk-forward output.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("5. RISK-OFF SLEEVE STABILITY CHECK")
    print("=" * 72)
    print("  Computing per-asset weight stability from target weight reconstruction...")
    stab_63  = _weight_stability(prices, VOL_BASE, risk_on_me)
    stab_126 = _weight_stability(prices, VOL_EXP, risk_on_me)

    risk_off_check = ["IEF", "GLD", "TLT"]
    risk_on_check  = ["SPY", "MTUM"]
    print(f"\n  {'Asset':6} {'Type':8} {'Mean(63d)':>10} {'Std(63d)':>10} {'Range(63d)':>11} "
          f"{'Mean(126d)':>11} {'Std(126d)':>10} {'Range(126d)':>12} {'Std reduced?':>13}")
    print("  " + "-" * 96)
    for a in risk_off_check + risk_on_check:
        atype = "risk-off" if a in RISK_OFF_ASSETS_BASE else "risk-on"
        s63  = stab_63.get(a,  {"mean": float("nan"), "std": float("nan"), "range": float("nan")})
        s126 = stab_126.get(a, {"mean": float("nan"), "std": float("nan"), "range": float("nan")})
        reduced = "YES" if s126["std"] < s63["std"] else "NO"
        print(f"  {a:6} {atype:8} {s63['mean']:>10.4f} {s63['std']:>10.4f} {s63['range']:>11.4f} "
              f"{s126['mean']:>11.4f} {s126['std']:>10.4f} {s126['range']:>12.4f} {reduced:>13}")

    n_reduced = sum(
        1 for a in risk_off_check
        if a in stab_63 and a in stab_126 and stab_126[a]["std"] < stab_63[a]["std"]
    )
    print(f"\n  Risk-off weight std reduced in {n_reduced}/{len(risk_off_check)} assets")
    if n_reduced == len(risk_off_check):
        print("  CONFIRMED: 126d vol smoothing reduces weight churn in all risk-off assets")
    elif n_reduced == 0:
        print("  WARNING: no weight std reduction in risk-off sleeve -- smoothing not effective here")
    else:
        print(f"  PARTIAL: {n_reduced}/{len(risk_off_check)} risk-off assets show reduced weight churn")

    # =========================================================================
    print("\n" + "=" * 72)
    print("6. CRASH-PERIOD BEHAVIOR")
    print("=" * 72)

    for year, label in [(2020, "2020 COVID"), (2022, "2022 rate shock")]:
        sb = _crash_segs(df_base, year)
        se = _crash_segs(df_exp, year)
        print(f"\n  {label}:")
        if len(sb) == 0:
            print(f"    No OOS segments covering {year} -- not evaluable in fast-mode")
            print(f"    (min_train_months=60 from 2018-01-01 -> first OOS ~2023)")
            continue
        b_shr = sb["Strategy_Sharpe"].mean()
        e_shr = se["Strategy_Sharpe"].mean()
        b_mdd = sb["Strategy_MaxDD"].mean()
        e_mdd = se["Strategy_MaxDD"].mean()
        b_to  = sb["Strategy_Turnover"].mean() if "Strategy_Turnover" in sb else float("nan")
        e_to  = se["Strategy_Turnover"].mean() if "Strategy_Turnover" in se else float("nan")
        print(f"    Segments covering {year}: {len(sb)}")
        print(f"    Sharpe:   baseline={_fmt_f(b_shr)}  exp={_fmt_f(e_shr)}  delta={_fmt_f(e_shr-b_shr, sign=True)}")
        print(f"    MaxDD:    baseline={_fmt_pct(b_mdd)}  exp={_fmt_pct(e_mdd)}  delta={_fmt_pct(e_mdd-b_mdd, sign=True)}")
        if not np.isnan(b_to):
            print(f"    Turnover: baseline={_fmt_pct(b_to)}  exp={_fmt_pct(e_to)}  delta={_fmt_pct(e_to-b_to, sign=True)}")
        if e_mdd < b_mdd - 0.01:
            print(f"    FLAG: experiment worsens MaxDD by >1% in {year} -> reject signal regardless of overall Sharpe")
        elif e_shr > b_shr:
            print(f"    BETTER: experiment improves Sharpe in {year}")
        elif abs(e_shr - b_shr) < 0.02:
            print(f"    NEUTRAL: Sharpe difference < 0.02 in {year}")
        else:
            print(f"    WORSE: experiment has lower Sharpe in {year}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("7. ESCALATION DECISION")
    print("=" * 72)
    high_churn = (not np.isnan(to_ratio) and to_ratio > 1.5)
    shr_thresh = 0.04 if high_churn else 0.02

    esc_shr  = shr_d >= shr_thresh
    esc_cagr = cagr_d >= 0.0025
    esc_mdd  = mdd_d > 0.01
    esc_diff = diff_improved
    escalate = esc_shr or esc_cagr or esc_mdd or esc_diff

    print(f"  High churn flag:                   {'YES (threshold +0.04)' if high_churn else 'NO  (threshold +0.02)'}")
    print(f"  Sharpe delta >= {shr_thresh:.2f}?              {'YES' if esc_shr else 'NO '}  ({shr_d:+.3f})")
    print(f"  CAGR delta >= +0.25%?              {'YES' if esc_cagr else 'NO '}  ({cagr_d:+.2%})")
    print(f"  MaxDD improves > +1.0%?            {'YES' if esc_mdd else 'NO '}  ({mdd_d:+.2%})")
    print(f"  Difficult-period improvement?      {'YES' if esc_diff else 'NO '}")
    print()

    # Special rejection: TO increased (unexpected for smoothing experiment)
    to_increased = has_to and to_d > 0.005
    if to_increased:
        print(f"  SPECIAL FLAG: turnover INCREASED (+{to_d:.2%}). Vol smoothing failed to reduce churn.")
        print(f"  This overrides any Sharpe improvement -- the structural hypothesis is rejected.")

    if kill:
        verdict = "FAST-MODE VERDICT: REJECTED (kill switch)"
    elif to_increased and not esc_shr:
        verdict = "FAST-MODE VERDICT: REJECTED (TO increased, no Sharpe improvement)"
    elif escalate:
        verdict = "FAST-MODE VERDICT: PASS -- escalate to full walk-forward validation"
    else:
        verdict = "FAST-MODE VERDICT: INSUFFICIENT -- does not meet escalation thresholds"

    print(f"  {verdict}")
    print()

    # Failure interpretation
    if kill or (not escalate):
        print("  FAILURE INTERPRETATION:")
        if has_to and to_d > -0.01:
            print("  - Turnover did not materially decrease.")
            print("    VOL_LOOKBACK=126 is still short enough that monthly vol estimates")
            print("    change substantially. Consider 189d or quarterly vol recalibration.")
        elif has_to and to_d < -0.02 and shr_d < 0:
            print("  - Turnover decreased but Sharpe worsened.")
            print("    The inverse-vol scaling with 63d window is providing useful risk management.")
            print("    Slower weights hold excess risk-off allocation longer in stress periods,")
            print("    or miss beneficial risk-on shifts post-recovery.")
            print("    Do not suppress vol scaling to reduce turnover.")
        elif has_to and to_d < -0.02 and abs(shr_d) < 0.02:
            print("  - Turnover decreased, performance roughly neutral.")
            print("    This is a CONDITIONAL ACCEPT: the reduction is real but the fast-mode")
            print("    window may not cover 2020/2022 sufficiently.")
            print("    Run full walk-forward before accepting.")

    print("=" * 72)


if __name__ == "__main__":
    main()
