"""Experiment: Tolerance-Band Execution Filter (tau = 1.5%).

Single variable change: tolerance 0.0 -> 0.015.
Implementation: trade-vector filtering.
  trade = new_w - prev_weights
  exec_trade = trade * (abs(trade) > tau)
  w_exec = prev_weights + exec_trade
  w_exec = w_exec / w_exec.sum()   [renormalize]

No patching required -- tolerance is a new named parameter in engine + walk_forward.
Fast-mode window: 2018-01-01 to 2024-12-31.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import VOL_LOOKBACK
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.WARNING)

FAST_START = "2018-01-01"
FAST_END   = "2024-12-31"
TAU        = 0.015

# Reference baseline metrics from prior fast-mode runs (VOL_LOOKBACK experiment baseline)
PRIOR_CAGR   = 0.1456
PRIOR_SHARPE = 1.163
PRIOR_TO     = 0.7193

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
    market_lookback_months=24,
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


def _pct(v: float, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    return f"{v:+.2%}" if sign else f"{v:.2%}"


def _f(v: float, d: int = 3, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    return fmt.format(v)


def _difficult_sharpe(df: pd.DataFrame) -> float:
    segs = _segments_only(df)
    difficult = {2020, 2022}

    def _covers(row):
        try:
            y0 = pd.Period(row["test_start"], freq="M").year
            y1 = pd.Period(row["test_end"], freq="M").year
            return any(y in difficult for y in range(y0, y1 + 1))
        except Exception:
            return False

    mask = segs.apply(_covers, axis=1)
    return float(segs.loc[mask, "Strategy_Sharpe"].mean()) if mask.any() else float("nan")


def _crash_sharpe_mdd(df: pd.DataFrame, year: int) -> tuple[float, float]:
    segs = _segments_only(df)

    def _covers(row):
        try:
            y0 = pd.Period(row["test_start"], freq="M").year
            y1 = pd.Period(row["test_end"], freq="M").year
            return y0 <= year <= y1
        except Exception:
            return False

    mask = segs.apply(_covers, axis=1)
    if not mask.any():
        return float("nan"), float("nan")
    sub = segs[mask]
    return float(sub["Strategy_Sharpe"].mean()), float(sub["Strategy_MaxDD"].mean())


def main():
    print("=" * 72)
    print("EXPERIMENT: Tolerance-Band Execution Filter  tau = 1.5%")
    print(f"Fast-mode window: {FAST_START} -> {FAST_END}")
    print(f"Single variable: tolerance 0.000 -> {TAU:.3f}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. PRE-RUN VERIFICATION
    # ------------------------------------------------------------------
    print("\n1. PRE-RUN VERIFICATION")
    print("-" * 44)
    print(f"   VOL_LOOKBACK (config):        {VOL_LOOKBACK}  [must be 63]")
    print(f"   market_lookback_months:       24  [both runs]")
    print(f"   tolerance (baseline):         0.000")
    print(f"   tolerance (experiment):       {TAU:.3f}")
    print(f"   Implementation: trade-vector filter, renorm after partial suppression")
    print(f"   Parameters changed vs prior experiments: tolerance only")
    if VOL_LOOKBACK != 63:
        print("   STOP: VOL_LOOKBACK != 63. Fix config.py before running.")
        sys.exit(1)
    print("   Pre-run checks: PASS")

    # ------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------
    print(f"\n   Running BASELINE  (tolerance=0.000) ...")
    df_base = run_walk_forward_evaluation(**SHARED_KWARGS, tolerance=0.0)

    # ------------------------------------------------------------------
    # EXPERIMENT
    # ------------------------------------------------------------------
    print(f"   Running EXPERIMENT (tolerance={TAU:.3f}) ...")
    df_exp = run_walk_forward_evaluation(**SHARED_KWARGS, tolerance=TAU)

    if df_base.empty or df_exp.empty:
        print("ERROR: empty results.")
        sys.exit(1)

    # Verify identical OOS segments
    segs_b = _segments_only(df_base)[["test_start", "test_end"]].reset_index(drop=True)
    segs_e = _segments_only(df_exp)[["test_start", "test_end"]].reset_index(drop=True)
    seg_match = segs_b.equals(segs_e)
    print(f"\n   OOS segments identical: {'YES' if seg_match else 'NO -- STOP'}")
    if not seg_match:
        print("   Segment mismatch. Cannot compare. Exiting.")
        sys.exit(1)
    print(f"   OOS segment count: {len(segs_b)}")

    ob = _overall(df_base)
    oe = _overall(df_exp)

    cagr_b = _m(ob, "Strategy_CAGR");   cagr_e = _m(oe, "Strategy_CAGR")
    shr_b  = _m(ob, "Strategy_Sharpe"); shr_e  = _m(oe, "Strategy_Sharpe")
    mdd_b  = _m(ob, "Strategy_MaxDD");  mdd_e  = _m(oe, "Strategy_MaxDD")
    vol_b  = _m(ob, "Strategy_Vol");    vol_e  = _m(oe, "Strategy_Vol")
    to_b   = _m(ob, "Strategy_Turnover"); to_e = _m(oe, "Strategy_Turnover")
    has_to = not (np.isnan(to_b) or np.isnan(to_e))

    cagr_d = cagr_e - cagr_b
    shr_d  = shr_e  - shr_b
    mdd_d  = mdd_e  - mdd_b
    vol_d  = vol_e  - vol_b
    to_d   = (to_e - to_b) if has_to else float("nan")
    to_r   = (to_e / to_b) if (has_to and to_b > 0) else float("nan")

    # ------------------------------------------------------------------
    # BASELINE REGRESSION CHECK
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BASELINE REGRESSION CHECK")
    print("=" * 72)
    print(f"   {'Metric':20} {'Prior':>10} {'Current':>10} {'|Delta|':>10} {'OK?':>6}")
    print("   " + "-" * 58)
    for label, prior, cur in [
        ("CAGR",    PRIOR_CAGR,   cagr_b),
        ("Sharpe",  PRIOR_SHARPE, shr_b),
        ("Turnover",PRIOR_TO,     to_b),
    ]:
        if np.isnan(cur):
            continue
        delta = abs(cur - prior)
        ok = delta < (0.002 if label == "CAGR" else 0.01)
        p_s = f"{prior:.4f}" if label in ("Sharpe","Turnover") else f"{prior:.2%}"
        c_s = f"{cur:.4f}"   if label in ("Sharpe","Turnover") else f"{cur:.2%}"
        d_s = f"{delta:.4f}" if label in ("Sharpe","Turnover") else f"{delta:.2%}"
        print(f"   {label:20} {p_s:>10} {c_s:>10} {d_s:>10} {'OK' if ok else 'DIFF':>6}")
    print()
    print("   Note: tolerance=0.0 must be identical to unmodified engine output.")
    print("   If delta > tolerance above, new tolerance parameter may be altering engine")
    print("   behavior even at zero -- STOP and investigate.")

    # ------------------------------------------------------------------
    # 2. METRICS vs BASELINE
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("2. METRICS vs BASELINE")
    print("=" * 72)
    print(f"   {'Metric':30} {'Baseline (tau=0)':>16} {'Exp (tau=1.5%)':>14} {'Delta':>10}")
    print("   " + "-" * 72)
    rows = [
        ("CAGR",    _pct(cagr_b), _pct(cagr_e), _pct(cagr_d, sign=True)),
        ("Sharpe",  _f(shr_b),    _f(shr_e),    _f(shr_d, sign=True)),
        ("MaxDD",   _pct(mdd_b),  _pct(mdd_e),  _pct(mdd_d, sign=True)),
        ("Vol",     _pct(vol_b),  _pct(vol_e),  _pct(vol_d, sign=True)),
        ("Turnover",_pct(to_b) if has_to else "n/a",
                    _pct(to_e) if has_to else "n/a",
                    _pct(to_d, sign=True) if has_to else "n/a"),
    ]
    for label, b, e, d in rows:
        print(f"   {label:30} {b:>16} {e:>14} {d:>10}")

    # ------------------------------------------------------------------
    # 3. KILL SWITCH EVALUATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("3. KILL SWITCH EVALUATION")
    print("=" * 72)
    ks_shr  = shr_d  < 0.02
    ks_cagr = cagr_d < 0.0025
    ds_b = _difficult_sharpe(df_base)
    ds_e = _difficult_sharpe(df_exp)
    diff_ok = (not np.isnan(ds_e) and not np.isnan(ds_b) and ds_e > ds_b)
    ks_diff = not diff_ok
    kill = ks_shr and ks_cagr and ks_diff

    ds_b_s = _f(ds_b) if not np.isnan(ds_b) else "n/a"
    ds_e_s = _f(ds_e) if not np.isnan(ds_e) else "n/a"
    print(f"   Sharpe delta < +0.02?             {'YES' if ks_shr else 'NO ':3}  ({shr_d:+.3f})")
    print(f"   CAGR delta < +0.25%?              {'YES' if ks_cagr else 'NO ':3}  ({cagr_d:+.2%})")
    print(f"   No difficult-period improvement?  {'YES' if ks_diff else 'NO ':3}  "
          f"(baseline={ds_b_s}, exp={ds_e_s})")
    print(f"   Kill switch fires?                {'YES -> REJECT' if kill else 'NO -> continue'}")

    # ------------------------------------------------------------------
    # 4. TURNOVER ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("4. TURNOVER ANALYSIS")
    print("=" * 72)
    if has_to:
        print(f"   Baseline turnover (tau=0.000):    {to_b:.2%}")
        print(f"   Experiment turnover (tau=0.015):  {to_e:.2%}")
        print(f"   Absolute delta:                  {to_d:+.2%}")
        print(f"   Ratio (exp / baseline):          {to_r:.3f}x")
        print()
        print(f"   Attribution context (full-history baseline, prior diagnostic):")
        print(f"     Vol-rebalancing component: 79.9% of total TO  (~90pp of 113pp)")
        print(f"     Signal-driven component:   43.5% of total TO  (~49pp of 113pp)")
        print(f"     tau=1.5% targets small vol-driven moves; large signal moves pass through")
        print()
        if to_d > 0.005:
            print(f"   FLAG: turnover INCREASED (+{to_d:.2%}).")
            print(f"     Renormalization after partial suppression can redistribute weight")
            print(f"     to other assets, generating additional trades. Trade-vector filter")
            print(f"     does not guarantee aggregate turnover reduction when some assets pass.")
        elif abs(to_d) < 0.01:
            print(f"   FLAG: turnover nearly unchanged (|delta|={abs(to_d):.2%}).")
            print(f"     Most vol-rebalancing trades exceed tau=1.5% per asset.")
            print(f"     tau=1.5% is too coarse to suppress the dominant churn source.")
        elif to_d < 0:
            pct_red = -to_d / to_b
            print(f"   Turnover reduced by {-to_d:.2%} ({pct_red:.1%} relative reduction).")
            vol_component = 0.799 * to_b
            target_pct = -to_d / vol_component
            print(f"   Captured {target_pct:.1%} of the vol-rebalancing component.")
    else:
        print("   Turnover column not in walk-forward output.")

    # ------------------------------------------------------------------
    # 5. SUPPRESSED TRADE DISTRIBUTION
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("5. SUPPRESSED TRADE DISTRIBUTION")
    print("=" * 72)
    print("   Note: walk-forward output contains aggregate turnover only.")
    print("   Per-asset and per-month suppression breakdown requires a dedicated")
    print("   diagnostic run with return_weights=True on the full history.")
    if has_to:
        effective_suppression = max(0.0, -to_d)
        total_vol_to = 0.799 * to_b
        if to_d <= 0:
            frac = effective_suppression / total_vol_to if total_vol_to > 0 else 0.0
            print(f"\n   Implied suppression (from TO delta):  {effective_suppression:.2%} annualized")
            print(f"   As fraction of vol-rebalancing component (79.9%): {frac:.1%}")
            print(f"   Expected: risk-off sleeve (IEF, GLD, TLT) accounts for most suppression")
            print(f"   based on attribution finding that risk-off drives dominant vol churn.")
        else:
            print(f"\n   TO increased: suppression was outweighed by renorm-driven trades.")
            print(f"   The renormalization step spread suppressed weight to passing assets,")
            print(f"   creating larger-than-expected trades in other positions.")

    # ------------------------------------------------------------------
    # 6. CRASH-PERIOD BEHAVIOR
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("6. CRASH-PERIOD BEHAVIOR")
    print("=" * 72)
    for year, label in [(2020, "2020 COVID"), (2022, "2022 rate shock")]:
        b_shr, b_mdd = _crash_sharpe_mdd(df_base, year)
        e_shr, e_mdd = _crash_sharpe_mdd(df_exp, year)
        print(f"\n   {label} ({year}):")
        if np.isnan(b_shr):
            print(f"     No OOS segments cover {year} in this fast-mode window.")
            print(f"     (min_train_months=60 from {FAST_START}: first OOS starts ~2023)")
            print(f"     Crash-period check deferred to full walk-forward validation.")
        else:
            d_shr = e_shr - b_shr
            d_mdd = e_mdd - b_mdd
            print(f"     Segments covering {year}: available")
            print(f"     Sharpe: baseline={_f(b_shr)}  exp={_f(e_shr)}  delta={_f(d_shr, sign=True)}")
            print(f"     MaxDD:  baseline={_pct(b_mdd)}  exp={_pct(e_mdd)}  delta={_pct(d_mdd, sign=True)}")
            if e_mdd < b_mdd - 0.01:
                print(f"     FLAG: MaxDD worsens by >1% in {year} -> reject regardless of Sharpe")
            elif e_shr > b_shr:
                print(f"     BETTER: Sharpe improves in {year}")
            else:
                print(f"     NEUTRAL/WORSE in {year}")

    # ------------------------------------------------------------------
    # 7. ESCALATION DECISION
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("7. ESCALATION DECISION")
    print("=" * 72)
    to_increased = has_to and to_d > 0.005
    esc_shr  = shr_d  >= 0.02
    esc_cagr = cagr_d >= 0.0025
    esc_mdd  = mdd_d  > 0.01
    esc_diff = diff_ok
    escalate = esc_shr or esc_cagr or esc_mdd or esc_diff

    print(f"   Sharpe delta >= +0.02?            {'YES' if esc_shr else 'NO '}  ({shr_d:+.3f})")
    print(f"   CAGR delta >= +0.25%?             {'YES' if esc_cagr else 'NO '}  ({cagr_d:+.2%})")
    print(f"   MaxDD delta > +1.0%?              {'YES' if esc_mdd else 'NO '}  ({mdd_d:+.2%})")
    print(f"   Difficult-period improvement?     {'YES' if esc_diff else 'NO '}")
    print()
    if to_increased:
        print(f"   SPECIAL: turnover INCREASED ({to_d:+.2%}) -- renorm effect. Reject unless")
        print(f"   Sharpe improves materially enough to justify the extra cost.")
        print()

    if kill:
        verdict = "FAST-MODE VERDICT: REJECTED (kill switch fires)"
    elif to_increased and not esc_shr and not esc_cagr:
        verdict = "FAST-MODE VERDICT: REJECTED (TO increased, no compensating benefit)"
    elif escalate:
        verdict = "FAST-MODE VERDICT: PASS -- escalate to full walk-forward validation"
    else:
        verdict = "FAST-MODE VERDICT: INSUFFICIENT -- does not meet escalation thresholds"

    print(f"   {verdict}")
    print()

    if not escalate or kill:
        print("   FAILURE INTERPRETATION:")
        if to_increased:
            print("   - TO increased: renorm after partial suppression redistributes weight.")
            print("     The trade-vector filter does not bound aggregate turnover when some")
            print("     assets still trade and renorm inflates their target sizes.")
            print("     Options: (a) try proportional allocation approach -- keep suppressed")
            print("     assets at prior weight, do NOT renorm until an asset crosses tau;")
            print("     (b) try much lower tau (0.5%) to raise suppression rate;")
            print("     (c) accept that target-to-target TO is structural given 63d vol scaling.")
        elif has_to and abs(to_d) < 0.01:
            print("   - TO unchanged: vol-driven trades are predominantly >1.5% per asset.")
            print("     tau=1.5% is too coarse. Try tau=0.5% or accept turnover as structural.")
        elif has_to and to_d < -0.01 and shr_d < -0.02:
            print("   - TO reduced but Sharpe worsened: the small vol-driven adjustments are")
            print("     load-bearing for performance -- not pure noise. The 63d vol scaling")
            print("     provides useful risk adjustment that should not be suppressed.")
            print("     Next: accept turnover level, or explore alternative signal structure.")
        elif has_to and to_d < -0.01:
            print("   - TO reduced, Sharpe roughly neutral (near-zero but below kill threshold).")
            print("     Fast-mode window (2018-2024) does not include 2020/2022 OOS coverage.")
            print("     This result is ambiguous -- escalate to full walk-forward to validate.")

    print("=" * 72)


if __name__ == "__main__":
    main()
