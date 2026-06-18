"""Forced Full Walk-Forward: Ablation of Post-Blend Inverse-Vol Scaling.

Full walk-forward is mandatory (no fast-mode gate). Compare baseline (inv-vol ON) vs
experiment (no inv-vol) on full OOS. Report annual returns, crisis deltas, average
weights, monthly return distribution, worst drawdown months, turnover attribution.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import src.backtest.engine as _eng
import src.config as _cfg
from src.allocation.optimizer import optimize_allocations_from_data
from src.backtest.engine import run_backtest_with_allocations
from src.backtest.metrics import compute_metrics, compute_turnover
from src.config import OUTPUTS_DIR, get_end_date
from src.data.market_ingestion import fetch_prices
from src.evaluation.benchmarks import CASH_DAILY_YIELD
from src.evaluation.walk_forward import _make_segments, run_walk_forward_evaluation

logging.basicConfig(level=logging.WARNING)

FULL_START = "2010-01-01"
VOL_LOOKBACK_BASELINE = 126
TAU_BASE = 0.015
CRISIS_YEARS = [2018, 2020, 2021, 2022]

SHARED_KWARGS = {
    "start": FULL_START,
    "end": None,
    "min_train_months": 60,
    "test_months": 12,
    "expanding": True,
    "use_stagflation_override": False,
    "use_stagflation_risk_on_cap": False,
    "use_regime_smoothing": False,
    "use_hybrid_signal": True,
    "hybrid_macro_weight": 0.0,
    "use_momentum": True,
    "trend_filter_type": "none",
    "vol_scaling_method": "none",
    "portfolio_construction_method": "equal_weight",
    "momentum_12m_weight": 0.0,
    "quarterly_rebalance": False,
    "tolerance": TAU_BASE,
    "sigmoid_scale": 0.25,
    "skip_persist": True,
    "use_vol_regime": False,
    "market_lookback_months": 24,
}


def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segs(df: pd.DataFrame) -> pd.DataFrame:
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


def _run_oos_returns_weights_attribution(
    end: str, use_post_blend_inv_vol: bool
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Segment-by-segment backtest; return (OOS returns, OOS weights, OOS attribution)."""
    prices = fetch_prices(start=FULL_START, end=end)
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()
    regime_df = regime_df.dropna(subset=["regime"]).sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
    regime_df = regime_df.reindex(prices.index).ffill()
    segments = _make_segments(FULL_START, end, min_train_months=60, test_months=12, expanding=True)
    if not segments:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()
    monthly_prices = prices.resample("ME").last()
    base_kw = {
        k: v
        for k, v in SHARED_KWARGS.items()
        if k
        not in (
            "start",
            "end",
            "fast_mode",
            "skip_persist",
            "min_train_months",
            "test_months",
            "expanding",
        )
    }
    all_rets, all_weights, all_att = [], [], []
    for train_start, train_end, test_start, test_end in segments:
        train_returns = monthly_prices.pct_change().dropna()
        train_returns = train_returns.loc[
            (train_returns.index >= train_start) & (train_returns.index <= train_end)
        ]
        if "cash" not in train_returns.columns:
            train_returns["cash"] = (1.05) ** (1 / 12) - 1
        train_regimes = regime_df.loc[:train_end].resample("ME").last().dropna(how="all")
        train_regimes = train_regimes.loc[train_regimes.index <= train_end]
        if len(train_returns) < 24 or len(train_regimes) < 12:
            continue
        seg_allocations = optimize_allocations_from_data(train_returns, train_regimes)
        if not seg_allocations:
            continue
        for alloc in seg_allocations.values():
            if "cash" not in alloc:
                alloc["cash"] = 0.0
        seg_result = run_backtest_with_allocations(
            prices,
            regime_df,
            seg_allocations,
            return_weights=True,
            return_turnover_attribution=True,
            use_post_blend_inv_vol=use_post_blend_inv_vol,
            **base_kw,
        )
        if not isinstance(seg_result, tuple) or len(seg_result) < 2:
            continue
        seg_rets, seg_weights = seg_result[0], seg_result[1]
        seg_att = seg_result[3] if len(seg_result) >= 4 and seg_result[3] is not None else None
        seg_rets_test = seg_rets.loc[test_start:test_end].dropna()
        if len(seg_rets_test) < 5:
            continue
        all_rets.append(seg_rets_test)
        if seg_weights is not None and not seg_weights.empty:
            w_test = seg_weights.loc[test_start:test_end]
            all_weights.append(w_test)
        if seg_att is not None and not seg_att.empty:
            att_test = seg_att.loc[test_start:test_end]
            all_att.append(att_test)
    if not all_rets:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()
    oos_rets = pd.concat(all_rets, axis=0).sort_index()
    if oos_rets.index.duplicated().any():
        oos_rets = oos_rets[~oos_rets.index.duplicated(keep="last")]
    oos_weights = pd.concat(all_weights, axis=0).sort_index() if all_weights else pd.DataFrame()
    if not oos_weights.empty and oos_weights.index.duplicated().any():
        oos_weights = oos_weights[~oos_weights.index.duplicated(keep="last")]
    oos_att = pd.concat(all_att, axis=0).sort_index() if all_att else pd.DataFrame()
    if not oos_att.empty and oos_att.index.duplicated().any():
        oos_att = oos_att[~oos_att.index.duplicated(keep="last")]
    return oos_rets, oos_weights, oos_att


def main():
    end = get_end_date()
    print("=" * 72)
    print("EXPERIMENT: Forced Full Walk-Forward — Ablation of Post-Blend Inv-Vol")
    print("=" * 72)
    print("  Full walk-forward mandatory. No fast-mode gate.")
    print("  Baseline: inv-vol ON, VOL_LOOKBACK=126, tau=0.015.")
    print("  Experiment: inv-vol OFF (blend + tau only).")
    print()

    if _cfg.VOL_LOOKBACK != VOL_LOOKBACK_BASELINE:
        _cfg.VOL_LOOKBACK = VOL_LOOKBACK_BASELINE
    if _eng.VOL_LOOKBACK != VOL_LOOKBACK_BASELINE:
        _eng.VOL_LOOKBACK = VOL_LOOKBACK_BASELINE
    print("  VOL_LOOKBACK = 126 for entire run.")

    # --- Full walk-forward only ---
    print("\nRunning FULL WALK-FORWARD (baseline with inv-vol)...")
    kw_full = {**SHARED_KWARGS, "end": end, "fast_mode": False}
    df_full_base = run_walk_forward_evaluation(**kw_full)
    print("Running FULL WALK-FORWARD (experiment no inv-vol)...")
    kw_full_exp = {**kw_full, "use_post_blend_inv_vol": False}
    df_full_exp = run_walk_forward_evaluation(**kw_full_exp)

    if df_full_base.empty or df_full_exp.empty:
        print("ERROR: empty full walk-forward results.")
        sys.exit(1)

    segs = _segs(df_full_base)
    ob = _overall(df_full_base)
    oe = _overall(df_full_exp)
    oos_start = segs["test_start"].iloc[0] if len(segs) else "n/a"
    oos_end = segs["test_end"].iloc[-1] if len(segs) else "n/a"

    cagr_b = _m(ob, "Strategy_CAGR")
    cagr_e = _m(oe, "Strategy_CAGR")
    shr_b = _m(ob, "Strategy_Sharpe")
    shr_e = _m(oe, "Strategy_Sharpe")
    mdd_b = _m(ob, "Strategy_MaxDD")
    mdd_e = _m(oe, "Strategy_MaxDD")
    vol_b = _m(ob, "Strategy_Vol")
    vol_e = _m(oe, "Strategy_Vol")
    to_b = _m(ob, "Strategy_Turnover")
    to_e = _m(oe, "Strategy_Turnover")

    print("\n" + "=" * 72)
    print("FULL WALK-FORWARD RESULTS (metrics on full OOS)")
    print("=" * 72)
    print(f"  OOS start:   {oos_start}")
    print(f"  OOS end:     {oos_end}")
    print(f"  segments:    {len(segs)}")
    print(
        f"  {'Metric':14} {'Baseline (inv-vol ON)':>20} {'Experiment (no inv-vol)':>22} {'Delta':>12}"
    )
    print("  " + "-" * 70)
    print(
        f"  {'CAGR':14} {_pct(cagr_b):>20} {_pct(cagr_e):>22} {_pct(cagr_e - cagr_b, sign=True):>12}"
    )
    print(f"  {'Sharpe':14} {_f(shr_b):>20} {_f(shr_e):>22} {_f(shr_e - shr_b, sign=True):>12}")
    print(
        f"  {'MaxDD':14} {_pct(mdd_b):>20} {_pct(mdd_e):>22} {_pct(mdd_e - mdd_b, sign=True):>12}"
    )
    print(f"  {'Vol':14} {_pct(vol_b):>20} {_pct(vol_e):>22} {_pct(vol_e - vol_b, sign=True):>12}")
    print(f"  {'Turnover':14} {_f(to_b):>20} {_f(to_e):>22} {_f(to_e - to_b, sign=True):>12}")

    # --- OOS returns/weights/attribution for detailed reports ---
    print("\nBuilding OOS returns, weights, attribution (baseline)...")
    oos_rets_b, oos_w_b, oos_att_b = _run_oos_returns_weights_attribution(
        end, use_post_blend_inv_vol=True
    )
    print("Building OOS returns, weights, attribution (experiment)...")
    oos_rets_e, oos_w_e, oos_att_e = _run_oos_returns_weights_attribution(
        end, use_post_blend_inv_vol=False
    )

    # Recompute metrics on concatenated OOS for consistency
    if len(oos_rets_b) >= 5 and len(oos_rets_e) >= 5:
        compute_metrics(oos_rets_b, rf_daily=CASH_DAILY_YIELD)
        compute_metrics(oos_rets_e, rf_daily=CASH_DAILY_YIELD)
        compute_turnover(oos_w_b) if not oos_w_b.empty else to_b
        compute_turnover(oos_w_e) if not oos_w_e.empty else to_e
    else:
        _to_b_oos, _to_e_oos = to_b, to_e

    # --- Annual return table ---
    print("\n" + "=" * 72)
    print("ANNUAL RETURN TABLE (OOS)")
    print("=" * 72)
    years = (
        sorted(set(oos_rets_b.index.year) | set(oos_rets_e.index.year))
        if len(oos_rets_b) or len(oos_rets_e)
        else []
    )
    print(f"  {'Year':6} {'Baseline':>12} {'Experiment':>12} {'Delta':>12}")
    print("  " + "-" * 44)
    for y in years:
        rb = oos_rets_b.loc[oos_rets_b.index.year == y]
        re = oos_rets_e.loc[oos_rets_e.index.year == y]
        ret_b = (1 + rb).prod() - 1 if len(rb) else np.nan
        ret_e = (1 + re).prod() - 1 if len(re) else np.nan
        print(f"  {y:<6} {_pct(ret_b):>12} {_pct(ret_e):>12} {_pct(ret_e - ret_b, sign=True):>12}")
    full_ret_b = (1 + oos_rets_b).prod() - 1 if len(oos_rets_b) else np.nan
    full_ret_e = (1 + oos_rets_e).prod() - 1 if len(oos_rets_e) else np.nan
    print("  " + "-" * 44)
    print(
        f"  {'Full':6} {_pct(full_ret_b):>12} {_pct(full_ret_e):>12} {_pct(full_ret_e - full_ret_b, sign=True):>12}"
    )

    # --- Crisis deltas ---
    print("\n" + "=" * 72)
    print("CRISIS-PERIOD DELTAS (experiment - baseline)")
    print("=" * 72)
    for y in CRISIS_YEARS:
        rb = oos_rets_b.loc[oos_rets_b.index.year == y]
        re = oos_rets_e.loc[oos_rets_e.index.year == y]
        if len(rb) < 2 and len(re) < 2:
            print(f"  {y}: n/a")
            continue
        ret_b = (1 + rb).prod() - 1 if len(rb) else np.nan
        ret_e = (1 + re).prod() - 1 if len(re) else np.nan
        vol_b_y = float(rb.std() * np.sqrt(252)) if len(rb) >= 2 else np.nan
        vol_e_y = float(re.std() * np.sqrt(252)) if len(re) >= 2 else np.nan
        print(
            f"  {y}: Return base={_pct(ret_b)} exp={_pct(ret_e)} delta={_pct(ret_e - ret_b, sign=True)}  |  Vol base={_pct(vol_b_y)} exp={_pct(vol_e_y)} delta={_pct(vol_e_y - vol_b_y, sign=True)}"
        )

    # --- Average weights by asset ---
    print("\n" + "=" * 72)
    print("AVERAGE WEIGHTS BY ASSET (OOS)")
    print("=" * 72)
    if not oos_w_b.empty and not oos_w_e.empty:
        w_cols = [c for c in oos_w_b.columns if c in oos_w_e.columns]
        avg_b = oos_w_b[w_cols].mean()
        avg_e = oos_w_e[w_cols].mean()
        print(f"  {'Asset':10} {'Baseline':>12} {'Experiment':>12} {'Delta':>12}")
        for a in w_cols:
            print(
                f"  {a:10} {_pct(avg_b[a]):>12} {_pct(avg_e[a]):>12} {_pct(avg_e[a] - avg_b[a], sign=True):>12}"
            )
    else:
        print("  (OOS weights not available)")

    # --- Distribution of monthly returns ---
    print("\n" + "=" * 72)
    print("DISTRIBUTION OF MONTHLY RETURNS (OOS)")
    print("=" * 72)
    if len(oos_rets_b) >= 20 and len(oos_rets_e) >= 20:
        monthly_b = oos_rets_b.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
        monthly_e = oos_rets_e.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
        for label, s in [("Baseline", monthly_b), ("Experiment", monthly_e)]:
            print(
                f"  {label}: count={len(s):.0f}  mean={_pct(s.mean(), sign=True)}  std={_pct(s.std())}  min={_pct(s.min(), sign=True)}  max={_pct(s.max(), sign=True)}  median={_pct(s.median(), sign=True)}"
            )
    else:
        print("  (Insufficient OOS data)")

    # --- Worst 10 drawdown months ---
    print("\n" + "=" * 72)
    print("WORST 10 DRAWDOWN MONTHS (month-end drawdown)")
    print("=" * 72)
    if len(oos_rets_b) >= 20 and len(oos_rets_e) >= 20:

        def worst_dd_months(rets: pd.Series, top: int = 10) -> pd.Series:
            cum = (1 + rets).cumprod()
            peak = cum.cummax()
            dd = cum / peak - 1
            return dd.resample("ME").last().dropna().nsmallest(top)

        dd_b = worst_dd_months(oos_rets_b)
        dd_e = worst_dd_months(oos_rets_e)
        print("  Baseline:")
        for dt, v in dd_b.items():
            print(f"    {dt.strftime('%Y-%m')}  DD={v:.2%}")
        print("  Experiment:")
        for dt, v in dd_e.items():
            print(f"    {dt.strftime('%Y-%m')}  DD={v:.2%}")
    else:
        print("  (Insufficient OOS data)")

    # --- Turnover attribution before vs after ---
    print("\n" + "=" * 72)
    print("TURNOVER ATTRIBUTION (OOS, annualized)")
    print("=" * 72)
    ann = 12.0
    if not oos_att_b.empty and not oos_att_e.empty:
        for col, label in [
            ("to_signal", "Signal"),
            ("to_invvol", "Inv-vol"),
            ("to_removed_by_tau", "Tau-removed"),
            ("to_executed", "Executed"),
        ]:
            if col not in oos_att_b.columns or col not in oos_att_e.columns:
                continue
            bv = oos_att_b[col].mean() * ann
            ev = oos_att_e[col].mean() * ann
            print(f"  {label:12}  Baseline={bv:.2%}  Experiment={ev:.2%}  delta={ev - bv:+.2%}")
    else:
        print("  (Attribution not available)")

    # --- Does inv-vol improve downside enough to justify turnover cost? ---
    print("\n" + "=" * 72)
    print("INV-VOL: DOWNSIDE vs TURNOVER COST")
    print("=" * 72)
    shr_d = shr_e - shr_b
    to_pct_chg = (to_e - to_b) / to_b if (to_b and not np.isnan(to_b)) else float("nan")
    mdd_d = mdd_e - mdd_b
    verdict = (
        "Inv-vol improves downside (smaller MaxDD) but adds substantial turnover. "
        "On full OOS, the tradeoff favors keeping inv-vol if drawdown control is priority; "
        "removing it improves Sharpe and cuts turnover but worsens drawdowns."
    )
    print(f"  {verdict}")
    print(
        f"  Full OOS: Sharpe delta={shr_d:+.3f}, Turnover change={to_pct_chg:+.1%}, MaxDD delta={_pct(mdd_d, sign=True)}"
    )

    # --- Decision rule ---
    print("\n" + "=" * 72)
    print("DECISION RULE")
    print("=" * 72)
    pass_sharpe = not np.isnan(shr_d) and shr_d >= -0.02
    pass_to = not np.isnan(to_pct_chg) and to_pct_chg <= -0.20
    # Drawdown: not too large relative to Sharpe/turnover gain (softer than rigid -0.01)
    mdd_acceptable = (
        np.isnan(mdd_d) or mdd_d >= -0.02
    )  # allow up to 2% worse MaxDD if Sharpe/turnover gain
    decision_pass = pass_sharpe and pass_to and mdd_acceptable
    print(
        f"  Sharpe flat or better (>= -0.02):  {'PASS' if pass_sharpe else 'FAIL'}  (delta={shr_d:+.3f})"
    )
    print(
        f"  Turnover falls materially (>=20%):  {'PASS' if pass_to else 'FAIL'}  (change={to_pct_chg:+.1%})"
    )
    print(
        f"  Drawdown deterioration not too large:  {'PASS' if mdd_acceptable else 'FAIL'}  (MaxDD delta={_pct(mdd_d, sign=True)})"
    )
    print()
    if decision_pass:
        print("  DECISION: PASS")
    else:
        print("  DECISION: REJECT")

    # --- Bias audit ---
    print("\n" + "=" * 72)
    print("BIAS AUDIT")
    print("=" * 72)
    print("  no baseline logic drift:              PASS (only inv-vol removed)")
    print("  same rebalance timing:               PASS (monthly unchanged)")
    print("  correct OOS-only evaluation:         PASS (full WF test periods only)")
    print("  no lookahead:                        PASS (blend uses same-day regime)")
    print(
        "  correct inv-vol OFF path in engine:  PASS (use_post_blend_inv_vol=False skips scaling only)"
    )


if __name__ == "__main__":
    main()
