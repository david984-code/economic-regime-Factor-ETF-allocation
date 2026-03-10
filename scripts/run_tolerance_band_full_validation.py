"""Full Walk-Forward Validation: Tolerance-Band Execution Filter (tau = 1.5%).

Single variable change: tolerance 0.0 -> 0.015.
Full OOS coverage from 2010-01-01 (first OOS ~2015 after 60-month training).

Implementation reminder:
  trade = new_w - prev_w
  exec_trade = trade * (|trade| > tau)
  w_exec = prev_w + exec_trade,  then renormalize to sum=1

Trade suppression diagnostics use analytical weight reconstruction (same
logic as run_turnover_attribution.py) to compute per-asset suppression
frequencies without requiring a second database-backed backtest call.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (
    ASSETS, RISK_OFF_ASSETS_BASE, RISK_ON_ASSETS_BASE, TICKERS, VOL_LOOKBACK,
)
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation
from src.features.transforms import sigmoid

logging.basicConfig(level=logging.WARNING)

FULL_START = "2010-01-01"
FULL_END   = None
TAU        = 0.015

# Accepted walk-forward baseline (full history prior runs)
KNOWN_BASE_CAGR   = 0.075
KNOWN_BASE_SHARPE = 0.52
KNOWN_BASE_TO     = 1.18

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
    market_lookback_months=24,
)


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------

def _pct(v: float, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    return f"{v:+.2%}" if sign else f"{v:.2%}"


def _f(v: float, d: int = 3, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    return fmt.format(v)


def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _segs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["segment"] != "OVERALL"].copy()


def _m(row, col: str) -> float:
    v = row.get(col, float("nan"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _filter_year_range(segs: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    def overlaps(row):
        try:
            ts = pd.Period(row["test_start"], freq="M").year
            te = pd.Period(row["test_end"], freq="M").year
            return ts <= y1 and te >= y0
        except Exception:
            return False
    return segs[segs.apply(overlaps, axis=1)]


def _mean(segs: pd.DataFrame, col: str) -> float:
    return float(segs[col].dropna().mean()) if col in segs.columns and len(segs) else float("nan")


def _sortino_approx(segs: pd.DataFrame) -> float:
    """Approximate Sortino from segment-level Sharpe and CAGR - not rigorous; returns nan."""
    return float("nan")


# ---------------------------------------------------------------------------
# Analytical weight reconstruction for trade suppression diagnostics
# ---------------------------------------------------------------------------

def _compute_risk_on_me(spy_monthly: pd.Series, lookback: int = 24) -> pd.Series:
    """Reproduce engine risk_on pipeline: momentum -> expanding z-score -> sigmoid."""
    n = len(spy_monthly)
    raw = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback:
            raw[i] = spy_monthly.iloc[i] / spy_monthly.iloc[i - lookback] - 1
    raw_s = pd.Series(raw, index=spy_monthly.index)
    min_hist = max(lookback, 12)
    z = raw_s.copy() * 0.0
    for i in range(n):
        trailing = raw_s.iloc[: i + 1].dropna()
        if len(trailing) >= min_hist:
            z.iloc[i] = (raw_s.iloc[i] - trailing.mean()) / trailing.std()
        else:
            z.iloc[i] = 0.0
    return sigmoid(z * 0.25)


def _vol_scaled_weights(risk_on: float, std_dict: dict) -> dict[str, float]:
    """
    Reproduce engine: equal-weight per sleeve, blend by risk_on,
    then inverse-vol scale within each sleeve, then blend again.
    Matches vol_scaled_weights_from_std logic used by engine.
    """
    ro_sleeve = RISK_ON_ASSETS_BASE
    rof_sleeve = RISK_OFF_ASSETS_BASE
    n_ro = len(ro_sleeve)
    n_rof = len(rof_sleeve)

    # Equal-weight base per sleeve
    w_ro_eq  = {a: 1.0 / n_ro  for a in ro_sleeve}
    w_rof_eq = {a: 1.0 / n_rof for a in rof_sleeve}

    # Inverse-vol scaling within each sleeve
    def _invvol(sleeve_w: dict, std_dict: dict) -> dict:
        invvols = {}
        for a, w in sleeve_w.items():
            s = std_dict.get(a)
            invvols[a] = 1.0 / s if (s is not None and s > 0) else 1.0
        total = sum(invvols.values())
        return {a: v / total for a, v in invvols.items()} if total > 0 else sleeve_w

    w_ro_iv  = _invvol(w_ro_eq,  std_dict)
    w_rof_iv = _invvol(w_rof_eq, std_dict)

    # Blend by risk_on
    blended = {}
    for a in ro_sleeve:
        blended[a] = risk_on * w_ro_iv[a]
    for a in rof_sleeve:
        blended[a] = (1.0 - risk_on) * w_rof_iv[a]

    total = sum(blended.values())
    if total > 0:
        blended = {a: v / total for a, v in blended.items()}
    return blended


def _apply_tolerance(new_w: np.ndarray, prev_w: np.ndarray,
                     tau: float, asset_order: list) -> np.ndarray:
    """Trade-vector filter: suppress trades where |delta| <= tau, then renorm."""
    trade = new_w - prev_w
    exec_trade = np.where(np.abs(trade) > tau, trade, 0.0)
    w_exec = prev_w + exec_trade
    total = w_exec.sum()
    return (w_exec / total) if total > 0 else new_w


def _reconstruct_weights_daily(prices: pd.DataFrame, tau: float = 0.0
                                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct daily weight snapshots (weight at each rebalance date) for
    both baseline (tau=0) and experiment (tau>0). Also return the monthly
    risk_on series for regime context.

    Returns:
        w_base_me : month-end target weights, baseline
        w_exp_me  : month-end target weights, experiment
        ro_me     : month-end risk_on
    """
    spy_monthly = prices["SPY"].resample("ME").last()
    risk_on_me = _compute_risk_on_me(spy_monthly, lookback=24)

    # 63-day rolling std on daily prices
    rolling_std = prices[TICKERS].rolling(VOL_LOOKBACK, min_periods=1).std()

    all_assets = list(RISK_ON_ASSETS_BASE) + list(RISK_OFF_ASSETS_BASE)
    n_assets = len(all_assets)

    # Rebalance dates: first trading day of each month
    daily_index = prices.index
    months = pd.Series(daily_index).dt.to_period("M").values
    month_changed = np.concatenate([[True], months[1:] != months[:-1]])
    rebal_dates = daily_index[month_changed]

    # Initialise weight arrays
    prev_base = np.ones(n_assets) / n_assets
    prev_exp  = np.ones(n_assets) / n_assets

    rec_base, rec_exp, rec_ro, rec_dates = [], [], [], []

    for rd in rebal_dates:
        # Get risk_on: use last month-end before or on rd
        me_idx = risk_on_me.index.asof(rd)
        if pd.isna(me_idx):
            ro = 0.5
        else:
            ro = float(risk_on_me.loc[me_idx])

        # Get rolling std at this rebalance date
        std_idx = rolling_std.index.asof(rd)
        if pd.isna(std_idx):
            std_row = rolling_std.iloc[0]
        else:
            std_row = rolling_std.loc[std_idx]
        std_dict = {a: float(std_row[a]) if a in std_row.index and pd.notna(std_row[a]) else None
                    for a in TICKERS}

        # Compute target weights
        wdict = _vol_scaled_weights(ro, std_dict)
        new_w = np.array([wdict.get(a, 0.0) for a in all_assets])

        # Baseline (no filter)
        base_w = new_w.copy()
        # Experiment (trade-vector filter)
        exp_w = _apply_tolerance(new_w, prev_exp, tau, all_assets)

        prev_base = base_w
        prev_exp  = exp_w

        rec_base.append(base_w)
        rec_exp.append(exp_w)
        rec_ro.append(ro)
        rec_dates.append(rd)

    w_base_me = pd.DataFrame(rec_base, index=pd.DatetimeIndex(rec_dates), columns=all_assets)
    w_exp_me  = pd.DataFrame(rec_exp,  index=pd.DatetimeIndex(rec_dates), columns=all_assets)
    ro_me     = pd.Series(rec_ro, index=pd.DatetimeIndex(rec_dates), name="risk_on")

    return w_base_me, w_exp_me, ro_me


def _suppression_stats(w_base: pd.DataFrame, w_exp: pd.DataFrame,
                       all_assets: list, tau: float) -> dict:
    """Compute per-asset suppression statistics by comparing target vs executed weights."""
    n = len(w_base)
    stats = {"n_rebalances": n}

    # Months where at least one asset was suppressed
    # Asset is "suppressed" when |exp - base| > 0.0001 (floating-pt threshold)
    diff_mat = (w_exp - w_base).abs()
    suppressed_mat = diff_mat > 0.0001  # any difference means filter acted
    months_any_suppressed = suppressed_mat.any(axis=1).sum()
    stats["pct_months_any_suppressed"] = months_any_suppressed / n if n > 0 else 0.0

    # Median suppressed assets per month
    suppressed_counts = suppressed_mat.sum(axis=1)
    stats["median_suppressed_per_month"] = float(suppressed_counts.median())
    stats["mean_suppressed_per_month"]   = float(suppressed_counts.mean())

    # Per-asset suppression frequency (% of months where filter changed that asset's weight)
    per_asset = {}
    for a in all_assets:
        if a in diff_mat.columns:
            freq = (diff_mat[a] > 0.0001).mean()
            med_size = diff_mat[a][diff_mat[a] > 0.0001].median()
            per_asset[a] = {
                "suppression_freq": float(freq),
                "median_delta": float(med_size) if not pd.isna(med_size) else 0.0,
            }
    stats["per_asset"] = per_asset
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("FULL WALK-FORWARD VALIDATION: Tolerance-Band Filter  tau = 1.5%")
    print(f"Full dataset start: {FULL_START}  |  OOS coverage starts ~2015")
    print(f"Single variable: tolerance 0.000 -> {TAU:.3f}")
    print("=" * 72)

    # -- Load prices for analytical diagnostics --------------------------------
    print("\nLoading price data...")
    prices = fetch_prices(start=FULL_START, end=FULL_END)
    date_range = f"{prices.index[0].date()} to {prices.index[-1].date()}"
    print(f"Price data: {date_range}")

    # -- PRE-RUN VERIFICATION --------------------------------------------------
    print(f"\nVOL_LOOKBACK = {VOL_LOOKBACK}  [must be 63]")
    if VOL_LOOKBACK != 63:
        print("STOP: VOL_LOOKBACK != 63.")
        sys.exit(1)
    print(f"market_lookback_months = 24  [both runs]")
    print(f"tolerance: 0.000 (baseline)  /  {TAU:.3f} (experiment)")
    print("Parameter diff: tolerance only. All else identical.")

    # -- RUN BOTH WALK-FORWARDS ------------------------------------------------
    print("\nRunning BASELINE  (tolerance=0.000)...")
    df_base = run_walk_forward_evaluation(**SHARED_KWARGS, tolerance=0.0)

    print("Running EXPERIMENT (tolerance=0.015)...")
    df_exp = run_walk_forward_evaluation(**SHARED_KWARGS, tolerance=TAU)

    if df_base.empty or df_exp.empty:
        print("ERROR: empty results.")
        sys.exit(1)

    segs_b = _segs(df_base)
    segs_e = _segs(df_exp)

    idx_b = segs_b[["test_start", "test_end"]].reset_index(drop=True)
    idx_e = segs_e[["test_start", "test_end"]].reset_index(drop=True)
    seg_match = idx_b.equals(idx_e)
    print(f"\nOOS segments identical: {'YES' if seg_match else 'NO -- STOP'}")
    if not seg_match:
        sys.exit(1)
    n_segs = len(segs_b)
    print(f"OOS segment count: {n_segs}")
    oos_start = segs_b["test_start"].iloc[0]
    oos_end   = segs_b["test_end"].iloc[-1]
    print(f"OOS coverage: {oos_start} to {oos_end}")

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

    # =========================================================================
    print("\n" + "=" * 72)
    print("1. FULL-PERIOD METRICS vs BASELINE")
    print("=" * 72)
    print(f"\n   Known full-history baseline: CAGR~{KNOWN_BASE_CAGR:.1%}  "
          f"Sharpe~{KNOWN_BASE_SHARPE:.2f}  TO~{KNOWN_BASE_TO:.0%}")
    print(f"   (Current baseline may differ slightly due to date updates)\n")
    print(f"   {'Metric':30} {'Baseline':>14} {'Exp (tau=1.5%)':>14} {'Delta':>10}")
    print("   " + "-" * 70)
    rows = [
        ("CAGR",    _pct(cagr_b), _pct(cagr_e), _pct(cagr_d, sign=True)),
        ("Sharpe",  _f(shr_b),    _f(shr_e),    _f(shr_d, sign=True)),
        ("MaxDD",   _pct(mdd_b),  _pct(mdd_e),  _pct(mdd_d, sign=True)),
        ("Vol",     _pct(vol_b),  _pct(vol_e),  _pct(vol_d, sign=True)),
        ("Turnover",
         _pct(to_b) if has_to else "n/a",
         _pct(to_e) if has_to else "n/a",
         _pct(to_d, sign=True) if has_to else "n/a"),
        ("Sortino", "see note", "see note", "n/a"),
    ]
    for label, b, e, d in rows:
        print(f"   {label:30} {b:>14} {e:>14} {d:>10}")
    print(f"\n   Note: Sortino not available from segment-level walk-forward output.")
    print(f"         Sharpe ratio is the primary risk-adjusted metric here.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("2. SUBPERIOD COMPARISON")
    print("=" * 72)
    subperiods = [
        ("2013-2017", 2013, 2017),
        ("2018-2020", 2018, 2020),
        ("2021-2022", 2021, 2022),
        ("2023-pres", 2023, 2030),
    ]
    print(f"\n   {'Period':12} {'B CAGR':>8} {'E CAGR':>8} {'Dd':>6} "
          f"{'B Shr':>7} {'E Shr':>7} {'Ds':>6} "
          f"{'B MDD':>8} {'E MDD':>8} {'Dm':>7} "
          f"{'B TO':>7} {'E TO':>7} {'Dt':>7}")
    print("   " + "-" * 100)
    for label, y0, y1 in subperiods:
        sb = _filter_year_range(segs_b, y0, y1)
        se = _filter_year_range(segs_e, y0, y1)
        if len(sb) == 0:
            print(f"   {label:12} {'no OOS data':>8}")
            continue
        bc = _mean(sb, "Strategy_CAGR");   ec = _mean(se, "Strategy_CAGR")
        bs = _mean(sb, "Strategy_Sharpe"); es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD");  em = _mean(se, "Strategy_MaxDD")
        bt = _mean(sb, "Strategy_Turnover"); et = _mean(se, "Strategy_Turnover")
        has_sub_to = not (np.isnan(bt) or np.isnan(et))
        n = len(sb)
        print(f"   {label:12} "
              f"{_pct(bc):>8} {_pct(ec):>8} {_pct(ec-bc, sign=True):>6} "
              f"{_f(bs):>7} {_f(es):>7} {_f(es-bs, sign=True):>6} "
              f"{_pct(bm):>8} {_pct(em):>8} {_pct(em-bm, sign=True):>7} "
              f"{(_pct(bt) if has_sub_to else 'n/a'):>7} "
              f"{(_pct(et) if has_sub_to else 'n/a'):>7} "
              f"{(_pct(et-bt, sign=True) if has_sub_to else 'n/a'):>7} "
              f"[{n} seg]")

    # =========================================================================
    print("\n" + "=" * 72)
    print("3. CRISIS-PERIOD CHECKS")
    print("=" * 72)
    crises = [
        ("2018 Q4",           2018, 2019, "Q4 equity selloff"),
        ("2020 COVID",        2020, 2020, "Max drawdown test"),
        ("2022 rate shock",   2022, 2022, "Bonds + equities both fell"),
    ]
    for label, y0, y1, note in crises:
        sb = _filter_year_range(segs_b, y0, y1)
        se = _filter_year_range(segs_e, y0, y1)
        print(f"\n   {label}  ({note})")
        if len(sb) == 0:
            print(f"     No OOS segments cover {y0}-{y1}.")
            if y0 < 2015:
                print(f"     Expected: min 60-month training means OOS starts ~2015.")
            else:
                print(f"     Check: OOS coverage is {oos_start} to {oos_end}.")
            continue
        bc = _mean(sb, "Strategy_CAGR");   ec = _mean(se, "Strategy_CAGR")
        bs = _mean(sb, "Strategy_Sharpe"); es = _mean(se, "Strategy_Sharpe")
        bm = _mean(sb, "Strategy_MaxDD");  em = _mean(se, "Strategy_MaxDD")
        bt = _mean(sb, "Strategy_Turnover"); et = _mean(se, "Strategy_Turnover")
        has_sub_to = not (np.isnan(bt) or np.isnan(et))
        print(f"     Segments: {len(sb)}")
        print(f"     CAGR:    baseline={_pct(bc)}  exp={_pct(ec)}  delta={_pct(ec-bc, sign=True)}")
        print(f"     Sharpe:  baseline={_f(bs)}   exp={_f(es)}   delta={_f(es-bs, sign=True)}")
        print(f"     MaxDD:   baseline={_pct(bm)}  exp={_pct(em)}  delta={_pct(em-bm, sign=True)}")
        if has_sub_to:
            print(f"     Turnover: baseline={_pct(bt)}  exp={_pct(et)}  delta={_pct(et-bt, sign=True)}")

        # Interpretation
        if em < bm - 0.01:
            print(f"     FLAG: MaxDD worsens by >{abs(em-bm):.1%} -> filter may suppress defensive moves")
        elif em > bm + 0.01:
            print(f"     GOOD: MaxDD improves by >{abs(em-bm):.1%} -> filter does not hurt crisis navigation")
        elif es < bs - 0.05:
            print(f"     CONCERN: Sharpe materially worse ({es-bs:+.3f}) in stress period")
        elif es > bs + 0.05:
            print(f"     GOOD: Sharpe improves in stress period")
        else:
            print(f"     NEUTRAL: performance approximately flat in this crisis period")

    # =========================================================================
    print("\n" + "=" * 72)
    print("4. TRADE SUPPRESSION DIAGNOSTICS")
    print("=" * 72)
    print("\n   Reconstructing weight series analytically (same vol-scale pipeline as engine)...")
    print(f"   tau = {TAU:.3f}  |  VOL_LOOKBACK = {VOL_LOOKBACK}  |  lookback = 24M")

    try:
        w_base_me, w_exp_me, ro_me = _reconstruct_weights_daily(prices, tau=TAU)
        all_assets = list(RISK_ON_ASSETS_BASE) + list(RISK_OFF_ASSETS_BASE)

        stats = _suppression_stats(w_base_me, w_exp_me, all_assets, TAU)

        print(f"\n   Total rebalance events:              {stats['n_rebalances']}")
        print(f"   Rebalances with >= 1 suppressed:     "
              f"{stats['pct_months_any_suppressed']:.1%}")
        print(f"   Median suppressed assets / month:    {stats['median_suppressed_per_month']:.1f}")
        print(f"   Mean suppressed assets / month:      {stats['mean_suppressed_per_month']:.1f}")

        print(f"\n   Per-asset suppression frequency (% of rebalances filter acted on asset):")
        print(f"\n   {'Asset':8} {'Sleeve':9} {'Suppress freq':>14} {'Median |delta|':>15}")
        print("   " + "-" * 50)
        pa = stats["per_asset"]
        for a in ["IEF", "GLD", "TLT", "SPY", "MTUM", "VLUE", "QUAL", "USMV", "IJR", "VIG"]:
            if a not in pa:
                continue
            sleeve = "risk-off" if a in RISK_OFF_ASSETS_BASE else "risk-on"
            freq   = pa[a]["suppression_freq"]
            med_d  = pa[a]["median_delta"]
            print(f"   {a:8} {sleeve:9} {freq:>14.1%} {med_d:>15.4f}")

        # Summary: risk-on vs risk-off suppression
        ro_freqs  = [pa[a]["suppression_freq"] for a in RISK_ON_ASSETS_BASE  if a in pa]
        rof_freqs = [pa[a]["suppression_freq"] for a in RISK_OFF_ASSETS_BASE if a in pa]
        avg_ro  = np.mean(ro_freqs)  if ro_freqs  else float("nan")
        avg_rof = np.mean(rof_freqs) if rof_freqs else float("nan")
        print(f"\n   Average suppression frequency by sleeve:")
        print(f"     Risk-on  sleeve: {avg_ro:.1%}")
        print(f"     Risk-off sleeve: {avg_rof:.1%}")
        if not np.isnan(avg_rof) and not np.isnan(avg_ro):
            if avg_rof > avg_ro + 0.05:
                print(f"   CONFIRMED: risk-off assets suppressed more frequently (consistent with")
                print(f"   attribution finding: risk-off drives dominant vol churn)")
            elif avg_ro > avg_rof + 0.05:
                print(f"   UNEXPECTED: risk-on assets suppressed more than risk-off.")
                print(f"   The filter may be disproportionately affecting growth asset trades.")
            else:
                print(f"   SYMMETRIC: filter acts similarly on both sleeves.")

        # Check whether crisis-period rebalances were suppressed (risk: delayed defensive shift)
        print(f"\n   Crisis-period filter activity:")
        for label, yr in [("2020", 2020), ("2022", 2022)]:
            crisis_dates = [d for d in w_base_me.index if d.year == yr]
            if not crisis_dates:
                print(f"     {label}: no rebalance dates in this year in reconstruction window.")
                continue
            crisis_idx = [w_base_me.index.get_loc(d) for d in crisis_dates]
            diffs = (w_exp_me - w_base_me).abs().iloc[crisis_idx]
            any_suppressed = (diffs > 0.0001).any(axis=1).mean()
            ro_vals = ro_me.iloc[crisis_idx]
            print(f"     {label}: {len(crisis_dates)} rebalances  |  "
                  f"% with suppression: {any_suppressed:.0%}  |  "
                  f"mean risk_on: {ro_vals.mean():.3f}  [0=fully defensive, 1=fully risk-on]")
            if any_suppressed > 0.5:
                if ro_vals.mean() < 0.4:
                    print(f"            WARNING: filter is active during risk-off regime in {label}.")
                    print(f"            Defensive rebalancing may be partially suppressed.")
                else:
                    print(f"            Filter active but risk_on is moderate/elevated ({ro_vals.mean():.3f})")
                    print(f"            Suppression unlikely to delay defensive shift significantly.")

    except Exception as exc:
        print(f"\n   Analytical reconstruction failed: {exc}")
        print(f"   Suppression diagnostics not available. Key metrics from walk-forward only.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("5. TURNOVER BENEFIT QUALITY")
    print("=" * 72)
    if has_to:
        print(f"\n   Full walk-forward turnover comparison:")
        print(f"     Baseline (tau=0.000):   {to_b:.2%}")
        print(f"     Experiment (tau=0.015): {to_e:.2%}")
        print(f"     Absolute reduction:     {-to_d:.2%}  ({-to_d/to_b:.1%} relative)")
        print(f"     Ratio (exp / base):     {to_r:.3f}x")
        print(f"\n   Fast-mode reference: 71.93% -> 47.43% (-24.50pp, 0.66x)")
        to_persist = to_d / (-0.2450) if abs(-0.2450) > 0.001 else float("nan")
        if has_to and not np.isnan(to_d):
            print(f"   Full-history vs fast-mode reduction consistency:")
            print(f"     Fast-mode delta: -24.50pp  |  Full-history delta: {to_d:+.2%}")
            if to_d < 0:
                print(f"     Turnover reduction persists in full walk-forward: YES")
                if abs(to_d) > 0.15:
                    print(f"     Magnitude > 15pp: material and robust reduction")
                elif abs(to_d) > 0.05:
                    print(f"     Magnitude 5-15pp: meaningful but partial")
                else:
                    print(f"     Magnitude < 5pp: small in full history despite fast-mode signal")
            else:
                print(f"     FLAG: turnover increased in full walk-forward despite fast-mode reduction")

        # Regime analysis of turnover
        print(f"\n   Turnover by subperiod (experiment vs baseline):")
        print(f"   {'Period':12} {'Base TO':>9} {'Exp TO':>9} {'Delta':>9} {'Ratio':>7}")
        print("   " + "-" * 52)
        for label, y0, y1 in [("2013-2017",2013,2017),("2018-2020",2018,2020),
                               ("2021-2022",2021,2022),("2023-pres",2023,2030)]:
            sb = _filter_year_range(segs_b, y0, y1)
            se = _filter_year_range(segs_e, y0, y1)
            if len(sb) == 0:
                continue
            bt = _mean(sb, "Strategy_Turnover")
            et = _mean(se, "Strategy_Turnover")
            if np.isnan(bt) or np.isnan(et):
                print(f"   {label:12} {'n/a':>9} {'n/a':>9}")
                continue
            r = et / bt if bt > 0 else float("nan")
            print(f"   {label:12} {bt:>9.2%} {et:>9.2%} {et-bt:>+9.2%} {r:>7.3f}x")
    else:
        print("   Turnover not available in walk-forward output.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("6. FINAL VERDICT")
    print("=" * 72)

    # Assessment criteria
    perf_approx_flat = abs(shr_d) <= 0.05 and abs(cagr_d) <= 0.005
    perf_materially_worse = shr_d < -0.05 or cagr_d < -0.01
    to_materially_reduced = has_to and to_d < -0.10
    to_ratio_acceptable   = has_to and to_r < 0.85

    # Check crisis periods
    crisis_mdd_deltas = []
    for label, y0, y1, _ in crises:
        sb = _filter_year_range(segs_b, y0, y1)
        se = _filter_year_range(segs_e, y0, y1)
        if len(sb) == 0:
            continue
        bm = _mean(sb, "Strategy_MaxDD")
        em = _mean(se, "Strategy_MaxDD")
        if not np.isnan(bm) and not np.isnan(em):
            crisis_mdd_deltas.append((label, em - bm))
    crisis_mdd_worsens = any(d < -0.02 for _, d in crisis_mdd_deltas)
    crisis_mdd_ok      = not crisis_mdd_worsens

    print(f"\n   Performance approximately flat (|dSharpe|<=0.05, |dCAGR|<=0.5%)?  "
          f"{'YES' if perf_approx_flat else 'NO '}"
          f"  (dSharpe={shr_d:+.3f}, dCAGR={cagr_d:+.2%})")
    print(f"   Performance materially worse (Sharpe<-0.05 or CAGR<-1%)?          "
          f"{'YES' if perf_materially_worse else 'NO '}")
    print(f"   Turnover materially reduced (delta > -10pp)?                       "
          f"{'YES' if to_materially_reduced else ('NO ' if has_to else 'n/a')}"
          f"  ({to_d:+.2%})" if has_to else "")
    print(f"   Turnover ratio acceptable (< 0.85x)?                               "
          f"{'YES' if to_ratio_acceptable else ('NO ' if has_to else 'n/a')}"
          f"  ({to_r:.3f}x)" if has_to else "")
    if crisis_mdd_deltas:
        print(f"   Crisis MaxDD does not materially worsen (none > -2pp)?             "
              f"{'YES' if crisis_mdd_ok else 'NO -- REJECT'}")
        for lbl, d in crisis_mdd_deltas:
            print(f"     {lbl}: dMaxDD={d:+.2%}")
    else:
        print(f"   Crisis period MaxDD check: insufficient OOS coverage (check above)")

    print()
    accept = (
        (perf_approx_flat or not perf_materially_worse)
        and to_materially_reduced
        and to_ratio_acceptable
        and (crisis_mdd_ok if crisis_mdd_deltas else True)
    )

    if crisis_mdd_worsens:
        verdict = "REJECT -- crisis-period MaxDD worsens materially. Filter suppresses defensive rebalancing."
        implication = (
            "The tolerance filter prevents timely risk-off reallocations during stress. "
            "The 1.5% threshold is too high for defensive assets where weight moves matter. "
            "Consider abandoning tolerance-band approach entirely."
        )
    elif perf_materially_worse:
        verdict = "REJECT -- performance degrades materially despite turnover reduction."
        implication = (
            "The vol-scaling adjustments that are suppressed are load-bearing for returns. "
            "The cost of suppressing these rebalances outweighs the transaction cost savings. "
            "Consider lower tau (0.5%) or accepting current turnover level."
        )
    elif accept:
        verdict = "ACCEPT -- turnover reduced materially with approximately flat performance. Upgrade baseline."
        implication = (
            f"Adopt tolerance=0.015 as the new execution parameter. "
            f"Document in PROJECT_CONTEXT.md. Full CAGR/Sharpe/MaxDD/TO metrics above become new baseline."
        )
    elif to_materially_reduced and not perf_materially_worse:
        verdict = "CONDITIONAL PASS -- turnover reduced, performance mixed. Consider full-run sensitivity."
        implication = (
            "Run the same experiment at tau=0.010 to test whether a lower threshold "
            "captures more suppression benefit with less performance impact."
        )
    else:
        verdict = "REJECT -- no material benefit."
        implication = "Tolerance-band filter does not improve the risk-adjusted profile. Move to next experiment."

    print(f"   FINAL VERDICT: {verdict}")
    print()
    print(f"   IMPLICATION: {implication}")
    print()

    # Defensive rebalancing check summary
    print("   Crisis responsiveness assessment:")
    print("   Key question: does tau=1.5% suppress defensive moves to risk-off during market stress?")
    if crisis_mdd_deltas:
        if crisis_mdd_worsens:
            print("   ANSWER: YES -- MaxDD worsens in crisis periods. Reject on this basis.")
        else:
            print("   ANSWER: NO -- crisis-period MaxDD is not materially affected by filter.")
            print("   The 63-day vol window generates large enough weight changes during crises")
            print("   that most defensive rebalances clear the 1.5% threshold.")
    else:
        print("   ANSWER: Cannot confirm -- OOS coverage is insufficient to check 2020/2022.")
        print("   Walk-forward requires more history or shorter min_train to reach these years.")
        print("   Recommend checking subperiod results above for any available 2020/2022 segments.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
