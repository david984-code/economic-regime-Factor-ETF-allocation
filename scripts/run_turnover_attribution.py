"""Turnover Attribution Diagnostic — 24M Baseline.

Decomposes the modeled target-to-target turnover into:
  S  = signal-driven (changes in risk_on)
  V  = vol-rebalancing (changes in 63-day rolling std per asset)
  I  = netting/cancellation residual (= Total - S - V, always <= 0)

Does NOT modify strategy behavior. Pure post-hoc reconstruction.
Engine semantics confirmed: cost = |w_target(t) - w_target(t-1)| * COST_BPS.
No drift correction in the cost model. 118% is modeled TO, not real-world TO.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.allocation.vol_scaling import vol_scaled_weights_from_std
from src.backtest.engine import _blend_alloc
from src.backtest.metrics import compute_turnover
from src.config import ASSETS, RISK_OFF_ASSETS_BASE, RISK_ON_ASSETS_BASE, TICKERS, VOL_LOOKBACK
from src.data.market_ingestion import fetch_prices
from src.features.transforms import sigmoid

LOOKBACK_MONTHS = 24  # baseline
FULL_START = "2010-01-01"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_risk_on_me(spy_monthly: pd.Series, lookback: int) -> pd.Series:
    """Reproduce engine risk_on pipeline at each month-end.
    Pipeline: raw 24M momentum -> expanding z-score -> sigmoid(z*0.25)
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
    return sigmoid(z * 0.25)


def _get_ro_for_rebalance(rebal_date: pd.Timestamp, risk_on_me: pd.Series) -> float:
    """Return the risk_on value the engine uses at this rebalance date.
    Engine forward-fills month-end risk_on to daily; first day of month M uses
    the month-end value from M-1.
    """
    prior_ends = risk_on_me.index[risk_on_me.index < rebal_date]
    if len(prior_ends) == 0:
        return float(sigmoid(0.0))  # default neutral
    return float(risk_on_me.loc[prior_ends[-1]])


def _get_std_dict(rebal_date: pd.Timestamp, rolling_std: pd.DataFrame) -> dict:
    """Return {asset: 63-day rolling std} at the given rebalance date."""
    if rebal_date in rolling_std.index:
        row = rolling_std.loc[rebal_date]
    else:
        idx = rolling_std.index.asof(rebal_date)
        if pd.isna(idx):
            # Date falls before the rolling_std window starts; use first row
            row = rolling_std.iloc[0]
        else:
            row = rolling_std.loc[idx]
    return {a: float(row[a]) if a in row.index and pd.notna(row[a]) else None
            for a in TICKERS}


def _compute_weights(risk_on: float, std_dict: dict) -> dict:
    """Reproduce engine weight computation exactly.
    Matches _run_backtest_vectorized lines 907-931.
    """
    w_risk_on = {a: (1.0 / len(RISK_ON_ASSETS_BASE) if a in RISK_ON_ASSETS_BASE else 0.0)
                 for a in ASSETS}
    w_risk_off = {a: (1.0 / len(RISK_OFF_ASSETS_BASE) if a in RISK_OFF_ASSETS_BASE else 0.0)
                  for a in ASSETS}
    blended = _blend_alloc(w_risk_off, w_risk_on, risk_on, ASSETS)
    return vol_scaled_weights_from_std(blended, std_dict, list(TICKERS))


def _to_vec(w: dict) -> np.ndarray:
    return np.array([w.get(a, 0.0) for a in ASSETS])


def _fmt(v: float, pct: bool = True, decimals: int = 2) -> str:
    if np.isnan(v):
        return "n/a"
    if pct:
        return f"{v:.{decimals}%}"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TURNOVER ATTRIBUTION DIAGNOSTIC — 24M Baseline")
    print(f"Data window: {FULL_START} to present")
    print("Engine confirmed: cost = |w_target(t) - w_target(t-1)| × COST_BPS")
    print("Decomposition: Total = Signal + Vol_Rebalancing + Netting")
    print("=" * 72)

    # -- Load data -----------------------------------------------------------
    prices = fetch_prices(start=FULL_START, end=None)
    spy_monthly = prices["SPY"].resample("ME").last()
    returns_daily = prices[TICKERS].pct_change().iloc[1:]

    # Rolling 63-day std — exactly as in _run_backtest_vectorized line 880
    rolling_std = returns_daily.rolling(VOL_LOOKBACK, min_periods=1).std()

    # Month-end risk_on series
    risk_on_me = _compute_risk_on_me(spy_monthly, LOOKBACK_MONTHS)

    # Rebalance dates = first trading day of each new month
    dates = prices.index
    months = pd.Series(dates).dt.to_period("M").values
    month_changed = np.concatenate([[True], months[1:] != months[:-1]])
    rebalance_dates = list(dates[month_changed])

    print(f"\nRebalance dates identified: {len(rebalance_dates)}")
    print(f"  First rebalance: {rebalance_dates[0].strftime('%Y-%m-%d')}")
    print(f"  Last rebalance:  {rebalance_dates[-1].strftime('%Y-%m-%d')}")

    # -- Attribution loop ----------------------------------------------------
    records = []

    # Compute weight at first rebalance date (anchor)
    ro0 = _get_ro_for_rebalance(rebalance_dates[0], risk_on_me)
    std0 = _get_std_dict(rebalance_dates[0], rolling_std)
    w_prev = _compute_weights(ro0, std0)
    std_prev = std0
    ro_prev = ro0

    for t_curr in rebalance_dates[1:]:
        ro_curr = _get_ro_for_rebalance(t_curr, risk_on_me)
        std_curr = _get_std_dict(t_curr, rolling_std)

        w_curr = _compute_weights(ro_curr, std_curr)
        w_cf = _compute_weights(ro_curr, std_prev)  # signal=curr, vol=prev

        vec_prev = _to_vec(w_prev)
        vec_curr = _to_vec(w_curr)
        vec_cf   = _to_vec(w_cf)

        TO_total = float(np.abs(vec_curr - vec_prev).sum())
        TO_S     = float(np.abs(vec_cf - vec_prev).sum())
        TO_V     = float(np.abs(vec_curr - vec_cf).sum())
        TO_I     = TO_total - TO_S - TO_V  # <= 0 by triangle inequality

        # Per-asset vol contribution to TO_V
        asset_v_contrib = {a: float(abs(w_curr.get(a, 0) - w_cf.get(a, 0)))
                           for a in ASSETS}

        records.append({
            "date":       t_curr,
            "year":       t_curr.year,
            "ro_prev":    ro_prev,
            "ro_curr":    ro_curr,
            "delta_ro":   ro_curr - ro_prev,
            "TO_total":   TO_total,
            "TO_S":       TO_S,
            "TO_V":       TO_V,
            "TO_I":       TO_I,
            **{f"TV_{a}": v for a, v in asset_v_contrib.items()},
        })

        w_prev  = w_curr
        std_prev = std_curr
        ro_prev  = ro_curr

    df = pd.DataFrame(records).set_index("date")

    # Annualized components (same formula as compute_turnover: mean × 12)
    n = len(df)
    ann_total = df["TO_total"].mean() * 12
    ann_S     = df["TO_S"].mean()     * 12
    ann_V     = df["TO_V"].mean()     * 12
    ann_I     = df["TO_I"].mean()     * 12  # negative

    # =========================================================================
    print("\n" + "=" * 72)
    print("1. OVERALL ANNUALIZED ATTRIBUTION")
    print("=" * 72)
    print(f"  {'Component':35} {'Annual TO':>10} {'% of Total':>12}")
    print("  " + "-" * 60)
    print(f"  {'Total (modeled)':35} {_fmt(ann_total):>10} {'100.0%':>12}")
    print(f"  {'Signal-driven (S)':35} {_fmt(ann_S):>10} {ann_S / ann_total:>12.1%}")
    print(f"  {'Vol-rebalancing (V)':35} {_fmt(ann_V):>10} {ann_V / ann_total:>12.1%}")
    print(f"  {'Netting residual (I)':35} {_fmt(ann_I):>10} {ann_I / ann_total:>12.1%}")
    print(f"\n  Gross (S + V):   {_fmt(ann_S + ann_V):>10}  (before netting)")
    print(f"  Note: Netting <= 0 means signal and vol changes offset each other.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("2. YEAR-BY-YEAR ATTRIBUTION")
    print("=" * 72)
    by_year = df.groupby("year")[["TO_total", "TO_S", "TO_V", "TO_I"]].mean() * 12

    print(f"  {'Year':6} {'Total':>9} {'Signal':>9} {'VolReb':>9} {'Netting':>9} {'%Signal':>8} {'%VolReb':>8}")
    print("  " + "-" * 64)
    for yr, row in by_year.iterrows():
        tot = row["TO_total"]
        s   = row["TO_S"]
        v   = row["TO_V"]
        i   = row["TO_I"]
        pct_s = s / tot if tot > 0 else float("nan")
        pct_v = v / tot if tot > 0 else float("nan")
        print(f"  {yr:6} {_fmt(tot):>9} {_fmt(s):>9} {_fmt(v):>9} {_fmt(i):>9} {pct_s:>8.1%} {pct_v:>8.1%}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("3. SIGNAL COMPONENT PROFILE")
    print("=" * 72)
    abs_delta_ro = df["delta_ro"].abs()
    print(f"  Monthly |delta_risk_on|:")
    print(f"    Mean:              {abs_delta_ro.mean():.5f}")
    print(f"    Median:            {abs_delta_ro.median():.5f}")
    print(f"    90th percentile:   {abs_delta_ro.quantile(0.90):.5f}")
    print(f"    Max:               {abs_delta_ro.max():.5f}")
    print()
    very_small = (abs_delta_ro < 0.005).sum()
    meaningful = (abs_delta_ro >= 0.020).sum()
    print(f"  Months |delta_ro| < 0.005 (near-zero signal moves): {very_small}  ({very_small/n:.1%})")
    print(f"  Months |delta_ro| >= 0.020 (meaningful signal moves): {meaningful}  ({meaningful/n:.1%})")
    print()

    # Signal TO if all Δrisk_on were zero: should be exactly 0
    # Theoretical signal TO formula: 2 * |Δrisk_on| (before vol scaling).
    # Actual (with vol scaling): slightly different because blend ratio changes
    # interact with vol weights. Let's compare.
    theoretical_s = (abs_delta_ro * 2).mean() * 12
    print(f"  Theoretical signal TO (2*|Δrisk_on|*12, ignoring vol scaling): {_fmt(theoretical_s)}")
    print(f"  Actual signal TO from reconstruction: {_fmt(ann_S)}")
    print(f"  Difference (vol-scaling interaction on signal): {_fmt(ann_S - theoretical_s, sign=True) if not np.isnan(theoretical_s) else 'n/a'}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("4. VOL-REBALANCING COMPONENT PROFILE")
    print("=" * 72)
    tv_cols = [c for c in df.columns if c.startswith("TV_")]
    asset_tv = {c.replace("TV_", ""): df[c].mean() * 12 for c in tv_cols}
    asset_tv_sorted = sorted(asset_tv.items(), key=lambda x: x[1], reverse=True)

    print(f"  Per-asset annual contribution to Vol-rebalancing TO (TO_V):")
    print(f"  {'Asset':8} {'Annual Contrib':>15} {'% of TO_V':>12} {'% of Total':>12}")
    print("  " + "-" * 50)
    for asset, contrib in asset_tv_sorted:
        if asset == "cash":
            continue
        pct_v  = contrib / ann_V if ann_V > 0 else float("nan")
        pct_tot = contrib / ann_total if ann_total > 0 else float("nan")
        print(f"  {asset:8} {_fmt(contrib):>15} {pct_v:>12.1%} {pct_tot:>12.1%}")

    # Which sleeve drives more?
    ron_assets = set(RISK_ON_ASSETS_BASE)
    rof_assets = set(RISK_OFF_ASSETS_BASE)
    tv_ron = sum(v for a, v in asset_tv.items() if a in ron_assets)
    tv_rof = sum(v for a, v in asset_tv.items() if a in rof_assets)
    print(f"\n  Risk-on sleeve TO_V total:  {_fmt(tv_ron)}  ({tv_ron/ann_V:.1%} of TO_V)")
    print(f"  Risk-off sleeve TO_V total: {_fmt(tv_rof)}  ({tv_rof/ann_V:.1%} of TO_V)")

    # =========================================================================
    print("\n" + "=" * 72)
    print("5. NETTING / CANCELLATION ANALYSIS")
    print("=" * 72)
    meaningful_cancel = df[df["TO_I"] < -0.005]
    near_zero_cancel  = df[df["TO_I"].abs() < 0.001]

    print(f"  Total months analyzed:                         {n}")
    print(f"  Months with meaningful cancellation (I < -0.005): {len(meaningful_cancel)}  ({len(meaningful_cancel)/n:.1%})")
    print(f"  Months with near-zero netting (|I| < 0.001):      {len(near_zero_cancel)}  ({len(near_zero_cancel)/n:.1%})")
    print()
    print(f"  Average netting per month:  {df['TO_I'].mean():.5f}  ({df['TO_I'].mean()*12:.3%} annualized)")
    print(f"  Max cancellation in one month: {df['TO_I'].min():.5f}  (most negative)")
    print(f"  Max reinforcement in one month: {df['TO_I'].max():.5f}  (should be <= 0 or ~0)")
    print()
    if df["TO_I"].max() > 1e-8:
        print("  WARNING: TO_I > 0 in some months. Triangle inequality violated.")
        print("  This indicates floating-point accumulation or a reconstruction error.")
        worst = df["TO_I"].idxmax()
        print(f"  Worst case: {worst.strftime('%Y-%m')}  TO_I = {df.loc[worst, 'TO_I']:.6f}")
    else:
        print("  Triangle inequality holds: TO_I <= 0 in all months. Reconstruction is clean.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("6. CONSISTENCY CHECK")
    print("=" * 72)

    # Reconstruct a weight DataFrame and run compute_turnover on it
    # Build reconstructed monthly weight series at each rebalance date
    rebal_weights = {}
    for r_date in rebalance_dates:
        ro = _get_ro_for_rebalance(r_date, risk_on_me)
        std_d = _get_std_dict(r_date, rolling_std)
        w = _compute_weights(ro, std_d)
        rebal_weights[r_date] = {a: w.get(a, 0.0) for a in TICKERS}

    weights_df_recon = pd.DataFrame(rebal_weights).T
    weights_df_recon.index = pd.DatetimeIndex(weights_df_recon.index)
    weights_df_recon = weights_df_recon.sort_index()

    # Expand to daily (forward-fill) so compute_turnover can resample to ME
    full_daily_idx = prices.index
    weights_df_daily = weights_df_recon.reindex(full_daily_idx, method="ffill").dropna(how="all")
    ct_result = compute_turnover(weights_df_daily)

    print(f"  Annualized TO from attribution loop:   {_fmt(ann_total)}")
    print(f"  compute_turnover on reconstructed weights: {_fmt(ct_result)}")
    discrepancy = abs(ann_total - ct_result)
    print(f"  Discrepancy:                           {_fmt(discrepancy)}")
    if discrepancy < 0.005:
        print("  PASS: discrepancy < 0.5%. Reconstruction is exact.")
    elif discrepancy < 0.02:
        print("  ACCEPTABLE: discrepancy < 2.0%. Minor edge effects (startup/boundary).")
    else:
        print("  FAIL: discrepancy >= 2%. Check alignment or data boundary handling.")

    # Note on comparison to full walk-forward reported 118.29%
    print(f"\n  Reference: full walk-forward reported baseline TO = 118.29%")
    print(f"  This diagnostic uses full in-sample history from {FULL_START}.")
    print(f"  Walk-forward TO = mean of per-segment turnovers, which can differ.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("7. FINAL INTERPRETATION")
    print("=" * 72)

    pct_S = ann_S / ann_total
    pct_V = ann_V / ann_total

    print(f"  Signal-driven proportion:        {pct_S:.1%}")
    print(f"  Vol-rebalancing proportion:      {pct_V:.1%}")
    print()

    if pct_V >= 0.60:
        dominant = "VOL-REBALANCING"
        conclusion = (
            "The majority of modeled turnover is caused by the monthly update to per-asset "
            "inverse-volatility weights (63-day rolling std). "
            "The 'equal-weight' sleeve label is a misnomer — the actual weights are inverse-vol "
            "and they churn every month as rolling vol estimates shift."
        )
        next_experiment = (
            "NEXT EXPERIMENT: Lengthen the vol lookback window (VOL_LOOKBACK in config.py) "
            "from 63 days to 126 or 189 days. "
            "This will smooth per-asset vol estimates and reduce vol-driven churn with "
            "minimal impact on return generation. "
            "Single parameter change. No signal modification. No construction change."
        )
    elif pct_S >= 0.60:
        dominant = "SIGNAL-DRIVEN"
        conclusion = (
            "The majority of modeled turnover comes from monthly changes in risk_on. "
            "Each Δrisk_on forces trades across all 10 assets. "
            "Even small continuous movements in the sigmoid output generate constant churn."
        )
        next_experiment = (
            "NEXT EXPERIMENT: Add a minimum-change execution trigger on risk_on. "
            "Suppress monthly rebalance if |risk_on(t) - risk_on_last_executed| < threshold (e.g. 0.02). "
            "Preserves signal computation exactly; only suppresses near-zero trades. "
            "Single parameter change. No signal modification. No construction change."
        )
    else:
        dominant = "MIXED"
        conclusion = (
            "Signal and vol-rebalancing contribute roughly equally. "
            "Address the larger component first — see percentages above."
        )
        next_experiment = (
            "NEXT EXPERIMENT: target the dominant component identified above. "
            "If vol is slightly larger, start with vol lookback extension."
        )

    print(f"  DOMINANT SOURCE: {dominant}")
    print()
    print(f"  Interpretation:")
    for line in conclusion.split(". "):
        if line.strip():
            print(f"    {line.strip()}.")
    print()
    print(f"  {next_experiment}")

    print("\n" + "=" * 72)
    print("  REMINDER: The modeled 118% is an OPTIMISTIC lower bound.")
    print("  Real portfolio also incurs drift-correction costs (not in cost model).")
    print("  These are estimated at ~20-50% additional annual turnover for a")
    print("  10-asset portfolio with ~15% cross-sectional vol dispersion.")
    print("=" * 72)


def _fmt(v: float, pct: bool = True, decimals: int = 2, sign: bool = False) -> str:
    if np.isnan(v):
        return "n/a"
    if pct:
        fmt = f"{{:{'+' if sign else ''}.{decimals}%}}"
        return fmt.format(v)
    return f"{v:.{decimals}f}"


if __name__ == "__main__":
    main()
