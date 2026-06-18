"""Print the current live portfolio allocation from the accepted baseline model.

Signal pipeline:
  SPY 24M momentum -> expanding z-score -> sigmoid(z * 0.25) -> risk_on
  Equal-weight sleeves -> blend -> inverse-vol scale (VOL_LOOKBACK=63)
  Tolerance filter tau=0.015 applies at rebalance vs prior holdings (not shown here).
"""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import pandas as pd

from src.allocation.vol_scaling import vol_scaled_weights_from_std
from src.config import RISK_OFF_ASSETS_BASE, RISK_ON_ASSETS_BASE, TICKERS, VOL_LOOKBACK
from src.data.market_ingestion import fetch_prices
from src.features.transforms import sigmoid


def main():
    prices = fetch_prices(start="2010-01-01", end=None)
    spy_monthly = prices["SPY"].resample("ME").last()

    # --- Signal pipeline (exact baseline) ---
    n = len(spy_monthly)
    lookback = 24
    raw = [
        spy_monthly.iloc[i] / spy_monthly.iloc[i - lookback] - 1 if i >= lookback else float("nan")
        for i in range(n)
    ]
    raw_s = pd.Series(raw, index=spy_monthly.index)

    min_hist = max(lookback, 12)
    z_vals = []
    for i in range(n):
        trailing = raw_s.iloc[: i + 1].dropna()
        if len(trailing) >= min_hist:
            z_vals.append((raw_s.iloc[i] - trailing.mean()) / trailing.std())
        else:
            z_vals.append(0.0)
    z_s = pd.Series(z_vals, index=spy_monthly.index)
    risk_on_series = sigmoid(z_s * 0.25)

    latest_me = spy_monthly.index[-1]
    raw_mom = float(raw_s.iloc[-1])
    z_score = float(z_s.iloc[-1])
    risk_on = float(risk_on_series.iloc[-1])

    # --- Inverse-vol scaling (63-day rolling std) ---
    rolling_std = prices[TICKERS].rolling(VOL_LOOKBACK, min_periods=1).std()
    std_row = rolling_std.iloc[-1]
    std_dict = {
        a: float(std_row[a]) if a in std_row.index and pd.notna(std_row[a]) else None
        for a in TICKERS
    }

    # Equal-weight base per sleeve -> blend by risk_on
    n_ro = len(RISK_ON_ASSETS_BASE)
    n_rof = len(RISK_OFF_ASSETS_BASE)
    blended = {}
    for a in RISK_ON_ASSETS_BASE:
        blended[a] = risk_on * (1.0 / n_ro)
    for a in RISK_OFF_ASSETS_BASE:
        blended[a] = (1.0 - risk_on) * (1.0 / n_rof)
    total = sum(blended.values())
    blended = {a: v / total for a, v in blended.items()}

    # Inverse-vol scale (final target weights)
    all_assets = RISK_ON_ASSETS_BASE + RISK_OFF_ASSETS_BASE
    w_final = vol_scaled_weights_from_std(blended, std_dict, all_assets)

    # --- Output ---
    print("=" * 50)
    print("CURRENT PORTFOLIO ALLOCATION")
    print(f"Signal date:  {latest_me.strftime('%Y-%m-%d')} (last month-end)")
    print(f"Price data:   through {prices.index[-1].date()}")
    print(f"VOL_LOOKBACK: {VOL_LOOKBACK} days")
    print("=" * 50)

    print()
    print("SIGNAL")
    print(f"  24M SPY momentum:  {raw_mom:+.2%}")
    print(f"  Expanding z-score: {z_score:+.3f}")
    print(f"  risk_on:           {risk_on:.4f}")
    print(
        f"  Interpretation:    {risk_on * 100:.1f}% risk-on  /  {(1 - risk_on) * 100:.1f}% risk-off"
    )

    print()
    print("TARGET WEIGHTS  (inverse-vol scaled)")
    print(f"  {'Asset':<8}  {'Sleeve':<10}  {'Weight':>8}")
    print("  " + "-" * 32)
    for a in RISK_ON_ASSETS_BASE:
        print(f"  {a:<8}  {'risk-on':<10}  {w_final.get(a, 0.0):>8.2%}")
    print("  " + "-" * 32)
    for a in RISK_OFF_ASSETS_BASE:
        print(f"  {a:<8}  {'risk-off':<10}  {w_final.get(a, 0.0):>8.2%}")
    print("  " + "-" * 32)

    ro_total = sum(w_final.get(a, 0.0) for a in RISK_ON_ASSETS_BASE)
    rof_total = sum(w_final.get(a, 0.0) for a in RISK_OFF_ASSETS_BASE)
    total_w = sum(w_final.values())
    print(f"  {'Risk-on total':<18}  {ro_total:>8.2%}")
    print(f"  {'Risk-off total':<18}  {rof_total:>8.2%}")
    print(f"  {'TOTAL':<18}  {total_w:>8.2%}")

    print()
    print("NOTE: The tolerance filter (tau=1.5%) applies at the next monthly")
    print("  rebalance and suppresses trades smaller than 1.5% vs current")
    print("  holdings. The weights above are the raw target before filtering.")


if __name__ == "__main__":
    main()
