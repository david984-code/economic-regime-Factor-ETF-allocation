"""Show exactly what time period the backtest metrics cover."""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import pandas as pd

from src.evaluation.walk_forward import run_walk_forward_evaluation


def overlaps(row, y0, y1):
    ts = pd.Period(row["test_start"], freq="M").year
    te = pd.Period(row["test_end"], freq="M").year
    return ts <= y1 and te >= y0


def main():
    print("Running full walk-forward (this takes ~2 minutes)...")
    df = run_walk_forward_evaluation(
        start="2010-01-01",
        end=None,
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=False,
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
        tolerance=0.015,
    )

    segs = df[df["segment"] != "OVERALL"].copy()
    overall = df[df["segment"] == "OVERALL"].iloc[0]

    first_oos = pd.Period(segs["test_start"].iloc[0], freq="M")
    last_oos = pd.Period(segs["test_end"].iloc[-1], freq="M")
    n_months = (last_oos - first_oos).n + 1

    print()
    print("=" * 55)
    print("WALK-FORWARD STRUCTURE")
    print("=" * 55)
    print("  Training data starts:   2010-01-01")
    print("  Min training required:  60 months  (5 years)")
    print(f"  First OOS window:       {segs['test_start'].iloc[0]}")
    print(f"  Last OOS window:        {segs['test_end'].iloc[-1]}")
    print(f"  Total OOS segments:     {len(segs)}")
    print("  Each segment length:    12 months")
    print(f"  OOS span:               {n_months} months  (~{n_months / 12:.1f} years)")
    print()
    print("  How it works:")
    print("  - Model trains on all data up to a cutoff date")
    print("  - It is then tested on the NEXT 12 months it has never seen")
    print("  - The cutoff advances by 12 months and repeats")
    print("  - CAGR/Sharpe are measured across all these OOS windows")
    print("  - No look-ahead bias: future data never influences past decisions")

    print()
    print("=" * 55)
    print("ACCEPTED BASELINE METRICS  (out-of-sample only)")
    print("=" * 55)
    print(f"  CAGR:     {overall.Strategy_CAGR:.2%}  per year")
    print(f"  Sharpe:   {overall.Strategy_Sharpe:.3f}")
    print(f"  MaxDD:    {overall.Strategy_MaxDD:.2%}")
    print(f"  Vol:      {overall.Strategy_Vol:.2%}  annualized")
    print(f"  Turnover: {overall.Strategy_Turnover:.2%}  per year")

    print()
    print("=" * 55)
    print("SUBPERIOD BREAKDOWN  (average metrics per era)")
    print("=" * 55)
    print(f"  {'Period':<14} {'Segs':>5} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("  " + "-" * 48)
    for label, y0, y1 in [
        ("2015-2017", 2015, 2017),
        ("2018-2020", 2018, 2020),
        ("2021-2022", 2021, 2022),
        ("2023-now", 2023, 2030),
    ]:
        sub = segs[segs.apply(overlaps, axis=1, args=(y0, y1))]
        if len(sub) == 0:
            continue
        print(
            f"  {label:<14} {len(sub):>5} "
            f"{sub.Strategy_CAGR.mean():>8.1%} "
            f"{sub.Strategy_Sharpe.mean():>8.2f} "
            f"{sub.Strategy_MaxDD.mean():>8.1%}"
        )

    print()
    print("  Key: 2021-2022 is the weak period (rate shock; signal was slow to respond)")
    print("  Next experiment targets this: 200-Day SMA Crash Gate")


if __name__ == "__main__":
    main()
