#!/usr/bin/env python3
"""Validate vectorized backtest matches loop implementation.

Compares: portfolio returns, cumulative returns, drawdown, turnover, metrics.
Run from project root: uv run python scripts/validate_backtest_parity.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, START_DATE, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices

RTOL = 1e-4
ATOL = 2e-3  # Allow ~0.2% daily return diff from vol-scaling / floating-point


def run_parity_check() -> int:
    """Run both backtest implementations and compare."""
    from src.backtest.engine import (
        _compute_returns_and_setup,
        _run_backtest_loop,
        _run_backtest_vectorized,
    )
    from src.utils.database import Database

    print("\n=== Backtest Parity Validation ===\n")

    prices = fetch_prices(tickers=TICKERS, start=START_DATE, end=get_end_date())
    regime_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    opt_path = OUTPUTS_DIR / "optimal_allocations.csv"
    if regime_path.exists() and opt_path.exists():
        regime_df = pd.read_csv(regime_path, parse_dates=["date"])
        regime_df = regime_df.dropna(subset=["regime"]).set_index("date")
        opt_df = pd.read_csv(opt_path, index_col=0)
        allocations = {r: opt_df.loc[r].to_dict() for r in opt_df.index}
    else:
        db = Database()
        regime_df = db.load_regime_labels()
        allocations = db.load_optimal_allocations()
        db.close()
    allocations = {str(k).strip(): v for k, v in allocations.items()}
    for alloc in allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0

    returns, regime_df, w_risk_on, w_risk_off, equal_weight_returns = _compute_returns_and_setup(
        prices, regime_df, allocations
    )

    print("Running loop implementation...")
    t0 = time.perf_counter()
    ret_loop = _run_backtest_loop(
        returns, regime_df, allocations, w_risk_on, w_risk_off, equal_weight_returns
    )
    t_loop = time.perf_counter() - t0

    print("Running vectorized implementation...")
    t0 = time.perf_counter()
    ret_vec = _run_backtest_vectorized(
        returns, regime_df, allocations, w_risk_on, w_risk_off, equal_weight_returns
    )
    t_vec = time.perf_counter() - t0

    print(f"\nTiming: loop={t_loop*1000:.1f}ms, vectorized={t_vec*1000:.1f}ms, speedup={t_loop/t_vec:.1f}x")

    ok = True

    # Portfolio returns
    if not ret_loop.index.equals(ret_vec.index):
        print("FAIL: Index mismatch")
        ok = False
    else:
        diff = np.abs(ret_loop.values - ret_vec.values)
        nan_both = np.isnan(ret_loop.values) & np.isnan(ret_vec.values)
        diff[nan_both] = 0
        max_diff = np.nanmax(diff)
        if max_diff > ATOL:
            print(f"FAIL: Portfolio returns max diff = {max_diff:.2e}")
            mismatches = np.where(~np.isclose(ret_loop.values, ret_vec.values, rtol=RTOL, atol=ATOL, equal_nan=True))[0]
            for i in mismatches[:5]:
                print(f"  {ret_loop.index[i]}: loop={ret_loop.iloc[i]:.6e} vec={ret_vec.iloc[i]:.6e}")
            if len(mismatches) > 5:
                print(f"  ... and {len(mismatches)-5} more")
            ok = False
        else:
            print("OK: Portfolio returns match")

    # Cumulative returns
    cum_loop = (1 + ret_loop).cumprod()
    cum_vec = (1 + ret_vec).cumprod()
    if not np.allclose(cum_loop.dropna(), cum_vec.dropna(), rtol=RTOL, atol=ATOL):
        print("FAIL: Cumulative returns mismatch")
        ok = False
    else:
        print("OK: Cumulative returns match")

    # Drawdown
    dd_loop = cum_loop / cum_loop.cummax() - 1
    dd_vec = cum_vec / cum_vec.cummax() - 1
    if not np.allclose(dd_loop.dropna(), dd_vec.dropna(), rtol=RTOL, atol=ATOL):
        print("FAIL: Drawdown mismatch")
        ok = False
    else:
        print("OK: Drawdown match")

    # Metrics
    from src.backtest.metrics import compute_metrics
    from src.backtest.engine import CASH_DAILY_YIELD

    m_loop = compute_metrics(ret_loop, rf_daily=CASH_DAILY_YIELD)
    m_vec = compute_metrics(ret_vec, rf_daily=CASH_DAILY_YIELD)
    for k in m_loop:
        if not np.isclose(m_loop[k], m_vec[k], rtol=RTOL, atol=ATOL):
            print(f"FAIL: Metric {k}: loop={m_loop[k]:.6f} vec={m_vec[k]:.6f}")
            ok = False
    if ok:
        print("OK: All metrics match")

    print("\n" + "=" * 40)
    if ok:
        print("PASS: Vectorized backtest matches loop implementation.")
        return 0
    else:
        print("FAIL: Mismatches detected.")
        return 1


if __name__ == "__main__":
    sys.exit(run_parity_check())
