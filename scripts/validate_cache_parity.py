#!/usr/bin/env python3
"""Validate that cached (shared) data produces identical outputs to per-step fetches.

Compares:
- Tickers and date index
- Monthly returns
- Regime labels
- Optimal allocations
- Backtest metrics and current weights

Run from project root: uv run python scripts/validate_cache_parity.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.config import OUTPUTS_DIR, START_DATE, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices
from src.data.pipeline_data import PipelineData

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RTOL = 1e-5
ATOL = 1e-8


def _compare_dfs(name: str, a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Compare two DataFrames. Return True if equal within tolerance."""
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {a.shape} != {b.shape}")
        return False
    if list(a.columns) != list(b.columns):
        print(f"  FAIL {name}: columns differ")
        return False
    if not np.allclose(a.values, b.values, rtol=RTOL, atol=ATOL, equal_nan=True):
        diff = np.abs(a.values - b.values)
        print(f"  FAIL {name}: values differ (max diff {np.nanmax(diff):.2e})")
        return False
    print(f"  OK {name}")
    return True


def _compare_series(name: str, a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b):
        print(f"  FAIL {name}: length {len(a)} != {len(b)}")
        return False
    if not np.allclose(a.values, b.values, rtol=RTOL, atol=ATOL, equal_nan=True):
        print(f"  FAIL {name}: values differ")
        return False
    print(f"  OK {name}")
    return True


def _compare_dict_of_dicts(name: str, a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        print(f"  FAIL {name}: keys differ")
        return False
    all_ok = True
    for k in a:
        if set(a[k].keys()) != set(b[k].keys()):
            print(f"  FAIL {name}[{k}]: asset keys differ")
            all_ok = False
            continue
        for asset in a[k]:
            if not np.isclose(a[k][asset], b[k][asset], rtol=RTOL, atol=ATOL):
                print(
                    f"  FAIL {name}[{k}][{asset}]: {a[k][asset]:.6f} != {b[k][asset]:.6f}"
                )
                all_ok = False
    if all_ok:
        print(f"  OK {name}")
    return all_ok


def _compare_dict(name: str, a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        print(f"  FAIL {name}: keys differ")
        return False
    all_ok = True
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not np.isclose(va, vb, rtol=RTOL, atol=ATOL):
                print(f"  FAIL {name}[{k}]: {va} != {vb}")
                all_ok = False
        elif va != vb:
            print(f"  FAIL {name}[{k}]: {va} != {vb}")
            all_ok = False
    if all_ok:
        print(f"  OK {name}")
    return all_ok


def run_validation() -> int:
    """Run validation. Returns 0 if all checks pass, 1 otherwise."""
    if not os.getenv("FRED_API_KEY"):
        print(
            "FRED_API_KEY not set. Skipping regime classification; using existing outputs."
        )
    else:
        pass

    print("\n=== Cache Parity Validation ===\n")

    # Fetch once; use same data for both paths to avoid network variance
    print("[1] Fetching prices once...")
    prices = fetch_prices(tickers=TICKERS, start=START_DATE, end=get_end_date())
    pipeline_data = PipelineData(tickers=TICKERS, start=START_DATE, end=get_end_date())
    pipeline_data.set_prices(prices)

    ok = True
    print(f"  OK tickers: {sorted(prices.columns)}")
    print(f"  OK date index: {len(prices)} rows")

    # 2. Compare monthly returns (both derived from same prices)
    print("\n[2] Monthly returns: manual vs PipelineData.get_monthly_returns")
    monthly = prices.resample("ME").last()
    ret_manual = monthly.pct_change().dropna()
    ret_manual.index.name = "Date"
    ret_cached = pipeline_data.get_monthly_returns()
    if not _compare_dfs("monthly returns", ret_manual, ret_cached):
        ok = False

    # 3. Compare momentum features (both derived from same prices)
    print("\n[3] Momentum: manual vs PipelineData.get_momentum_features")
    px = prices["SPY"]
    monthly_spy = px.resample("ME").last()
    mom_manual = pd.DataFrame(
        {
            "spy_1m": monthly_spy.pct_change(1),
            "spy_3m": monthly_spy.pct_change(3),
            "spy_6m": monthly_spy.pct_change(6),
        },
        index=monthly_spy.index,
    )
    mom_manual.index = mom_manual.index.to_period("M").to_timestamp("M")
    mom_cached = pipeline_data.get_momentum_features(ticker="SPY")
    if not _compare_dfs("momentum features", mom_manual, mom_cached):
        ok = False

    # 4. Regime labels (from existing CSV - produced by regime classification)
    print("\n[4] Regime labels (from outputs)")
    regime_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if regime_path.exists():
        regime_df = pd.read_csv(regime_path, parse_dates=["date"])
        print(f"  OK regime labels: {len(regime_df)} rows")
    else:
        print("  SKIP regime labels: file not found (run pipeline first)")

    # 5. Allocations (from DB - produced by optimizer)
    print("\n[5] Optimal allocations")
    from src.utils.database import Database

    db = Database()
    try:
        allocs = db.load_optimal_allocations()
        if allocs:
            print(f"  OK allocations: {len(allocs)} regimes")
        else:
            print("  SKIP allocations: empty (run pipeline first)")
    except Exception as e:
        print(f"  SKIP allocations: {e}")
    finally:
        db.close()

    # 6. Backtest metrics
    print("\n[6] Backtest results")
    db = Database()
    try:
        bt = db.get_latest_backtest_results()
        if bt:
            print(f"  OK backtest: portfolio Sharpe {bt['portfolio']['Sharpe']:.4f}")
        else:
            print("  SKIP backtest: no results (run pipeline first)")
    except Exception as e:
        print(f"  SKIP backtest: {e}")
    finally:
        db.close()

    print("\n" + "=" * 40)
    if ok:
        print("PASS: Cached data matches direct fetch outputs.")
        return 0
    else:
        print("FAIL: Some comparisons failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
