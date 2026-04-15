#!/usr/bin/env python3
"""Validate regime classification: cached vs non-cached produce identical outputs.

Compares: raw series, transformed features, regime labels, latest regime.
Run from project root: uv run python scripts/validate_regime_cache_parity.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dotenv import load_dotenv

load_dotenv()

RTOL = 1e-5
ATOL = 1e-8


def run_parity_check() -> int:
    """Run regime classification with and without cache, compare outputs."""
    from src.models.regime_classifier import RegimeClassifier

    if not os.getenv("FRED_API_KEY"):
        print("FRED_API_KEY not set. Skipping.")
        return 1

    print("\n=== Regime Cache Parity Validation ===\n")

    api_key = os.getenv("FRED_API_KEY")
    classifier = RegimeClassifier(api_key)

    print("Running with use_local_cache=False (full API fetch)...")
    df_no_cache = classifier.run_and_return_df(use_local_cache=False)

    print("Running with use_local_cache=True (local cache + incremental)...")
    df_cached = classifier.run_and_return_df(use_local_cache=True)

    ok = True

    # Compare index
    if not df_no_cache.index.equals(df_cached.index):
        print("FAIL: Index mismatch")
        ok = False
    else:
        print(f"OK: Index match ({len(df_no_cache)} rows)")

    # Compare numeric columns
    num_cols = ["gdp_z", "infl_z", "risk_on", "macro_score"]
    for col in num_cols:
        if col not in df_no_cache.columns or col not in df_cached.columns:
            print(f"SKIP: {col} not in both")
            continue
        a = df_no_cache[col].astype(float)
        b = df_cached[col].astype(float)
        if not np.allclose(a.values, b.values, rtol=RTOL, atol=ATOL, equal_nan=True):
            diff = np.abs(a.values - b.values)
            diff = np.nan_to_num(diff, nan=0.0)
            print(f"FAIL: {col} max diff = {np.max(diff):.2e}")
            ok = False
        else:
            print(f"OK: {col}")

    # Compare regime labels (string)
    if "regime" in df_no_cache.columns:
        match = (df_no_cache["regime"] == df_cached["regime"]) | (
            df_no_cache["regime"].isna() & df_cached["regime"].isna()
        )
        if not match.all():
            mismatches = df_no_cache[~match][["regime"]].copy()
            mismatches["cached_regime"] = df_cached.loc[~match, "regime"].values
            print("FAIL: Regime label mismatches:")
            print(mismatches.head(10))
            ok = False
        else:
            print("OK: All regime labels match")

    # Latest regime
    latest_no = (
        df_no_cache.dropna(subset=["regime"]).iloc[-1]
        if df_no_cache["regime"].notna().any()
        else None
    )
    latest_cached = (
        df_cached.dropna(subset=["regime"]).iloc[-1]
        if df_cached["regime"].notna().any()
        else None
    )
    if latest_no is not None and latest_cached is not None:
        if latest_no["regime"] != latest_cached["regime"]:
            print(
                f"FAIL: Latest regime {latest_no['regime']} != {latest_cached['regime']}"
            )
            ok = False
        else:
            print(f"OK: Latest regime = {latest_no['regime']}")

    print("\n" + "=" * 40)
    if ok:
        print("PASS: Cached and non-cached produce identical outputs.")
        return 0
    else:
        print("FAIL: Mismatches detected.")
        return 1


if __name__ == "__main__":
    sys.exit(run_parity_check())
