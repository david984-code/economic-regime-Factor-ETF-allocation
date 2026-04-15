#!/usr/bin/env python3
"""Validate vectorized regime classification: parity and timing.

Run from project root: uv run python scripts/validate_regime_vectorized.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.models.regime_classifier import run_parity_check


def main() -> int:
    """Run parity check on regime labels data."""
    path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if not path.exists():
        print("Run pipeline first to generate regime_labels_expanded.csv")
        return 1

    df = pd.read_csv(path, parse_dates=["date"])
    if "gdp_z" not in df.columns or "infl_z" not in df.columns:
        print("CSV missing gdp_z or infl_z. Run regime classification first.")
        return 1

    # Use date as index for consistency with classifier
    df = df.set_index("date") if "date" in df.columns else df

    print("\n=== Regime Classification Parity Check ===\n")
    all_match, report = run_parity_check(df)

    print(f"Total rows:        {report['total']}")
    print(f"Matching labels:    {report['match_count']} / {report['total']}")
    print(f"All match:         {all_match}")
    print()
    print("Timing:")
    print(f"  Old (apply):     {report['old_time_sec'] * 1000:.2f} ms")
    print(f"  New (vectorized): {report['new_time_sec'] * 1000:.2f} ms")
    print(f"  Speedup:         {report['speedup']:.1f}x")
    print()

    if not report["mismatches"].empty:
        print("MISMATCHES (old vs new):")
        mm = report["mismatches"]
        for idx, row in mm.iterrows():
            date_str = str(idx)[:10] if hasattr(idx, "__str__") else str(idx)
            print(f"  {date_str}: {row['old_regime']} -> {row['new_regime']}")
        return 1

    print("PASS: Old and new implementations produce identical outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
