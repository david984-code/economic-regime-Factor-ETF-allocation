"""Run the walk-forward backtest using vintage (point-in-time) regime labels.

Usage:
    python scripts/run_vintage_walk_forward.py [--no-restore] [--out PATH]

This is the actual moment-of-truth: the WF backtest with revised-data labels
gave the published CAGR/Sharpe/MaxDD numbers. Re-running with vintage labels
quantifies how much of the apparent edge was lookahead.

Steps:
  1. Back up the existing regime_labels_expanded.csv (revised-data labels).
  2. Build a vintage-aware regime_labels_expanded.csv by:
     - Taking outputs/regime_labels_vintage.csv (2010-2026 monthly vintage)
     - Padding pre-2010 with "Unknown" / risk_on=0.5 to match expected schema
     - Forward-filling monthly labels to daily within each month
  3. Run collect_walk_forward_oos_returns over the same window.
  4. Compute metrics (CAGR / Sharpe / Sortino / MaxDD / Calmar).
  5. Restore the original CSV.
  6. Save the vintage OOS daily series to scripts/analysis/output/.

Output: scripts/analysis/output/vintage_wf_daily_returns.csv
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUTS_DIR

EXPANDED = OUTPUTS_DIR / "regime_labels_expanded.csv"
ASOF = OUTPUTS_DIR / "regime_labels_asof.csv"
VINTAGE_MONTHLY = OUTPUTS_DIR / "regime_labels_vintage.csv"

OUT = Path(r"C:\Users\dns81\Quant\scripts\analysis\output")
OUT.mkdir(parents=True, exist_ok=True)

RF_DAILY = (1 + 0.045) ** (1 / 252) - 1


def build_vintage_expanded() -> pd.DataFrame:
    """Build a vintage-aware regime_labels_expanded.csv matching the existing schema."""
    vint = pd.read_csv(VINTAGE_MONTHLY, index_col="date", parse_dates=True)
    # Read existing expanded CSV to get the index range and schema
    existing = pd.read_csv(EXPANDED, index_col="date", parse_dates=True)

    # Reindex vintage to expanded's full index, then ffill monthly labels to daily
    new_df = existing.copy()
    # Wipe regime + risk_on for in-range dates (2010-01 onward) and refill with vintage
    cutoff = vint.index.min()
    in_range = new_df.index >= cutoff
    new_df.loc[in_range, "regime"] = np.nan
    new_df.loc[in_range, "risk_on"] = np.nan

    # Place vintage values at month-end, then ffill within month
    for d, row in vint.iterrows():
        # Find rows in expanded between d and the next vintage date
        idx_mask = (new_df.index >= d) & (new_df.index < d + pd.offsets.MonthEnd(1))
        new_df.loc[idx_mask, "regime"] = row["regime"]
        new_df.loc[idx_mask, "risk_on"] = row["risk_on"]

    # Forward-fill any gaps
    new_df["regime"] = new_df["regime"].ffill()
    new_df["risk_on"] = new_df["risk_on"].ffill()
    # Pre-cutoff rows that were "Unknown" stay as they were
    return new_df


def compute_metrics(daily: pd.Series) -> dict:
    r = daily.dropna()
    n_years = len(r) / 252.0
    cagr_v = (1 + r).prod() ** (1 / n_years) - 1
    vol_v = r.std(ddof=1) * np.sqrt(252)
    excess = r - RF_DAILY
    sharpe_v = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    dn = excess[excess < 0]
    sortino_v = (
        excess.mean() / np.sqrt((dn**2).mean()) * np.sqrt(252) if len(dn) > 1 else float("nan")
    )
    w = (1 + r).cumprod()
    mdd = (w / w.cummax() - 1).min()
    calmar_v = cagr_v / abs(mdd) if mdd != 0 else float("nan")
    return {
        "CAGR_pct": round(float(cagr_v) * 100, 2),
        "Vol_pct": round(float(vol_v) * 100, 2),
        "Sharpe": round(float(sharpe_v), 3),
        "Sortino": round(float(sortino_v), 3),
        "MaxDD_pct": round(float(mdd) * 100, 2),
        "Calmar": round(float(calmar_v), 3),
        "n_days": int(len(r)),
        "n_years": round(n_years, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-restore",
        action="store_true",
        help="Leave vintage labels in place after run (default: restore revised labels).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT / "vintage_wf_daily_returns.csv",
        help="Output CSV path for vintage walk-forward daily returns.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # 1. Backup
    backup = EXPANDED.with_suffix(".csv.bak_revised")
    log.info("Backing up revised labels: %s -> %s", EXPANDED.name, backup.name)
    shutil.copy(EXPANDED, backup)
    asof_backup = ASOF.with_suffix(".csv.bak_revised")
    if ASOF.exists():
        shutil.copy(ASOF, asof_backup)

    try:
        # 2. Build & write vintage labels
        log.info("Building vintage-aware regime_labels_expanded.csv ...")
        vintage_expanded = build_vintage_expanded()
        vintage_expanded.to_csv(EXPANDED)
        if ASOF.exists():
            vintage_expanded.to_csv(ASOF)
        log.info("  Wrote %d daily rows", len(vintage_expanded))

        # 3. Run walk-forward
        log.info("Running walk-forward OOS with vintage labels ...")
        from src.config import BASELINE_WALK_FORWARD
        from src.evaluation.walk_forward import collect_walk_forward_oos_returns

        t0 = time.time()
        oos_daily = collect_walk_forward_oos_returns(use_vol_regime=False, **BASELINE_WALK_FORWARD)
        log.info("  WF complete in %.1fs (%d daily obs)", time.time() - t0, len(oos_daily))

        # 4. Save & compute metrics
        out_csv = args.out
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        oos_daily.to_csv(out_csv, header=["strategy_vintage_wf"])
        log.info("  Saved daily series -> %s", out_csv)

        m = compute_metrics(oos_daily)
        print()
        print("=" * 70)
        print("  VINTAGE WALK-FORWARD METRICS  (vs published revised-data numbers)")
        print("=" * 70)
        print()
        print(f"  Window: {oos_daily.index.min().date()} to {oos_daily.index.max().date()}")
        print(f"  N days: {m['n_days']} ({m['n_years']} years)")
        print()
        print(f"  {'Metric':<15} {'Vintage':>12} {'Published':>12} {'Delta':>10}")
        print(f"  {'-' * 50}")
        ref = {
            "CAGR_pct": 10.99,
            "Vol_pct": 9.98,
            "Sharpe": 0.654,
            "Sortino": 0.625,
            "MaxDD_pct": -18.27,
            "Calmar": 0.602,
        }
        for k in ["CAGR_pct", "Vol_pct", "Sharpe", "Sortino", "MaxDD_pct", "Calmar"]:
            delta = m[k] - ref[k]
            sign = "+" if delta >= 0 else ""
            print(f"  {k:<15} {m[k]:>12} {ref[k]:>12} {sign}{delta:>9.2f}")

    finally:
        # 5. Restore original revised labels for safety unless --no-restore
        if args.no_restore:
            log.info("--no-restore: leaving vintage labels in place.")
        else:
            log.info("Restoring revised labels from backup ...")
            shutil.copy(backup, EXPANDED)
            if asof_backup.exists():
                shutil.copy(asof_backup, ASOF)
            log.info("  Restored.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
