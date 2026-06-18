"""Recompute the down-month hit rate vs 60/40 using vintage walk-forward returns.

Background: the 76.3% hit rate cited in earlier versions of the README was
computed on revised-data regime labels. After the ALFRED vintage rerun
(Appendix D in docs/bootstrap_reconciliation.md) showed vintage-vs-revised
label correlation of 0.19, the hit rate needed to be recomputed on the
vintage label stream to be apples-to-apples with the rest of the headline
table.

This script reproduces the 57.9% vintage hit rate and its cluster-aware
6-month-block 95% CI [41.9%, 73.5%] using the same methodology as
Appendix C.2 (which produced the 76.3% revised figure).

Inputs:
  --vintage-csv  daily OOS returns from scripts/run_vintage_walk_forward.py
                 (default: outputs/vintage_wf_daily_returns.csv)
  --bench-csv    daily SPY + IEF returns (any source with columns
                 spy_daily_ret, ief_daily_ret indexed by Date)

Outputs: prints hit rate, naive 95% CI, cluster-aware 95% CI, and the
revised-data hit rate for sanity-check parity with the published 76.3%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def to_monthly(daily: pd.Series) -> pd.Series:
    """Compound daily returns to month-end."""
    return (1 + daily).resample("ME").prod() - 1


def block_bootstrap_ci(
    beats: np.ndarray,
    down: np.ndarray,
    block: int = 6,
    n_boot: int = 10_000,
    seed: int = 20260618,
) -> tuple[float, float, float]:
    """Cluster-aware circular block bootstrap on the hit rate."""
    rng = np.random.default_rng(seed)
    n = len(beats)
    hits = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        nd = down[idx].sum()
        if nd > 0:
            hits[i] = (beats[idx] * down[idx]).sum() / nd
    hits = hits[~np.isnan(hits)]
    lo, hi = np.percentile(hits, [2.5, 97.5])
    return float(lo), float(np.median(hits)), float(hi)


def compute_hit_rate(strat_daily: pd.Series, spy: pd.Series, ief: pd.Series) -> dict:
    """Return hit rate + naive & cluster-aware CIs for a strategy vs 60/40."""
    strat_m = to_monthly(strat_daily)
    sixty40_m = 0.6 * to_monthly(spy) + 0.4 * to_monthly(ief)
    down = sixty40_m < 0
    beats = (strat_m > sixty40_m) & down
    n_down = int(down.sum())
    n_beats = int(beats.sum())
    rate = n_beats / n_down if n_down else float("nan")

    from scipy.stats import beta as _beta

    lo_naive = float(_beta.ppf(0.025, n_beats, n_down - n_beats + 1)) if n_beats > 0 else 0.0
    hi_naive = float(_beta.ppf(0.975, n_beats + 1, n_down - n_beats)) if n_beats < n_down else 1.0
    cb_lo, cb_med, cb_hi = block_bootstrap_ci(
        beats.astype(int).values,
        down.astype(int).values,
    )
    return {
        "n_down": n_down,
        "n_beats": n_beats,
        "hit_rate": rate,
        "naive_ci": (lo_naive, hi_naive),
        "cluster_ci": (cb_lo, cb_hi),
        "cluster_median": cb_med,
        "n_months": len(strat_m),
        "first_month": strat_m.index.min(),
        "last_month": strat_m.index.max(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vintage-csv",
        default="outputs/vintage_wf_daily_returns.csv",
        help="Daily vintage WF returns CSV (cols: Date, strategy_vintage_wf)",
    )
    p.add_argument(
        "--bench-csv",
        default="outputs/revised_wf_daily_series.csv",
        help="Daily benchmark CSV with spy_daily_ret + ief_daily_ret columns",
    )
    p.add_argument(
        "--revised-col",
        default="strategy_b_wf",
        help="Column name for revised-data strategy returns in --bench-csv (for parity check)",
    )
    args = p.parse_args()

    v = pd.read_csv(args.vintage_csv, parse_dates=["Date"]).set_index("Date")
    b = pd.read_csv(args.bench_csv, parse_dates=["Date"]).set_index("Date")
    df = v.join(b[["spy_daily_ret", "ief_daily_ret"]], how="inner").dropna()

    print(f"Aligned daily rows: {len(df)} ({df.index.min().date()} → {df.index.max().date()})")

    vintage = compute_hit_rate(df["strategy_vintage_wf"], df["spy_daily_ret"], df["ief_daily_ret"])
    print()
    print("=== VINTAGE HIT RATE (point-in-time labels) ===")
    print(
        f"60/40 down months: {vintage['n_down']}; strategy beats: {vintage['n_beats']}; "
        f"hit rate: {vintage['hit_rate'] * 100:.1f}%"
    )
    print(
        f"Naive 95% CI: [{vintage['naive_ci'][0] * 100:.1f}%, {vintage['naive_ci'][1] * 100:.1f}%]"
    )
    print(
        f"Cluster-aware (6mo block) 95% CI: "
        f"[{vintage['cluster_ci'][0] * 100:.1f}%, {vintage['cluster_ci'][1] * 100:.1f}%] "
        f"(median {vintage['cluster_median'] * 100:.1f}%)"
    )

    if args.revised_col in b.columns:
        rb = b.dropna()
        revised = compute_hit_rate(rb[args.revised_col], rb["spy_daily_ret"], rb["ief_daily_ret"])
        print()
        print("=== REVISED HIT RATE (parity check vs published 76.3%) ===")
        print(
            f"60/40 down months: {revised['n_down']}; strategy beats: {revised['n_beats']}; "
            f"hit rate: {revised['hit_rate'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
