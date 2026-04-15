"""Compare Stagflation override experiment run vs baseline run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.model_results_db import (
    compare_runs,
    list_runs,
)


def main() -> None:
    runs = list_runs()
    if len(runs) < 2:
        print("Need at least 2 runs. Run walk-forward with override, then compare.")
        sys.exit(1)

    new_run_id = runs[0]["run_id"]
    baseline_run_id = runs[1]["run_id"]

    print("=" * 70)
    print("STAGFLATION OVERRIDE EXPERIMENT: COMPARISON")
    print("=" * 70)
    print(f"\nNew run (override=True):  {new_run_id}")
    print(f"Baseline run (override=False or pre-change): {baseline_run_id}")

    c = compare_runs(new_run_id, baseline_run_id)
    r_new = c["run_1"]
    r_base = c["run_2"]

    print("\n" + "-" * 70)
    print("OVERALL METRICS (model_runs)")
    print("-" * 70)
    metrics = [
        "strategy_cagr",
        "strategy_sharpe",
        "strategy_maxdd",
        "strategy_vol",
        "strategy_turnover",
    ]
    print(f"{'Metric':<25} {'New':>12} {'Baseline':>12} {'Change':>12}")
    for m in metrics:
        v_new = r_new.get(m)
        v_base = r_base.get(m)
        if v_new is not None and v_base is not None:
            chg = v_new - v_base
            f"{(chg / v_base) * 100:+.1f}%" if v_base != 0 else "N/A"
            print(f"{m:<25} {v_new:>12.4f} {v_base:>12.4f} {chg:>+12.4f}")
        else:
            print(f"{m:<25} {str(v_new):>12} {str(v_base):>12}")

    # Load segment-level data for Stagflation comparison
    print("\n" + "-" * 70)
    print("STAGFLATION SEGMENTS (from CSV)")
    print("-" * 70)

    csv_new = OUTPUTS_DIR / f"walk_forward_{new_run_id}.csv"
    csv_base = OUTPUTS_DIR / f"walk_forward_{baseline_run_id}.csv"

    if not csv_base.exists():
        print("Baseline CSV not found. Using model_runs only.")
        return

    df_new = pd.read_csv(csv_new)
    df_base = pd.read_csv(csv_base)
    df_new = df_new[df_new["segment"] != "OVERALL"]
    df_base = df_base[df_base["segment"] != "OVERALL"]

    # Tag Stagflation segments
    regimes = pd.read_csv(
        OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regimes["month"] = pd.to_datetime(regimes["date"]).dt.to_period("M")

    def tag_regime(df):
        seg_regimes = []
        for _, row in df.iterrows():
            ts = pd.Period(row["test_start"], freq="M")
            te = pd.Period(row["test_end"], freq="M")
            sub = regimes[(regimes["month"] >= ts) & (regimes["month"] <= te)]
            dom = (
                sub["regime"].mode().iloc[0]
                if len(sub) > 0 and len(sub["regime"].mode()) > 0
                else "Unknown"
            )
            seg_regimes.append(dom)
        return seg_regimes

    df_new["dominant_regime"] = tag_regime(df_new)
    df_base["dominant_regime"] = tag_regime(df_base)

    stag_new = df_new[df_new["dominant_regime"] == "Stagflation"]
    stag_base = df_base[df_base["dominant_regime"] == "Stagflation"]

    if len(stag_new) > 0 and len(stag_base) > 0:
        print(f"Stagflation segments: New={len(stag_new)}, Baseline={len(stag_base)}")
        print(
            f"\nNew (override):     Strategy CAGR={stag_new['Strategy_CAGR'].mean():.2%}, Sharpe={stag_new['Strategy_Sharpe'].mean():.3f}"
        )
        print(
            f"Baseline:           Strategy CAGR={stag_base['Strategy_CAGR'].mean():.2%}, Sharpe={stag_base['Strategy_Sharpe'].mean():.3f}"
        )
        cagr_chg = stag_new["Strategy_CAGR"].mean() - stag_base["Strategy_CAGR"].mean()
        sharpe_chg = (
            stag_new["Strategy_Sharpe"].mean() - stag_base["Strategy_Sharpe"].mean()
        )
        print(f"Change in Stagflation: CAGR {cagr_chg:+.2%}, Sharpe {sharpe_chg:+.3f}")
    else:
        print("No Stagflation segments in one or both runs.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
