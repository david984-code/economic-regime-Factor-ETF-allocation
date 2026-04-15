"""Analyze walk-forward results for robustness assessment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.model_results_db import get_latest_run

BENCHMARKS = ["SPY", "60/40", "Equal_Weight", "Risk_On_Off"]
METRICS = ["CAGR", "Sharpe", "MaxDD", "Vol"]
STRATEGY_PREFIX = "Strategy_"
BENCH_PREFIXES = {b: f"{b}_" for b in BENCHMARKS}


def load_latest() -> pd.DataFrame:
    """Load latest walk-forward results (segments only, exclude OVERALL)."""
    r = get_latest_run()
    if not r:
        csv_path = OUTPUTS_DIR / "walk_forward_results.csv"
    else:
        csv_path = OUTPUTS_DIR / f"walk_forward_{r['run_id']}.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "walk_forward_results.csv"
    df = pd.read_csv(csv_path)
    df = df[df["segment"] != "OVERALL"].copy()
    return df


def main() -> None:
    df = load_latest()
    if df.empty:
        print("No segment data found.")
        return

    n = len(df)
    print("=" * 70)
    print("WALK-FORWARD ROBUSTNESS ASSESSMENT")
    print("=" * 70)
    print(f"\nSegments analyzed: {n} (excluding OVERALL row)")
    print(f"Date range: {df['test_start'].min()} to {df['test_end'].max()}")

    # Overall averages (segment-level)
    print("\n" + "-" * 70)
    print("1. OVERALL METRICS (averaged across segments)")
    print("-" * 70)
    rows = []
    for name, prefix in [("Strategy", STRATEGY_PREFIX)] + [
        (b, BENCH_PREFIXES[b]) for b in BENCHMARKS
    ]:
        row = {"": name}
        for m in METRICS:
            col = f"{prefix}{m}"
            if col in df.columns:
                row[m] = df[col].mean()
        if name == "Strategy":
            if "Strategy_HitRate" in df.columns:
                row["HitRate"] = df["Strategy_HitRate"].mean()
            if "Strategy_Turnover" in df.columns:
                row["Turnover"] = df["Strategy_Turnover"].mean()
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))

    # Beat rates
    print("\n" + "-" * 70)
    print("2. SEGMENT BEAT RATES (Strategy vs benchmark)")
    print("-" * 70)
    beat_rates = {}
    for b in BENCHMARKS:
        prefix = BENCH_PREFIXES[b]
        wins = 0
        for m in ["CAGR", "Sharpe"]:
            sc = f"{STRATEGY_PREFIX}{m}"
            bc = f"{prefix}{m}"
            if sc in df.columns and bc in df.columns:
                wins += (df[sc] > df[bc]).sum()
        beat_rates[b] = wins / (n * 2) if n > 0 else 0  # avg across CAGR and Sharpe
    for b in BENCHMARKS:
        cagr_wins = (
            (df[f"{STRATEGY_PREFIX}CAGR"] > df[f"{b}_CAGR"]).sum()
            if f"{b}_CAGR" in df.columns
            else 0
        )
        sharpe_wins = (
            (df[f"{STRATEGY_PREFIX}Sharpe"] > df[f"{b}_Sharpe"]).sum()
            if f"{b}_Sharpe" in df.columns
            else 0
        )
        print(
            f"  vs {b}: CAGR wins {cagr_wins}/{n} ({100 * cagr_wins / n:.1f}%), Sharpe wins {sharpe_wins}/{n} ({100 * sharpe_wins / n:.1f}%)"
        )

    # Best / worst segments
    print("\n" + "-" * 70)
    print("3. BEST SEGMENTS (by Strategy Sharpe)")
    print("-" * 70)
    df_s = df.sort_values(f"{STRATEGY_PREFIX}Sharpe", ascending=False)
    best = df_s.head(5)[
        [
            "test_start",
            "test_end",
            f"{STRATEGY_PREFIX}Sharpe",
            f"{STRATEGY_PREFIX}CAGR",
            "SPY_Sharpe",
            "SPY_CAGR",
        ]
    ]
    best.columns = [
        "test_start",
        "test_end",
        "Strategy_Sharpe",
        "Strategy_CAGR",
        "SPY_Sharpe",
        "SPY_CAGR",
    ]
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 70)
    print("4. WORST SEGMENTS (by Strategy Sharpe)")
    print("-" * 70)
    worst = df_s.tail(5)[
        [
            "test_start",
            "test_end",
            f"{STRATEGY_PREFIX}Sharpe",
            f"{STRATEGY_PREFIX}CAGR",
            "SPY_Sharpe",
            "SPY_CAGR",
        ]
    ]
    worst.columns = [
        "test_start",
        "test_end",
        "Strategy_Sharpe",
        "Strategy_CAGR",
        "SPY_Sharpe",
        "SPY_CAGR",
    ]
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Performance concentration
    print("\n" + "-" * 70)
    print("5. PERFORMANCE CONCENTRATION")
    print("-" * 70)
    top5_sharpe = df_s.head(int(n * 0.2))
    bot5_sharpe = df_s.tail(int(n * 0.2))
    print(
        f"  Top 20% of segments (by Sharpe): avg Strategy Sharpe {top5_sharpe[f'{STRATEGY_PREFIX}Sharpe'].mean():.3f}"
    )
    print(
        f"  Bottom 20% of segments: avg Strategy Sharpe {bot5_sharpe[f'{STRATEGY_PREFIX}Sharpe'].mean():.3f}"
    )
    sharpe_std = df[f"{STRATEGY_PREFIX}Sharpe"].std()
    print(f"  Std of segment Sharpes: {sharpe_std:.3f} (high = unstable)")

    # Regime breakdown (if regime labels available)
    print("\n" + "-" * 70)
    print("6. REGIME BREAKDOWN (test period dominant regime)")
    print("-" * 70)
    regime_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if regime_path.exists():
        regimes = pd.read_csv(regime_path, parse_dates=["date"])
        regimes["month"] = pd.to_datetime(regimes["date"]).dt.to_period("M")
        regimes["regime"].value_counts()
        # Map each segment's test period to regime
        seg_regimes = []
        for _, row in df.iterrows():
            ts = pd.Period(row["test_start"], freq="M")
            te = pd.Period(row["test_end"], freq="M")
            sub = regimes[(regimes["month"] >= ts) & (regimes["month"] <= te)]
            if len(sub) > 0:
                dom = (
                    sub["regime"].mode().iloc[0]
                    if len(sub["regime"].mode()) > 0
                    else "Unknown"
                )
            else:
                dom = "Unknown"
            seg_regimes.append(dom)
        df["dominant_regime"] = seg_regimes
        by_regime = (
            df.groupby("dominant_regime")
            .agg(
                {
                    f"{STRATEGY_PREFIX}Sharpe": "mean",
                    f"{STRATEGY_PREFIX}CAGR": "mean",
                    "SPY_Sharpe": "mean",
                    "SPY_CAGR": "mean",
                }
            )
            .round(4)
        )
        by_regime["n_segments"] = df.groupby("dominant_regime").size()
        print(by_regime.to_string())
    else:
        print("  (regime_labels_expanded.csv not found; skipping regime breakdown)")

    # Warning signs
    print("\n" + "-" * 70)
    print("7. WARNING SIGNS")
    print("-" * 70)
    avg_sharpe = df[f"{STRATEGY_PREFIX}Sharpe"].mean()
    avg_turnover = df[f"{STRATEGY_PREFIX}Turnover"].mean()
    spy_beat = (df[f"{STRATEGY_PREFIX}CAGR"] > df["SPY_CAGR"]).sum() / n * 100
    neg_sharpe_segs = (df[f"{STRATEGY_PREFIX}Sharpe"] < 0).sum()
    print(f"  Strategy avg Sharpe: {avg_sharpe:.3f}")
    print(f"  Strategy avg Turnover: {avg_turnover:.2f}")
    print(f"  Segments with negative Sharpe: {neg_sharpe_segs}/{n}")
    print(f"  CAGR beat rate vs SPY: {spy_beat:.1f}%")
    if spy_beat < 50:
        print("  [!] Strategy beats SPY in fewer than half of segments")
    if neg_sharpe_segs > n * 0.2:
        print("  [!] >20% of segments have negative Sharpe")
    if avg_turnover > 3 and avg_sharpe < 0.5:
        print("  [!] High turnover relative to modest Sharpe")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
