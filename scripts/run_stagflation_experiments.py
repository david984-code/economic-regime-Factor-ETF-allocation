"""Run baseline, override, and risk_on_cap experiments; compare in SQLite."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.walk_forward import run_walk_forward_evaluation
from src.evaluation.model_results_db import list_runs, get_run_segments
from src.config import OUTPUTS_DIR
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _run_experiment(name: str, use_override: bool, use_cap: bool, cap_value: float = 0.2) -> str | None:
    """Run one experiment, return run_id."""
    from src.config import START_DATE, get_end_date

    print(f"\n--- Running {name} ---")
    df = run_walk_forward_evaluation(
        start=START_DATE,
        end=get_end_date(),
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=use_override,
        use_stagflation_risk_on_cap=use_cap,
        stagflation_risk_on_cap=cap_value,
    )
    if df.empty:
        return None
    run_id = df["run_id"].iloc[0]
    print(f"  run_id: {run_id}")
    return run_id


def _stagflation_segment_metrics(run_id: str, regime_df: pd.DataFrame) -> dict:
    """Compute Stagflation-only segment metrics for a run."""
    seg_df = get_run_segments(run_id)
    if seg_df.empty:
        return {}
    regimes = regime_df.copy()
    regimes["month"] = regimes.index.to_period("M")
    stag_cagrs = []
    stag_sharpes = []
    for _, row in seg_df.iterrows():
        ts = pd.Period(row["test_start"], freq="M")
        te = pd.Period(row["test_end"], freq="M")
        sub = regimes[(regimes["month"] >= ts) & (regimes["month"] <= te)]
        if len(sub) > 0:
            dom = sub["regime"].mode().iloc[0] if len(sub["regime"].mode()) > 0 else ""
            if dom == "Stagflation":
                c = row.get("strategy_cagr")
                s = row.get("strategy_sharpe")
                if pd.notna(c):
                    stag_cagrs.append(c)
                if pd.notna(s):
                    stag_sharpes.append(s)
    return {
        "stagflation_cagr": np.mean(stag_cagrs) if stag_cagrs else np.nan,
        "stagflation_sharpe": np.mean(stag_sharpes) if stag_sharpes else np.nan,
        "stagflation_n": len(stag_cagrs),
    }


def _beat_rates(run_id: str) -> dict:
    """Compute beat rates vs benchmarks for a run."""
    seg_df = get_run_segments(run_id)
    if seg_df.empty:
        return {}
    beat = {"spy": 0, "b60_40": 0, "equal_weight": 0, "risk_on_off": 0}
    total = 0
    for _, row in seg_df.iterrows():
        sc = row.get("strategy_cagr")
        if pd.isna(sc):
            continue
        total += 1
        if pd.notna(row.get("spy_cagr")) and sc > row["spy_cagr"]:
            beat["spy"] += 1
        if pd.notna(row.get("b60_40_cagr")) and sc > row["b60_40_cagr"]:
            beat["b60_40"] += 1
        if pd.notna(row.get("equal_weight_cagr")) and sc > row["equal_weight_cagr"]:
            beat["equal_weight"] += 1
        if pd.notna(row.get("risk_on_off_cagr")) and sc > row["risk_on_off_cagr"]:
            beat["risk_on_off"] += 1
    return {k: v / total if total > 0 else 0 for k, v in beat.items()}


def _run_2021_2022_metrics(run_id: str) -> dict:
    """Compute metrics for 2021-2022 style segments (test_start in 2021 or 2022)."""
    seg_df = get_run_segments(run_id)
    if seg_df.empty:
        return {}
    sub = seg_df[
        seg_df["test_start"].str.startswith("2021") | seg_df["test_start"].str.startswith("2022")
    ]
    if sub.empty:
        return {"cagr_2021_2022": np.nan, "sharpe_2021_2022": np.nan, "n_2021_2022": 0}
    return {
        "cagr_2021_2022": sub["strategy_cagr"].mean(),
        "sharpe_2021_2022": sub["strategy_sharpe"].mean(),
        "n_2021_2022": len(sub),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline")
    parser.add_argument("--override-only", action="store_true", help="Run only override")
    parser.add_argument("--cap-only", action="store_true", help="Run only risk_on_cap")
    parser.add_argument("--compare-only", action="store_true",
                       help="Skip runs, compare last 3 by experiment_type")
    args = parser.parse_args()

    run_ids = {}

    if not args.compare_only:
        if args.baseline_only:
            rid = _run_experiment("baseline", use_override=False, use_cap=False)
            if rid:
                run_ids["baseline"] = rid
        elif args.override_only:
            rid = _run_experiment("override", use_override=True, use_cap=False)
            if rid:
                run_ids["override"] = rid
        elif args.cap_only:
            rid = _run_experiment("risk_on_cap", use_override=False, use_cap=True)
            if rid:
                run_ids["risk_on_cap"] = rid
        else:
            rid = _run_experiment("baseline", use_override=False, use_cap=False)
            if rid:
                run_ids["baseline"] = rid
            rid = _run_experiment("override", use_override=True, use_cap=False)
            if rid:
                run_ids["override"] = rid
            rid = _run_experiment("risk_on_cap", use_override=False, use_cap=True)
            if rid:
                run_ids["risk_on_cap"] = rid

    if args.compare_only or run_ids:
        runs = list_runs()
        by_type = {}
        for r in runs:
            et = r.get("experiment_type") or "unknown"
            if et not in by_type:
                by_type[et] = r
        if run_ids:
            for k, vid in run_ids.items():
                match = next((r for r in runs if r["run_id"] == vid), None)
                if match:
                    by_type[k] = match

        regime_df = pd.read_csv(OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"])
        regime_df = regime_df.dropna(subset=["regime"]).set_index("date")
        regime_df.index = pd.to_datetime(regime_df.index)

        exp_order = ["baseline", "stagflation_override", "stagflation_risk_on_cap"]
        report_lines = [
            "# Stagflation Experiment Comparison",
            "",
            "## 1. Overall Metrics",
            "",
            "| Experiment | run_id | CAGR | Sharpe | MaxDD | Vol | Turnover |",
            "|------------|--------|------|--------|-------|-----|----------|",
        ]
        stag_metrics = {}
        for label in exp_order:
            run = by_type.get(label)
            if run is None:
                report_lines.append(f"| {label} | - | - | - | - | - | - |")
                continue
            stag_metrics[label] = _stagflation_segment_metrics(run["run_id"], regime_df)
            report_lines.append(
                f"| {label} | {run['run_id'][:8]}... | "
                f"{run.get('strategy_cagr', 0):.2%} | {run.get('strategy_sharpe', 0):.3f} | "
                f"{run.get('strategy_maxdd', 0):.2%} | {run.get('strategy_vol', 0):.2%} | "
                f"{run.get('strategy_turnover', 0):.2f} |"
            )

        report_lines.extend([
            "",
            "## 2. Stagflation-Only Segment Performance",
            "",
            "| Experiment | Stagflation CAGR | Stagflation Sharpe | N segments |",
            "|------------|------------------|--------------------|-----------|",
        ])
        for label in exp_order:
            run = by_type.get(label)
            if run is None:
                report_lines.append(f"| {label} | - | - | - |")
                continue
            sm = stag_metrics.get(label, {})
            report_lines.append(
                f"| {label} | {sm.get('stagflation_cagr', np.nan):.2%} | "
                f"{sm.get('stagflation_sharpe', np.nan):.3f} | {sm.get('stagflation_n', 0)} |"
            )

        report_lines.extend([
            "",
            "## 3. Beat Rates vs Benchmarks",
            "",
            "| Experiment | vs SPY | vs 60/40 | vs Equal_Weight | vs Risk_On_Off |",
            "|------------|--------|----------|----------------|----------------|",
        ])
        for label in exp_order:
            run = by_type.get(label)
            if run is None:
                report_lines.append(f"| {label} | - | - | - | - |")
                continue
            br = _beat_rates(run["run_id"])
            report_lines.append(
                f"| {label} | {br.get('spy', 0):.1%} | {br.get('b60_40', 0):.1%} | "
                f"{br.get('equal_weight', 0):.1%} | {br.get('risk_on_off', 0):.1%} |"
            )

        report_lines.extend([
            "",
            "## 4. 2021-2022 Style Periods",
            "",
            "| Experiment | CAGR | Sharpe | N segments |",
            "|------------|------|--------|-----------|",
        ])
        for label in exp_order:
            run = by_type.get(label)
            if run is None:
                report_lines.append(f"| {label} | - | - | - |")
                continue
            m = _run_2021_2022_metrics(run["run_id"])
            report_lines.append(
                f"| {label} | {m.get('cagr_2021_2022', np.nan):.2%} | "
                f"{m.get('sharpe_2021_2022', np.nan):.3f} | {m.get('n_2021_2022', 0)} |"
            )

        cap_run = by_type.get("stagflation_risk_on_cap")
        baseline_run = by_type.get("baseline")
        if cap_run and baseline_run:
            cap_cagr = cap_run.get("strategy_cagr") or 0
            base_cagr = baseline_run.get("strategy_cagr") or 0
            cap_stag = stag_metrics.get("stagflation_risk_on_cap", {}).get("stagflation_cagr") or 0
            base_stag = stag_metrics.get("baseline", {}).get("stagflation_cagr") or 0
            if cap_cagr > base_cagr and cap_stag > base_stag:
                rec = "**KEEP** - Cap improves overall and Stagflation performance vs baseline."
            elif cap_cagr < base_cagr and cap_stag < base_stag:
                rec = "**REJECT** - Cap worsens overall and Stagflation performance vs baseline."
            else:
                rec = "**FURTHER TUNE** - Mixed results; consider testing different cap values (e.g. 0.15, 0.25)."
        else:
            rec = "Insufficient runs for recommendation."
        report_lines.extend([
            "",
            "## 5. Recommendation",
            "",
            rec,
            "",
            "---",
            "",
            "*Run `python scripts/run_stagflation_experiments.py` to refresh.*",
        ])

        out_path = OUTPUTS_DIR / "STAGFLATION_EXPERIMENT_RISK_ON_CAP.md"
        out_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\nReport written to {out_path}")

        for line in report_lines[:60]:
            print(line)


if __name__ == "__main__":
    main()
