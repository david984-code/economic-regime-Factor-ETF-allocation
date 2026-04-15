"""Automated parameter sensitivity scanner.

Sweeps parameter combinations through walk-forward validation and
persists structured results to SQLite + heatmap charts.

Usage:
    python -m src.research.sensitivity                    # full sweep
    python -m src.research.sensitivity --fast             # fast mode (8yr, fewer segments)
    python -m src.research.sensitivity --params sigmoid_scale,trend_filter_type
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

SENSITIVITY_DB = OUTPUTS_DIR / "sensitivity_results.db"

# --- Default parameter grid ---
DEFAULT_GRID: dict[str, list[Any]] = {
    "market_lookback_months": [12, 18, 24, 36],
    "sigmoid_scale": [0.15, 0.25, 0.35, 0.50],
    "trend_filter_type": ["none", "200dma", "10mma"],
    "tolerance": [0.0, 0.01, 0.015, 0.02],
}

# Baseline values (from PROJECT_CONTEXT.md accepted model)
BASELINE: dict[str, Any] = {
    "market_lookback_months": 24,
    "sigmoid_scale": 0.25,
    "trend_filter_type": "none",
    "tolerance": 0.015,
}

# Metrics to extract from walk-forward OVERALL row
METRICS_COLS = [
    "Strategy_CAGR",
    "Strategy_Sharpe",
    "Strategy_MaxDD",
    "Strategy_Vol",
    "Strategy_Turnover",
    "Strategy_HitRate",
]


@dataclass
class SweepResult:
    """Result of a single parameter combination."""

    params: dict[str, Any]
    metrics: dict[str, float]
    run_id: str
    elapsed_sec: float
    n_segments: int


@dataclass
class SensitivityReport:
    """Full sweep report."""

    sweep_id: str
    created_at: str
    grid: dict[str, list[Any]]
    baseline: dict[str, Any]
    results: list[SweepResult] = field(default_factory=list)
    total_elapsed_sec: float = 0.0


def _init_db() -> sqlite3.Connection:
    """Initialize sensitivity results database."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(SENSITIVITY_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensitivity_sweeps (
            sweep_id TEXT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            grid_json TEXT NOT NULL,
            baseline_json TEXT NOT NULL,
            n_combinations INTEGER NOT NULL,
            total_elapsed_sec REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensitivity_results (
            sweep_id TEXT NOT NULL,
            combo_idx INTEGER NOT NULL,
            params_json TEXT NOT NULL,
            run_id TEXT,
            elapsed_sec REAL,
            n_segments INTEGER,
            strategy_cagr REAL,
            strategy_sharpe REAL,
            strategy_maxdd REAL,
            strategy_vol REAL,
            strategy_turnover REAL,
            strategy_hit_rate REAL,
            sharpe_delta REAL,
            cagr_delta REAL,
            maxdd_delta REAL,
            PRIMARY KEY (sweep_id, combo_idx),
            FOREIGN KEY (sweep_id) REFERENCES sensitivity_sweeps(sweep_id)
        )
    """)
    conn.commit()
    return conn


def build_param_combos(
    grid: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate all parameter combinations from grid.

    Non-swept parameters are held at baseline values.
    """
    grid = grid or DEFAULT_GRID
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for vals in itertools.product(*values):
        combo = dict(BASELINE)
        combo.update(dict(zip(keys, vals, strict=True)))
        combos.append(combo)
    return combos


def _extract_overall_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Extract OVERALL row metrics from walk-forward results."""
    overall = df[df["segment"] == "OVERALL"]
    if overall.empty:
        return {col: float("nan") for col in METRICS_COLS}
    row = overall.iloc[0]
    return {col: float(row.get(col, float("nan"))) for col in METRICS_COLS}


def run_single_combo(
    params: dict[str, Any],
    fast_mode: bool = False,
) -> tuple[dict[str, float], str, int]:
    """Run walk-forward for a single parameter combination.

    Returns:
        (metrics_dict, run_id, n_segments)
    """
    from src.evaluation.walk_forward import run_walk_forward_evaluation

    wf_kwargs: dict[str, Any] = {
        "market_lookback_months": params.get(
            "market_lookback_months",
            BASELINE["market_lookback_months"],
        ),
        "sigmoid_scale": params.get("sigmoid_scale", BASELINE["sigmoid_scale"]),
        "trend_filter_type": params.get(
            "trend_filter_type",
            BASELINE["trend_filter_type"],
        ),
        "tolerance": params.get("tolerance", BASELINE["tolerance"]),
        "trend_filter_risk_on_cap": params.get("trend_filter_risk_on_cap", 0.3),
        "fast_mode": fast_mode,
        "skip_persist": True,
        "use_cache": True,
    }

    df = run_walk_forward_evaluation(**wf_kwargs)
    metrics = _extract_overall_metrics(df)
    run_id = str(df["run_id"].iloc[0]) if not df.empty else ""
    n_segments = len(df[df["segment"] != "OVERALL"]) if not df.empty else 0
    return metrics, run_id, n_segments


def run_sweep(
    grid: dict[str, list[Any]] | None = None,
    fast_mode: bool = False,
    max_combos: int | None = None,
) -> SensitivityReport:
    """Run full parameter sensitivity sweep.

    Args:
        grid: Parameter grid. Defaults to DEFAULT_GRID.
        fast_mode: Use fast mode for walk-forward (8yr, fewer segments).
        max_combos: Cap number of combinations (for testing).

    Returns:
        SensitivityReport with all results.
    """
    grid = grid or DEFAULT_GRID
    combos = build_param_combos(grid)
    if max_combos:
        combos = combos[:max_combos]

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = SensitivityReport(
        sweep_id=sweep_id,
        created_at=datetime.now().isoformat(),
        grid=grid,
        baseline=dict(BASELINE),
    )

    logger.info(
        "Starting sensitivity sweep: %d combinations, fast_mode=%s",
        len(combos),
        fast_mode,
    )

    # Run baseline first for delta computation
    logger.info("Running baseline: %s", BASELINE)
    t0 = time.perf_counter()
    baseline_metrics, _, _ = run_single_combo(BASELINE, fast_mode=fast_mode)
    baseline_time = time.perf_counter() - t0
    logger.info(
        "Baseline: Sharpe=%.3f CAGR=%.3f MaxDD=%.3f (%.1fs)",
        baseline_metrics.get("Strategy_Sharpe", float("nan")),
        baseline_metrics.get("Strategy_CAGR", float("nan")),
        baseline_metrics.get("Strategy_MaxDD", float("nan")),
        baseline_time,
    )

    t_total = time.perf_counter()
    for i, combo in enumerate(combos):
        logger.info("[%d/%d] Running: %s", i + 1, len(combos), combo)
        t0 = time.perf_counter()
        try:
            metrics, run_id, n_segments = run_single_combo(combo, fast_mode=fast_mode)
        except Exception as e:
            logger.error("[%d/%d] Failed: %s", i + 1, len(combos), e)
            metrics = {col: float("nan") for col in METRICS_COLS}
            run_id = ""
            n_segments = 0
        elapsed = time.perf_counter() - t0

        result = SweepResult(
            params=combo,
            metrics=metrics,
            run_id=run_id,
            elapsed_sec=elapsed,
            n_segments=n_segments,
        )
        report.results.append(result)

        sharpe = metrics.get("Strategy_Sharpe", float("nan"))
        cagr = metrics.get("Strategy_CAGR", float("nan"))
        logger.info(
            "[%d/%d] Sharpe=%.3f CAGR=%.3f (%.1fs)",
            i + 1,
            len(combos),
            sharpe,
            cagr,
            elapsed,
        )

    report.total_elapsed_sec = time.perf_counter() - t_total

    # Persist
    _persist_report(report, baseline_metrics)
    _save_summary_csv(report, baseline_metrics)
    _generate_heatmaps(report, baseline_metrics)

    logger.info(
        "Sweep complete: %d combos in %.1fs",
        len(combos),
        report.total_elapsed_sec,
    )
    return report


def _persist_report(
    report: SensitivityReport,
    baseline_metrics: dict[str, float],
) -> None:
    """Save sweep results to SQLite."""
    conn = _init_db()
    try:
        conn.execute(
            """
            INSERT INTO sensitivity_sweeps
            (sweep_id, created_at, grid_json, baseline_json, n_combinations, total_elapsed_sec)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report.sweep_id,
                report.created_at,
                json.dumps(report.grid),
                json.dumps(report.baseline),
                len(report.results),
                report.total_elapsed_sec,
            ),
        )
        bl_sharpe = baseline_metrics.get("Strategy_Sharpe", 0.0)
        bl_cagr = baseline_metrics.get("Strategy_CAGR", 0.0)
        bl_maxdd = baseline_metrics.get("Strategy_MaxDD", 0.0)

        for i, r in enumerate(report.results):
            s = r.metrics.get("Strategy_Sharpe", float("nan"))
            c = r.metrics.get("Strategy_CAGR", float("nan"))
            d = r.metrics.get("Strategy_MaxDD", float("nan"))
            conn.execute(
                """
                INSERT INTO sensitivity_results
                (sweep_id, combo_idx, params_json, run_id, elapsed_sec, n_segments,
                 strategy_cagr, strategy_sharpe, strategy_maxdd, strategy_vol,
                 strategy_turnover, strategy_hit_rate,
                 sharpe_delta, cagr_delta, maxdd_delta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.sweep_id,
                    i,
                    json.dumps(r.params),
                    r.run_id,
                    r.elapsed_sec,
                    r.n_segments,
                    r.metrics.get("Strategy_CAGR"),
                    r.metrics.get("Strategy_Sharpe"),
                    r.metrics.get("Strategy_MaxDD"),
                    r.metrics.get("Strategy_Vol"),
                    r.metrics.get("Strategy_Turnover"),
                    r.metrics.get("Strategy_HitRate"),
                    s - bl_sharpe if np.isfinite(s) else None,
                    c - bl_cagr if np.isfinite(c) else None,
                    d - bl_maxdd if np.isfinite(d) else None,
                ),
            )
        conn.commit()
        logger.info("Persisted sweep %s to %s", report.sweep_id, SENSITIVITY_DB)
    finally:
        conn.close()


def _save_summary_csv(
    report: SensitivityReport,
    baseline_metrics: dict[str, float],
) -> None:
    """Save summary CSV with all combos ranked by Sharpe."""
    rows: list[dict[str, Any]] = []
    bl_sharpe = baseline_metrics.get("Strategy_Sharpe", 0.0)
    bl_cagr = baseline_metrics.get("Strategy_CAGR", 0.0)

    for r in report.results:
        row: dict[str, Any] = dict(r.params)
        row.update(r.metrics)
        s = r.metrics.get("Strategy_Sharpe", float("nan"))
        c = r.metrics.get("Strategy_CAGR", float("nan"))
        row["Sharpe_Delta"] = s - bl_sharpe if np.isfinite(s) else float("nan")
        row["CAGR_Delta"] = c - bl_cagr if np.isfinite(c) else float("nan")
        row["n_segments"] = r.n_segments
        row["elapsed_sec"] = r.elapsed_sec
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Strategy_Sharpe", ascending=False)

    out_path = OUTPUTS_DIR / f"sensitivity_{report.sweep_id}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved sensitivity summary: %s", out_path)

    # Also save as latest
    latest = OUTPUTS_DIR / "sensitivity_latest.csv"
    df.to_csv(latest, index=False)


def _generate_heatmaps(
    report: SensitivityReport,
    baseline_metrics: dict[str, float],
) -> None:
    """Generate heatmap charts for 2D parameter interactions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_dir = OUTPUTS_DIR / "sensitivity_charts"
    chart_dir.mkdir(exist_ok=True)

    # Build results DataFrame
    rows: list[dict[str, Any]] = []
    bl_sharpe = baseline_metrics.get("Strategy_Sharpe", 0.0)
    for r in report.results:
        row = dict(r.params)
        s = r.metrics.get("Strategy_Sharpe", float("nan"))
        row["Sharpe"] = s
        row["Sharpe_Delta"] = s - bl_sharpe if np.isfinite(s) else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)

    # Generate heatmap for each pair of swept parameters
    swept_params = [k for k in report.grid if len(report.grid[k]) > 1]
    pairs = list(itertools.combinations(swept_params, 2))

    for p1, p2 in pairs:
        # Pivot: average Sharpe across other parameters
        pivot = df.pivot_table(
            values="Sharpe_Delta",
            index=p1,
            columns=p2,
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            origin="lower",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"Sharpe Delta: {p1} vs {p2}")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    color = "white" if abs(val) > 0.05 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:+.3f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

        fig.colorbar(im, label="Sharpe Delta vs Baseline")
        fig.tight_layout()
        fname = f"heatmap_{p1}_vs_{p2}.png"
        fig.savefig(chart_dir / fname, dpi=150)
        plt.close(fig)
        logger.info("Saved heatmap: %s", fname)

    # Single-parameter sensitivity plots
    for param in swept_params:
        param_df = (
            df.groupby(param)
            .agg(
                Sharpe_Mean=("Sharpe", "mean"),
                Sharpe_Std=("Sharpe", "std"),
                Sharpe_Delta_Mean=("Sharpe_Delta", "mean"),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(param_df))
        ax.bar(
            x,
            param_df["Sharpe_Delta_Mean"],
            yerr=param_df["Sharpe_Std"],
            color=["green" if v > 0 else "red" for v in param_df["Sharpe_Delta_Mean"]],
            alpha=0.7,
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_df[param]], rotation=45)
        ax.set_xlabel(param)
        ax.set_ylabel("Sharpe Delta vs Baseline")
        ax.set_title(f"Sensitivity: {param}")
        ax.axhline(y=0, color="black", linewidth=0.5)
        fig.tight_layout()
        fname = f"sensitivity_{param}.png"
        fig.savefig(chart_dir / fname, dpi=150)
        plt.close(fig)
        logger.info("Saved sensitivity plot: %s", fname)


def load_latest_sweep() -> pd.DataFrame | None:
    """Load the most recent sensitivity sweep results."""
    latest = OUTPUTS_DIR / "sensitivity_latest.csv"
    if not latest.exists():
        return None
    return pd.read_csv(latest)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter sensitivity scanner.")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 8yr window, fewer segments.",
    )
    p.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated parameter names to sweep (default: all).",
    )
    p.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="Max combinations to run (0 = all).",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_cli()

    grid = dict(DEFAULT_GRID)
    if args.params:
        selected = [p.strip() for p in args.params.split(",")]
        grid = {k: v for k, v in DEFAULT_GRID.items() if k in selected}
        # Hold non-selected at baseline
        for k in DEFAULT_GRID:
            if k not in grid:
                grid[k] = [BASELINE[k]]

    max_c = args.max_combos if args.max_combos > 0 else None
    run_sweep(grid=grid, fast_mode=args.fast, max_combos=max_c)
