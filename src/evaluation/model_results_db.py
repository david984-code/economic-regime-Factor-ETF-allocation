"""SQLite persistence for walk-forward evaluation results."""

import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import OUTPUTS_DIR

MODEL_RESULTS_DB = OUTPUTS_DIR / "model_results.db"

# CSV column name -> SQLite column name (for metrics)
_CSV_TO_DB: dict[str, str] = {
    "Strategy_CAGR": "strategy_cagr",
    "Strategy_Sharpe": "strategy_sharpe",
    "Strategy_MaxDD": "strategy_maxdd",
    "Strategy_Vol": "strategy_vol",
    "Strategy_HitRate": "strategy_hit_rate",
    "Strategy_Turnover": "strategy_turnover",
    "SPY_CAGR": "spy_cagr",
    "SPY_Sharpe": "spy_sharpe",
    "SPY_MaxDD": "spy_maxdd",
    "SPY_Vol": "spy_vol",
    "60/40_CAGR": "b60_40_cagr",
    "60/40_Sharpe": "b60_40_sharpe",
    "60/40_MaxDD": "b60_40_maxdd",
    "60/40_Vol": "b60_40_vol",
    "Equal_Weight_CAGR": "equal_weight_cagr",
    "Equal_Weight_Sharpe": "equal_weight_sharpe",
    "Equal_Weight_MaxDD": "equal_weight_maxdd",
    "Equal_Weight_Vol": "equal_weight_vol",
    "Risk_On_Off_CAGR": "risk_on_off_cagr",
    "Risk_On_Off_Sharpe": "risk_on_off_sharpe",
    "Risk_On_Off_MaxDD": "risk_on_off_maxdd",
    "Risk_On_Off_Vol": "risk_on_off_vol",
}


def _get_conn() -> sqlite3.Connection:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(str(MODEL_RESULTS_DB))


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            run_id TEXT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            min_train_months INTEGER NOT NULL,
            test_months INTEGER NOT NULL,
            expanding INTEGER NOT NULL,
            n_segments INTEGER NOT NULL,
            csv_path TEXT NOT NULL,
            strategy_cagr REAL,
            strategy_sharpe REAL,
            strategy_maxdd REAL,
            strategy_vol REAL,
            strategy_hit_rate REAL,
            strategy_turnover REAL,
            spy_cagr REAL,
            spy_sharpe REAL,
            spy_maxdd REAL,
            spy_vol REAL,
            b60_40_cagr REAL,
            b60_40_sharpe REAL,
            b60_40_maxdd REAL,
            b60_40_vol REAL,
            equal_weight_cagr REAL,
            equal_weight_sharpe REAL,
            equal_weight_maxdd REAL,
            equal_weight_vol REAL,
            risk_on_off_cagr REAL,
            risk_on_off_sharpe REAL,
            risk_on_off_maxdd REAL,
            risk_on_off_vol REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS walk_forward_segments (
            run_id TEXT NOT NULL,
            segment_idx INTEGER NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            test_start TEXT NOT NULL,
            test_end TEXT NOT NULL,
            is_overall INTEGER NOT NULL,
            strategy_cagr REAL,
            strategy_sharpe REAL,
            strategy_maxdd REAL,
            strategy_vol REAL,
            strategy_hit_rate REAL,
            strategy_turnover REAL,
            spy_cagr REAL,
            spy_sharpe REAL,
            spy_maxdd REAL,
            spy_vol REAL,
            b60_40_cagr REAL,
            b60_40_sharpe REAL,
            b60_40_maxdd REAL,
            b60_40_vol REAL,
            equal_weight_cagr REAL,
            equal_weight_sharpe REAL,
            equal_weight_maxdd REAL,
            equal_weight_vol REAL,
            risk_on_off_cagr REAL,
            risk_on_off_sharpe REAL,
            risk_on_off_maxdd REAL,
            risk_on_off_vol REAL,
            PRIMARY KEY (run_id, segment_idx),
            FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_walk_forward_segments_run_id "
        "ON walk_forward_segments(run_id)"
    )
    try:
        conn.execute("ALTER TABLE model_runs ADD COLUMN experiment_type TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()


def _get_experiment_type(params: dict[str, Any]) -> str:
    """Return experiment type label from params."""
    override = params.get("use_stagflation_override", True)
    cap = params.get("use_stagflation_risk_on_cap", False)
    if override:
        return "stagflation_override"
    if cap:
        return "stagflation_risk_on_cap"
    return "baseline"


def _row_to_db_values(row: pd.Series) -> dict[str, Any]:
    """Map CSV row to DB column values."""
    out: dict[str, Any] = {}
    for csv_col, db_col in _CSV_TO_DB.items():
        if csv_col in row.index:
            val = row[csv_col]
            out[db_col] = float(val) if pd.notna(val) else None
    return out


def persist_walk_forward_run(
    run_id: str,
    df: pd.DataFrame,
    params: dict[str, Any],
    csv_path: str,
) -> None:
    """Persist walk-forward run to SQLite.

    Args:
        run_id: Full UUID string.
        df: DataFrame with run_id column, segment, and metric columns.
        params: Dict with start_date, end_date, min_train_months, test_months, expanding.
        csv_path: Full path to run-specific CSV file.
    """
    conn = _get_conn()
    try:
        _create_tables(conn)
        overall = df[df["segment"] == "OVERALL"]
        if overall.empty:
            raise ValueError("DataFrame must contain OVERALL row")
        ov = overall.iloc[0]
        ov_vals = _row_to_db_values(ov)
        n_segments = len(df[df["segment"] != "OVERALL"])
        experiment_type = _get_experiment_type(params)
        conn.execute(
            """
            INSERT INTO model_runs (
                run_id, created_at, start_date, end_date,
                min_train_months, test_months, expanding, n_segments, csv_path,
                experiment_type,
                strategy_cagr, strategy_sharpe, strategy_maxdd, strategy_vol,
                strategy_hit_rate, strategy_turnover,
                spy_cagr, spy_sharpe, spy_maxdd, spy_vol,
                b60_40_cagr, b60_40_sharpe, b60_40_maxdd, b60_40_vol,
                equal_weight_cagr, equal_weight_sharpe, equal_weight_maxdd, equal_weight_vol,
                risk_on_off_cagr, risk_on_off_sharpe, risk_on_off_maxdd, risk_on_off_vol
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                run_id,
                datetime.utcnow().isoformat(),
                params["start_date"],
                params["end_date"],
                params["min_train_months"],
                params["test_months"],
                1 if params["expanding"] else 0,
                n_segments,
                csv_path,
                experiment_type,
                ov_vals.get("strategy_cagr"),
                ov_vals.get("strategy_sharpe"),
                ov_vals.get("strategy_maxdd"),
                ov_vals.get("strategy_vol"),
                ov_vals.get("strategy_hit_rate"),
                ov_vals.get("strategy_turnover"),
                ov_vals.get("spy_cagr"),
                ov_vals.get("spy_sharpe"),
                ov_vals.get("spy_maxdd"),
                ov_vals.get("spy_vol"),
                ov_vals.get("b60_40_cagr"),
                ov_vals.get("b60_40_sharpe"),
                ov_vals.get("b60_40_maxdd"),
                ov_vals.get("b60_40_vol"),
                ov_vals.get("equal_weight_cagr"),
                ov_vals.get("equal_weight_sharpe"),
                ov_vals.get("equal_weight_maxdd"),
                ov_vals.get("equal_weight_vol"),
                ov_vals.get("risk_on_off_cagr"),
                ov_vals.get("risk_on_off_sharpe"),
                ov_vals.get("risk_on_off_maxdd"),
                ov_vals.get("risk_on_off_vol"),
            ),
        )
        for _, r in df.iterrows():
            seg = r["segment"]
            if seg == "OVERALL":
                segment_idx = -1
                is_overall = 1
                train_start = ""
                train_end = ""
                test_start = str(r.get("test_start", ""))
                test_end = str(r.get("test_end", ""))
            else:
                segment_idx = int(seg)
                is_overall = 0
                train_start = str(r["train_start"])
                train_end = str(r["train_end"])
                test_start = str(r["test_start"])
                test_end = str(r["test_end"])
            vals = _row_to_db_values(r)
            conn.execute(
                """
                INSERT INTO walk_forward_segments (
                    run_id, segment_idx, train_start, train_end, test_start, test_end, is_overall,
                    strategy_cagr, strategy_sharpe, strategy_maxdd, strategy_vol,
                    strategy_hit_rate, strategy_turnover,
                    spy_cagr, spy_sharpe, spy_maxdd, spy_vol,
                    b60_40_cagr, b60_40_sharpe, b60_40_maxdd, b60_40_vol,
                    equal_weight_cagr, equal_weight_sharpe, equal_weight_maxdd, equal_weight_vol,
                    risk_on_off_cagr, risk_on_off_sharpe, risk_on_off_maxdd, risk_on_off_vol
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    segment_idx,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                    is_overall,
                    vals.get("strategy_cagr"),
                    vals.get("strategy_sharpe"),
                    vals.get("strategy_maxdd"),
                    vals.get("strategy_vol"),
                    vals.get("strategy_hit_rate"),
                    vals.get("strategy_turnover"),
                    vals.get("spy_cagr"),
                    vals.get("spy_sharpe"),
                    vals.get("spy_maxdd"),
                    vals.get("spy_vol"),
                    vals.get("b60_40_cagr"),
                    vals.get("b60_40_sharpe"),
                    vals.get("b60_40_maxdd"),
                    vals.get("b60_40_vol"),
                    vals.get("equal_weight_cagr"),
                    vals.get("equal_weight_sharpe"),
                    vals.get("equal_weight_maxdd"),
                    vals.get("equal_weight_vol"),
                    vals.get("risk_on_off_cagr"),
                    vals.get("risk_on_off_sharpe"),
                    vals.get("risk_on_off_maxdd"),
                    vals.get("risk_on_off_vol"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_latest_run() -> dict[str, Any] | None:
    """Return metadata for the most recent run, or None if no runs exist."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM model_runs ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        conn.close()


def list_runs() -> list[dict[str, Any]]:
    """Return all runs with summary metrics, newest first."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT run_id, created_at, n_segments, strategy_sharpe, strategy_cagr, "
            "strategy_maxdd, strategy_vol, strategy_turnover, spy_sharpe, csv_path, experiment_type "
            "FROM model_runs ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def compare_runs(run_id_1: str, run_id_2: str) -> dict[str, Any]:
    """Return side-by-side comparison of two runs."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM model_runs WHERE run_id IN (?, ?)",
            (run_id_1, run_id_2),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        if len(rows) == 0:
            raise ValueError(
                f"Neither run_id found. run_id_1={run_id_1!r}, run_id_2={run_id_2!r}"
            )
        found_ids = {r[0] for r in rows}
        if run_id_1 not in found_ids:
            raise ValueError(f"run_id_1 not found: {run_id_1!r}")
        if run_id_2 not in found_ids:
            raise ValueError(f"run_id_2 not found: {run_id_2!r}")
        by_id = {r[0]: dict(zip(cols, r)) for r in rows}
        return {"run_1": by_id[run_id_1], "run_2": by_id[run_id_2]}
    finally:
        conn.close()


def get_run_segments(run_id: str) -> pd.DataFrame:
    """Return segments for a run (excludes OVERALL row)."""
    conn = _get_conn()
    try:
        df = pd.read_sql(
            "SELECT * FROM walk_forward_segments WHERE run_id = ? AND is_overall = 0 "
            "ORDER BY segment_idx",
            conn,
            params=(run_id,),
        )
        return df
    finally:
        conn.close()
