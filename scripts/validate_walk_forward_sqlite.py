"""Validate SQLite walk-forward data matches CSV for the same run_id."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation.model_results_db import (
    MODEL_RESULTS_DB,
    get_latest_run,
    _CSV_TO_DB,
)

TOLERANCE = 1e-9


def validate_run(run_id: str | None = None) -> int:
    """Validate SQLite matches CSV for run_id. Defaults to latest run.

    Returns:
        0 on success, 1 on mismatch.
    """
    if run_id is None:
        latest = get_latest_run()
        if latest is None:
            print("ERROR: No runs in SQLite. Run walk-forward first.")
            return 1
        run_id = latest["run_id"]
        print(f"Using latest run_id: {run_id}")

    csv_path = OUTPUTS_DIR / f"walk_forward_{run_id}.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    csv_df = pd.read_csv(csv_path)
    if "run_id" not in csv_df.columns:
        print("ERROR: CSV missing run_id column.")
        return 1

    csv_df = csv_df[csv_df["run_id"] == run_id]
    if csv_df.empty:
        print(f"ERROR: No rows with run_id={run_id} in CSV.")
        return 1

    import sqlite3

    conn = sqlite3.connect(str(MODEL_RESULTS_DB))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT * FROM walk_forward_segments WHERE run_id = ? "
            "ORDER BY CASE WHEN segment_idx = -1 THEN 1 ELSE 0 END, segment_idx",
            (run_id,),
        )
        db_rows = cur.fetchall()
    finally:
        conn.close()

    if len(db_rows) != len(csv_df):
        print(
            f"ERROR: Row count mismatch. CSV={len(csv_df)}, SQLite={len(db_rows)}"
        )
        return 1

    csv_df = csv_df.sort_values(
        "segment",
        key=lambda s: s.map(lambda x: 999999 if x == "OVERALL" else int(x)),
    )
    csv_df = csv_df.reset_index(drop=True)

    for i, db_row in enumerate(db_rows):
        db_row = dict(db_row)
        csv_row = csv_df.iloc[i]

        segment_idx = db_row["segment_idx"]
        is_overall = db_row["is_overall"] == 1

        if is_overall:
            if csv_row["segment"] != "OVERALL":
                print(f"ERROR: Row {i}: segment mismatch. DB=OVERALL, CSV={csv_row['segment']}")
                return 1
        else:
            if int(csv_row["segment"]) != segment_idx:
                print(f"ERROR: Row {i}: segment_idx mismatch. DB={segment_idx}, CSV={csv_row['segment']}")
                return 1

        for csv_col, db_col in _CSV_TO_DB.items():
            if csv_col not in csv_row.index:
                continue
            csv_val = csv_row[csv_col]
            db_val = db_row.get(db_col)
            if pd.isna(csv_val) and db_val is None:
                continue
            if pd.isna(csv_val) or db_val is None:
                print(f"ERROR: Row {i} {csv_col}: CSV={csv_val!r} DB={db_val!r} (null mismatch)")
                return 1
            csv_f = float(csv_val)
            db_f = float(db_val)
            if abs(csv_f - db_f) > TOLERANCE:
                print(
                    f"ERROR: Row {i} {csv_col}: CSV={csv_f} DB={db_f} diff={abs(csv_f - db_f)}"
                )
                return 1

    print(f"OK: SQLite matches CSV for run_id={run_id} ({len(csv_df)} rows)")
    return 0


if __name__ == "__main__":
    run_id_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(validate_run(run_id=run_id_arg))
