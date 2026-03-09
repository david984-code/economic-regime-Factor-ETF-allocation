"""List walk-forward runs from SQLite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.model_results_db import get_latest_run, list_runs

if __name__ == "__main__":
    latest = get_latest_run()
    if latest is None:
        print("No runs found. Run walk-forward first.")
        sys.exit(0)
    print("Latest run:")
    print(f"  run_id: {latest['run_id']}")
    print(f"  created_at: {latest['created_at']}")
    print(f"  n_segments: {latest['n_segments']}")
    sh = latest.get("strategy_sharpe")
    cagr = latest.get("strategy_cagr")
    print(f"  strategy_sharpe: {sh:.4f}" if sh is not None else "  strategy_sharpe: N/A")
    print(f"  strategy_cagr: {cagr:.4f}" if cagr is not None else "  strategy_cagr: N/A")
    print()
    print("All runs:")
    for r in list_runs():
        sh = r.get("strategy_sharpe")
        cagr = r.get("strategy_cagr")
        sh_s = f"{sh:.3f}" if sh is not None else "N/A"
        cagr_s = f"{cagr:.3f}" if cagr is not None else "N/A"
        print(f"  {r['run_id'][:8]}... | {r['created_at']} | n={r['n_segments']} | "
              f"Sharpe={sh_s} | CAGR={cagr_s}")
