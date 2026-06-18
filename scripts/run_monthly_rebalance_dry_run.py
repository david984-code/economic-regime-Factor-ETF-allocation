"""Run monthly rebalance dry-run: load targets, broker or mock positions, report only.

No orders submitted. From repo root:
  uv run python scripts/run_monthly_rebalance_dry_run.py
  uv run python scripts/run_monthly_rebalance_dry_run.py --use-mock-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.monthly_rebalance_runner import main

if __name__ == "__main__":
    sys.exit(main())
