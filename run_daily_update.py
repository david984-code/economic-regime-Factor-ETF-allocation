"""Daily Portfolio Update Runner

Orchestrates the quantitative research pipeline:
1. Regime classification (FRED + rule-based)
2. ML regime forecast (next month)
3. Portfolio optimization (Sortino)
4. Backtest and save results

Designed for Windows Task Scheduler at 8:30 AM and 4:30 PM ET.
"""

import sys

from src.pipeline import run_daily_pipeline

if __name__ == "__main__":
    sys.exit(run_daily_pipeline())
