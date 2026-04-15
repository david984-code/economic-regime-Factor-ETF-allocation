"""Backward-compatibility wrapper for backtest.

Use: python -m src.backtest_polars
New import: from src.backtest.engine import run_backtest, main
"""

from src.backtest.engine import main

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
