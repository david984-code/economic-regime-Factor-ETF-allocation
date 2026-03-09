"""Backward-compatibility wrapper for portfolio optimization.

Use: python -m src.optimizer
New import: from src.allocation.optimizer import run_optimizer, main
"""

from src.allocation.optimizer import main

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
