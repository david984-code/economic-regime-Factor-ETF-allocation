"""Backward-compatibility wrapper for regime forecast.

Use: python -m src.regime_forecast
New import: from src.models.regime_forecast import main
"""

from src.models.regime_forecast import main

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
