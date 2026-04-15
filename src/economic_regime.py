"""Backward-compatibility wrapper for regime classification.

Use: python -m src.economic_regime
New import: from src.models.regime_classifier import RegimeClassifier, main
"""

from src.models.regime_classifier import main

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
