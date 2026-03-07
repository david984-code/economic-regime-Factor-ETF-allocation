"""Daily Portfolio Update Runner

This script orchestrates the complete pipeline for daily portfolio updates:
1. Fetch latest economic data and classify regimes
2. Optimize portfolio allocations per regime
3. Run backtest with Polars for performance metrics
4. Store results in database

Designed to run automatically before market open (8:30 AM ET) and
after market close (4:30 PM ET) via Windows Task Scheduler.
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"daily_update_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def run_module(module_name: str) -> bool:
    """Run a Python module and return success status.

    Args:
        module_name: Module to run (e.g., 'src.economic_regime')

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting: {module_name}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"[OK] Completed: {module_name}")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] Failed: {module_name}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error in {module_name}: {e}")
        return False


def main() -> int:
    """Run the complete daily update pipeline.

    Returns:
        0 if successful, 1 if any step fails
    """
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info(f"STARTING DAILY UPDATE - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    steps = [
        ("src.economic_regime", "Fetch economic data & classify regimes"),
        ("src.regime_forecast", "ML regime forecast (next month)"),
        ("src.optimizer", "Optimize portfolio allocations"),
        ("src.backtest_polars", "Run backtest & update database"),
    ]

    results = []
    for module, description in steps:
        logger.info(f"\n[STEP] {description}")
        success = run_module(module)
        results.append((module, success))

        if not success:
            logger.error(f"Pipeline stopped due to failure in {module}")
            break

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for module, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{module}: {status}")

    all_passed = all(success for _, success in results)
    if all_passed:
        logger.info(f"\n[SUCCESS] All steps completed successfully in {elapsed:.1f}s")
        logger.info(f"[SUCCESS] Database updated: outputs/allocations.db")
        return 0
    else:
        logger.error(f"\n[FAILED] Pipeline failed after {elapsed:.1f}s")
        return 1


if __name__ == "__main__":
    sys.exit(main())
