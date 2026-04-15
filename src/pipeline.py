"""Daily pipeline orchestration for quantitative research."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config import LOGS_DIR, START_DATE, get_end_date
from src.data.pipeline_data import PipelineData
from src.utils.fred_key import get_fred_api_key
from src.utils.logging_config import setup_logging
from src.utils.ticker_universe import parse_tickers_arg, resolve_tickers

logger = logging.getLogger(__name__)


def run_daily_pipeline(
    cli_tickers: list[str] | None = None,
    dry_run: bool = False,
    config_path: str | None = None,
    no_cache: bool = False,
) -> int:
    """Run the complete daily pipeline in-process.

    Steps:
        1. Regime classification (FRED + rule-based)
        2. Regime forecast (ML)
        3. Portfolio optimization (Sortino)
        4. Backtest and save results

    Args:
        cli_tickers: Override default ticker universe.
        dry_run: If True, run all steps but skip file writes and execution.
        config_path: Path to YAML config file (e.g. config/paper_trading.yaml).
        no_cache: If True, bypass all local data caches and fetch fresh.

    Returns:
        0 on success, 1 on failure.
    """
    load_dotenv()

    if config_path:
        import yaml

        cfg_file = Path(config_path)
        if not cfg_file.exists():
            logger.error("Config file not found: %s", config_path)
            return 1
        with open(cfg_file) as f:
            _user_cfg = yaml.safe_load(f)
        logger.info("[CONFIG] Loaded config from %s", config_path)
        if _user_cfg.get("dry_run", False):
            dry_run = True

    if dry_run:
        logger.info("[CONFIG] DRY RUN — no file writes, no order execution")
    if no_cache:
        logger.info("[CONFIG] NO CACHE — forcing fresh data downloads")

    api_key = get_fred_api_key()
    if not api_key:
        logger.error(
            "FRED API key not found. Set FRED_API_KEY or FRED_API_KEY_FILE (see README)."
        )
        return 1

    start = datetime.now()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"daily_update_{start.strftime('%Y%m%d')}.log"
    setup_logging(log_file=log_file)

    logger.info("=" * 80)
    logger.info("STARTING DAILY PIPELINE - %s", start.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)

    timings: dict[str, float] = {}

    # Step 1: Regime classification (FRED only, no market data)
    logger.info("\n[STEP] Regime classification")
    try:
        from src.data.fred_ingestion import get_fred_cache_stats

        t0 = time.perf_counter()
        _run_regime_classification(api_key)
        timings["regime_classification"] = time.perf_counter() - t0
        fred_hits, fred_misses = get_fred_cache_stats()
        logger.info("[FRED] Cache stats: %d hits, %d misses", fred_hits, fred_misses)
        logger.info(
            "[OK] Completed: Regime classification (%.2fs)",
            timings["regime_classification"],
        )
    except Exception as e:
        logger.exception("[FAIL] Regime classification: %s", e)
        return 1

    # Fetch market data ONCE for all downstream steps
    logger.info(
        "\n[STEP] Fetching market data (shared by forecast, optimizer, backtest)"
    )
    t0_fetch = time.perf_counter()
    tickers = resolve_tickers(cli_tickers)
    logger.info("[CONFIG] Universe: %d tickers: %s", len(tickers), ",".join(tickers))
    pipeline_data = PipelineData(
        tickers=tickers, start=START_DATE, end=get_end_date(), use_cache=not no_cache
    )
    pipeline_data.get_prices()  # Triggers fetch, populates cache
    timings["data_fetch"] = time.perf_counter() - t0_fetch
    logger.info(
        "[DATA] Market data fetched in %.2fs (%d tickers, %d rows) - will be reused by 3 steps",
        timings["data_fetch"],
        len(pipeline_data.get_prices().columns),
        len(pipeline_data.get_prices()),
    )

    # Steps 2–4: Use shared pipeline_data (no re-fetch)
    steps = [
        ("regime_forecast", "Regime forecast", _run_regime_forecast, [pipeline_data]),
        ("optimizer", "Portfolio optimization", _run_optimizer, [pipeline_data]),
        ("backtest", "Backtest", _run_backtest, [pipeline_data]),
    ]

    for key, name, func, args in steps:
        logger.info("\n[STEP] %s", name)
        try:
            t0 = time.perf_counter()
            func(*args)
            timings[key] = time.perf_counter() - t0
            logger.info("[OK] Completed: %s (%.2fs)", name, timings[key])
        except Exception as e:
            logger.exception("[FAIL] %s: %s", name, e)
            elapsed = (datetime.now() - start).total_seconds()
            logger.error("Pipeline failed after %.1fs", elapsed)
            return 1

    # Non-critical: auto-rebalance (1st trading day of month only)
    logger.info("\n[STEP] Auto-rebalance check")
    try:
        t0 = time.perf_counter()
        from src.execution.auto_rebalance import run_auto_rebalance

        rebal_result = run_auto_rebalance()
        timings["auto_rebalance"] = time.perf_counter() - t0
        logger.info(
            "[OK] Auto-rebalance: %s (%.2fs)",
            rebal_result.get("action", "?"),
            timings["auto_rebalance"],
        )
    except Exception as e:
        logger.warning("[SKIP] Auto-rebalance failed (non-critical): %s", e)

    # Non-critical: daily report
    logger.info("\n[STEP] Daily report")
    try:
        t0 = time.perf_counter()
        from src.daily_report import send_daily_report

        send_daily_report()
        timings["daily_report"] = time.perf_counter() - t0
        logger.info("[OK] Completed: Daily report (%.2fs)", timings["daily_report"])
    except Exception as e:
        logger.warning("[SKIP] Daily report failed (non-critical): %s", e)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: All steps completed in %.1fs", elapsed)
    logger.info("TIMING BY STEP:")
    for k in [
        "data_fetch",
        "regime_classification",
        "regime_forecast",
        "optimizer",
        "backtest",
        "auto_rebalance",
        "daily_report",
    ]:
        if k in timings:
            logger.info("  %s: %.2fs", k, timings[k])
    logger.info(
        "SUMMARY: Market data fetched once in %.2fs (3 steps reused cache)",
        timings.get("data_fetch", 0),
    )
    logger.info("=" * 80)
    return 0


def _run_regime_classification(api_key: str) -> None:
    from src.models.regime_classifier import RegimeClassifier

    RegimeClassifier(api_key).run()


def _run_regime_forecast(pipeline_data: PipelineData) -> None:
    from src.models.regime_forecast import main as forecast_main

    forecast_main(pipeline_data=pipeline_data)


def _run_optimizer(pipeline_data: PipelineData) -> None:
    from src.allocation.optimizer import run_optimizer

    run_optimizer(pipeline_data=pipeline_data)


def _run_backtest(pipeline_data: PipelineData) -> None:
    from src.backtest.engine import run_backtest

    run_backtest(pipeline_data=pipeline_data)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily pipeline: regime classification, forecast, optimizer, backtest.",
    )
    p.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated symbols (overrides PIPELINE_TICKERS env and config default).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run all steps but skip file writes and order execution.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML config file (e.g. config/paper_trading.yaml).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass local data caches and force fresh downloads.",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys

    load_dotenv()
    args = _parse_cli()
    cli_list = parse_tickers_arg(args.tickers)
    sys.exit(
        run_daily_pipeline(
            cli_tickers=cli_list,
            dry_run=args.dry_run,
            config_path=args.config,
            no_cache=args.no_cache,
        )
    )
