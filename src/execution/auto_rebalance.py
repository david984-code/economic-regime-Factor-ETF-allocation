"""Automatic monthly rebalance: generate target weights and submit paper orders.

Called by the daily pipeline. Only executes on rebalance days (1st trading day
of the month). Skips silently on non-rebalance days.

Safety: all pre-trade checks, duplicate-run protection, and paper-only
enforcement from safety.py are applied. Never submits live orders.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUTS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


def is_rebalance_day() -> bool:
    """Check if today is the 1st trading day of the month.

    Uses pandas market calendar logic: if today is a weekday and no
    trading day this month has occurred before today, it's rebalance day.
    """
    today = pd.Timestamp.today().normalize()

    # Not a weekday = not a trading day
    if today.dayofweek >= 5:
        return False

    # Check if any weekday this month preceded today
    month_start = today.replace(day=1)
    bdays = pd.bdate_range(month_start, today)
    return len(bdays) >= 1 and bdays[0] == today


def generate_target_weights() -> dict[str, float]:
    """Compute target weights from current regime, allocations, and vol scaling.

    Uses the same logic as run_backtest (blend → inv-vol scale) but only
    for the latest date, avoiding backtest trailing-weight artifacts.

    Returns:
        Dict of {ticker: weight} for the current rebalance.
    """
    import polars as pl

    from src.allocation.vol_scaling import vol_scaled_weights
    from src.backtest.engine import _avg_alloc, _blend_alloc
    from src.config import (
        ASSETS,
        REGIME_ALIASES,
        RISK_OFF_REGIMES,
        RISK_ON_REGIMES,
        TICKERS,
        VOL_LOOKBACK,
    )
    from src.data.market_ingestion import fetch_prices
    from src.utils.database import Database

    prices = fetch_prices(start="2024-01-01")
    db = Database()
    regime_df = db.load_regime_labels()
    allocations = db.load_optimal_allocations()
    forecast = None
    try:
        next_month = (pd.Timestamp.today().to_period("M") + 1).strftime("%Y-%m")
        forecast = db.load_latest_regime_forecast(next_month)
    except Exception:
        pass
    db.close()

    allocations = {str(k).strip(): v for k, v in allocations.items()}
    for alloc in allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0

    # Current regime and risk_on
    regime_df.index = pd.to_datetime(regime_df.index)
    regime_df = regime_df[regime_df.index.notna()]
    latest = regime_df.iloc[-1]
    regime = str(latest["regime"]).strip()
    risk_on = float(latest["risk_on"]) if pd.notna(latest["risk_on"]) else 0.5

    # Blend with forecast if available
    if forecast is not None:
        risk_on = 0.5 * risk_on + 0.5 * forecast["risk_on_forecast"]

    logger.info(
        "Target weight generation: regime=%s risk_on=%.3f forecast=%s",
        regime,
        risk_on,
        "yes" if forecast else "no",
    )

    # Build risk-on and risk-off sleeve averages
    w_risk_on = _avg_alloc(allocations, RISK_ON_REGIMES, ASSETS)
    w_risk_off = _avg_alloc(allocations, RISK_OFF_REGIMES, ASSETS)

    # Stagflation override: use raw Stagflation allocation
    regime_mapped = REGIME_ALIASES.get(regime, regime)
    if regime_mapped == "Stagflation" and "Stagflation" in allocations:
        base_weights = {str(k): float(v) for k, v in allocations["Stagflation"].items()}
    else:
        base_weights = _blend_alloc(w_risk_off, w_risk_on, risk_on, ASSETS)

    # Inverse-vol scaling using trailing returns
    returns = prices[TICKERS].pct_change().dropna()
    trailing = returns.tail(VOL_LOOKBACK)
    trailing_pl = pl.from_pandas(trailing)
    scaled = vol_scaled_weights(base_weights, trailing_pl, list(TICKERS))

    logger.info("Target weights: %s", {k: f"{v:.3f}" for k, v in sorted(scaled.items(), key=lambda x: x[1], reverse=True) if v > 0.005})
    return {k: float(v) for k, v in scaled.items() if v > 0.001}


def save_target_weights(weights: dict[str, float]) -> Path:
    """Save target weights to CSV for the rebalance runner."""
    rebalance_dir = OUTPUTS_DIR / "rebalance"
    rebalance_dir.mkdir(parents=True, exist_ok=True)

    csv_path = rebalance_dir / "target_weights.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "weight"])
        for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([symbol, f"{weight:.6f}"])

    json_path = rebalance_dir / "target_weights.json"
    with open(json_path, "w") as f:
        json.dump(weights, f, indent=2)

    logger.info("Target weights saved: %s", csv_path)
    return csv_path


def run_auto_rebalance() -> dict[str, Any]:
    """Full auto-rebalance: check day, generate weights, submit orders.

    Returns:
        Dict with status and details. Safe to call every day — skips on
        non-rebalance days.
    """
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "action": "skip",
    }

    # Step 1: Check if today is rebalance day
    if not is_rebalance_day():
        result["reason"] = "Not rebalance day (not 1st trading day of month)"
        logger.info("[REBALANCE] Skipping — not rebalance day")
        return result

    logger.info("[REBALANCE] Today is rebalance day — starting auto-rebalance")
    result["action"] = "rebalance"

    # Step 2: Generate target weights
    logger.info("[REBALANCE] Generating target weights...")
    weights = generate_target_weights()
    result["target_weights"] = weights
    csv_path = save_target_weights(weights)

    # Step 3: Load config and check if trading is enabled
    try:
        import yaml
    except ImportError as e:
        result["action"] = "error"
        result["error"] = f"PyYAML not installed: {e}"
        logger.error("[REBALANCE] %s", result["error"])
        return result

    config_path = PROJECT_ROOT / "config" / "paper_trading.yaml"
    if not config_path.exists():
        result["action"] = "error"
        result["error"] = "paper_trading.yaml not found"
        logger.error("[REBALANCE] %s", result["error"])
        return result

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not config.get("trading_enabled", False):
        result["action"] = "dry_run"
        result["reason"] = "trading_enabled=false in config"
        logger.info("[REBALANCE] Trading not enabled — dry run only")

    if not config.get("paper_only", True):
        result["action"] = "error"
        result["error"] = "paper_only must be true"
        logger.error("[REBALANCE] %s", result["error"])
        return result

    # Step 4: Run the rebalance (dry-run or live submission)
    from src.execution.create_orders import DEFAULT_TAU
    from src.execution.monthly_rebalance_runner import (
        fetch_live_positions_and_nav,
        run_dry_run,
    )

    rebal = config.get("rebalance") or {}
    tau = float(rebal.get("tau", DEFAULT_TAU))
    report_dir = OUTPUTS_DIR / "rebalance" / "reports"

    # Try to get live positions from IBKR
    live = fetch_live_positions_and_nav()
    if live is None:
        result["action"] = "dry_run"
        result["reason"] = "IBKR not connected — dry run only"
        logger.warning("[REBALANCE] IBKR not connected — running dry-run with mock")

    # Always run dry-run first for the report
    mock_path_str = rebal.get("mock_positions_path")
    mock_path = (PROJECT_ROOT / mock_path_str) if mock_path_str else None
    prices_path_str = rebal.get("prices_path")
    prices_path = (PROJECT_ROOT / prices_path_str) if prices_path_str else None

    preview = run_dry_run(
        target_weights_path=csv_path,
        tau=tau,
        mock_positions_path=mock_path,
        prices_path=prices_path,
        report_dir=report_dir,
        use_mock_only=(live is None),
    )

    result["turnover"] = preview.turnover_one_way
    result["order_count"] = len(preview.proposed_orders)
    result["orders"] = [
        {"symbol": o.symbol, "side": o.side, "shares": o.shares}
        for o in preview.proposed_orders
    ]

    # Step 5: Submit paper orders if trading is enabled and IBKR connected
    if config.get("trading_enabled") and live is not None:
        from src.execution.safety import (
            duplicate_run_refuse,
            pre_trade_checks,
            safety_config_from_paper_config,
            write_submission_marker,
        )
        from src.execution.submit_orders import submit_paper_orders

        safety_cfg = safety_config_from_paper_config(config)

        # Duplicate run protection
        marker_path = report_dir / "last_paper_submission.json"
        if rebal.get("duplicate_run_protection", True) and duplicate_run_refuse(marker_path):
            result["action"] = "skip"
            result["reason"] = "Already submitted today (duplicate protection)"
            logger.info("[REBALANCE] %s", result["reason"])
            return result

        # Pre-trade safety checks
        try:
            pre_trade_checks(
                preview,
                trading_enabled=True,
                paper_only=True,
                cfg=safety_cfg,
            )
        except ValueError as e:
            result["action"] = "blocked"
            result["error"] = str(e)
            logger.error("[REBALANCE] Safety check failed: %s", e)
            return result

        # Submit
        order_type = rebal.get("order_type", "MKT")
        account = config.get("account") or ""
        logger.info("[REBALANCE] Submitting %d paper orders...", len(preview.proposed_orders))
        submitted = submit_paper_orders(preview.proposed_orders, order_type=order_type, account=account)

        write_submission_marker(
            marker_path,
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            submitted_order_count=len(submitted),
            dry_run=False,
        )

        result["action"] = "submitted"
        result["submitted"] = [
            {"symbol": r.symbol, "side": r.side, "shares": r.shares, "status": r.status}
            for r in submitted
        ]
        logger.info("[REBALANCE] Submitted %d paper orders", len(submitted))

        # Post-trade reconciliation
        try:
            from src.execution.reconcile_post_trade import run_reconciliation

            recon_client_id = config.get("reconciliation_client_id", 2)
            rec = run_reconciliation(
                weights, report_dir=report_dir, client_id_override=recon_client_id
            )
            result["reconciliation_max_delta"] = rec.max_abs_delta
            logger.info("[REBALANCE] Reconciliation max_delta=%.4f", rec.max_abs_delta)
        except Exception as e:
            logger.warning("[REBALANCE] Reconciliation failed (non-fatal): %s", e)

    # Save auto-rebalance report
    report_path = report_dir / f"auto_rebalance_{datetime.now().strftime('%Y%m%d')}.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("[REBALANCE] Report saved: %s", report_path)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    r = run_auto_rebalance()
    print(json.dumps(r, indent=2, default=str))
