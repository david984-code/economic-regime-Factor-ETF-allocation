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
    """
    from src.data.market_ingestion import fetch_prices
    from src.utils.database import Database

    prices = fetch_prices(start="2024-01-01")
    db = Database()
    regime_df = db.load_regime_labels()
    allocations = db.load_optimal_allocations()
    db.close()
    allocations = {str(k).strip(): v for k, v in allocations.items()}
    for alloc in allocations.values():
        alloc.setdefault("cash", 0.0)
    regime, risk_on = _latest_regime_and_riskon(regime_df)
    # NOTE: prior versions blended an ML regime forecast into risk_on here
    # (removed 2026-06-07 -- unvalidated, never in the walk-forward path).
    logger.info("Target weight generation: regime=%s risk_on=%.3f", regime, risk_on)
    base_weights = _select_base_weights(allocations, regime, risk_on)
    scaled = _apply_vol_scaling(prices, base_weights)
    return scaled


def _latest_regime_and_riskon(regime_df: pd.DataFrame) -> tuple[str, float]:
    regime_df.index = pd.to_datetime(regime_df.index)
    regime_df = regime_df[regime_df.index.notna()]
    latest = regime_df.iloc[-1]
    regime = str(latest["regime"]).strip()
    risk_on = float(latest["risk_on"]) if pd.notna(latest["risk_on"]) else 0.5
    return regime, risk_on


def _select_base_weights(
    allocations: dict[str, dict[str, float]], regime: str, risk_on: float
) -> dict[str, float]:
    from src.backtest.engine import _avg_alloc, _blend_alloc
    from src.config import ASSETS, REGIME_ALIASES, RISK_OFF_REGIMES, RISK_ON_REGIMES
    w_risk_on = _avg_alloc(allocations, RISK_ON_REGIMES, ASSETS)
    w_risk_off = _avg_alloc(allocations, RISK_OFF_REGIMES, ASSETS)
    regime_mapped = REGIME_ALIASES.get(regime, regime)
    if regime_mapped == "Stagflation" and "Stagflation" in allocations:
        return {str(k): float(v) for k, v in allocations["Stagflation"].items()}
    return _blend_alloc(w_risk_off, w_risk_on, risk_on, ASSETS)


def _apply_vol_scaling(prices, base_weights: dict[str, float]) -> dict[str, float]:
    import polars as pl
    from src.allocation.vol_scaling import vol_scaled_weights
    from src.config import TICKERS, VOL_LOOKBACK
    returns = prices[TICKERS].pct_change().dropna()
    trailing_pl = pl.from_pandas(returns.tail(VOL_LOOKBACK))
    scaled = vol_scaled_weights(base_weights, trailing_pl, list(TICKERS))
    logger.info(
        "Target weights: %s",
        {k: f"{v:.3f}" for k, v in sorted(scaled.items(), key=lambda x: x[1], reverse=True) if v > 0.005},
    )
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


def _load_paper_config(result: dict[str, Any]) -> dict[str, Any] | None:
    """Load and validate config/paper_trading.yaml. Mutates result on error."""
    try:
        import yaml
    except ImportError as e:
        result["action"] = "error"
        result["error"] = f"PyYAML not installed: {e}"
        logger.error("[REBALANCE] %s", result["error"])
        return None
    config_path = PROJECT_ROOT / "config" / "paper_trading.yaml"
    if not config_path.exists():
        result["action"] = "error"
        result["error"] = "paper_trading.yaml not found"
        logger.error("[REBALANCE] %s", result["error"])
        return None
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not config.get("paper_only", True):
        result["action"] = "error"
        result["error"] = "paper_only must be true"
        logger.error("[REBALANCE] %s", result["error"])
        return None
    if not config.get("trading_enabled", False):
        result["action"] = "dry_run"
        result["reason"] = "trading_enabled=false in config"
        logger.info("[REBALANCE] Trading not enabled — dry run only")
    return config


def _build_preview(config: dict[str, Any], csv_path, live, result: dict[str, Any]):
    """Run dry-run preview and stash order summary into result."""
    from src.execution.create_orders import DEFAULT_TAU
    from src.execution.monthly_rebalance_runner import run_dry_run
    rebal = config.get("rebalance") or {}
    tau = float(rebal.get("tau", DEFAULT_TAU))
    report_dir = OUTPUTS_DIR / "rebalance" / "reports"
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
    return preview, report_dir, rebal


def _check_safety_gates(preview, config, rebal, report_dir, result: dict[str, Any]):
    """Return marker_path if safe to submit, else None and result is annotated."""
    from src.execution.safety import (
        duplicate_run_refuse,
        pre_trade_checks,
        safety_config_from_paper_config,
    )
    safety_cfg = safety_config_from_paper_config(config)
    marker_path = report_dir / "last_paper_submission.json"
    if rebal.get("duplicate_run_protection", True) and duplicate_run_refuse(marker_path):
        result["action"] = "skip"
        result["reason"] = "Already submitted today (duplicate protection)"
        logger.info("[REBALANCE] %s", result["reason"])
        return None
    try:
        pre_trade_checks(preview, trading_enabled=True, paper_only=True, cfg=safety_cfg)
    except ValueError as e:
        result["action"] = "blocked"
        result["error"] = str(e)
        logger.error("[REBALANCE] Safety check failed: %s", e)
        return None
    return marker_path


def _reconcile_post_trade(weights, config, report_dir, result: dict[str, Any]) -> None:
    """Best-effort post-trade reconciliation; failures are non-fatal."""
    try:
        from src.execution.reconcile_post_trade import run_reconciliation
        recon_client_id = config.get("reconciliation_client_id", 2)
        rec = run_reconciliation(weights, report_dir=report_dir, client_id_override=recon_client_id)
        result["reconciliation_max_delta"] = rec.max_abs_delta
        logger.info("[REBALANCE] Reconciliation max_delta=%.4f", rec.max_abs_delta)
    except Exception as e:
        logger.warning("[REBALANCE] Reconciliation failed (non-fatal): %s", e)


def _submit_with_safety(
    preview, config, weights, rebal, report_dir, result: dict[str, Any]
) -> None:
    """Run safety checks, submit paper orders, reconcile post-trade."""
    from src.execution.safety import write_submission_marker
    from src.execution.submit_orders import submit_paper_orders
    marker_path = _check_safety_gates(preview, config, rebal, report_dir, result)
    if marker_path is None:
        return
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
    _reconcile_post_trade(weights, config, report_dir, result)


def _save_rebalance_report(report_dir, result: dict[str, Any]) -> None:
    """Persist auto-rebalance result JSON."""
    report_path = report_dir / f"auto_rebalance_{datetime.now().strftime('%Y%m%d')}.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("[REBALANCE] Report saved: %s", report_path)


def run_auto_rebalance() -> dict[str, Any]:
    """Full auto-rebalance: check day, generate weights, submit orders.

    Safe to call every day; no-ops on non-rebalance days. Returns a dict
    with `action` ∈ {skip, dry_run, submitted, blocked, error} and the
    relevant details.
    """
    result: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "action": "skip"}
    if not is_rebalance_day():
        result["reason"] = "Not rebalance day (not 1st trading day of month)"
        logger.info("[REBALANCE] Skipping — not rebalance day")
        return result
    logger.info("[REBALANCE] Today is rebalance day — starting auto-rebalance")
    result["action"] = "rebalance"
    logger.info("[REBALANCE] Generating target weights...")
    weights = generate_target_weights()
    result["target_weights"] = weights
    csv_path = save_target_weights(weights)
    config = _load_paper_config(result)
    if config is None:
        return result
    from src.execution.monthly_rebalance_runner import fetch_live_positions_and_nav
    live = fetch_live_positions_and_nav()
    if live is None:
        result["action"] = "dry_run"
        result["reason"] = "IBKR not connected — dry run only"
        logger.warning("[REBALANCE] IBKR not connected — running dry-run with mock")
    preview, report_dir, rebal = _build_preview(config, csv_path, live, result)
    if config.get("trading_enabled") and live is not None:
        _submit_with_safety(preview, config, weights, rebal, report_dir, result)
    _save_rebalance_report(report_dir, result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    r = run_auto_rebalance()
    print(json.dumps(r, indent=2, default=str))
