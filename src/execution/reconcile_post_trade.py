"""Post-trade reconciliation: fetch positions, compare target vs executed weights, save report.

Uses broker portfolio (market value) to compute executed weights and compares to target.
Saves reconciliation report to report_dir. No order submission; read-only after trade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.execution.create_orders import PositionRow, positions_to_current_weights

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationReport:
    """Post-trade reconciliation: target vs executed weights and deltas."""

    nav: float
    target_weights: dict[str, float]
    executed_weights: dict[str, float]
    weight_delta: dict[str, float]
    max_abs_delta: float
    run_id: str


def _load_paper_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required.") from None
    config_path = PROJECT_ROOT / "config" / "paper_trading.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _nav_from_summary(summary: list[dict[str, str]]) -> float:
    for row in summary:
        if (row.get("tag") or "").strip() == "NetLiquidation":
            try:
                return float(row.get("value") or 0)
            except (TypeError, ValueError):
                pass
    for row in summary:
        if (row.get("tag") or "").strip() == "TotalCashValue":
            try:
                return float(row.get("value") or 0)
            except (TypeError, ValueError):
                pass
    return 0.0


def _portfolio_to_position_rows(portfolio: list[dict[str, Any]]) -> list[PositionRow]:
    out: list[PositionRow] = []
    for p in portfolio:
        symbol = (p.get("symbol") or "").strip()
        if not symbol:
            continue
        pos = float(p.get("position", 0))
        avg = float(p.get("avgCost", 0) or 0)
        mv = p.get("marketValue")
        mp = p.get("marketPrice")
        market_value = float(mv) if mv is not None else None
        market_price = float(mp) if mp is not None else None
        if market_value is not None and market_value < 0:
            market_value = None
        if market_price is not None and market_price <= 0:
            market_price = None
        out.append(
            PositionRow(
                symbol=symbol,
                position=pos,
                avg_cost=avg,
                market_value=market_value,
                market_price=market_price,
            )
        )
    return out


def fetch_post_trade_positions_and_nav(
    client_id_override: int | None = None,
) -> tuple[list[PositionRow], float] | None:
    """Fetch current portfolio and NAV from IBKR (post-trade). Returns None on failure.

    When called immediately after order submission, pass client_id_override (e.g. reconciliation_client_id)
    to use a different client_id and avoid "Error 326: client id is already in use".
    """
    try:
        from src.execution.ibkr_adapter import IBKRPaperAdapter
    except ImportError:
        logger.warning("IBKR adapter not available")
        return None
    config: dict[str, Any] | None = None
    if client_id_override is not None:
        base = _load_paper_config()
        config = {**base, "client_id": client_id_override}
        logger.info("Using client_id=%s for post-trade fetch (avoid collision with submission session)", client_id_override)
    adapter = IBKRPaperAdapter(config=config)
    try:
        adapter.connect()
    except Exception as e:
        logger.warning("Broker connection failed: %s", e)
        return None
    try:
        summary = adapter.get_account_summary()
        nav = _nav_from_summary(summary)
        portfolio = adapter.get_portfolio()
        rows = _portfolio_to_position_rows(portfolio)
        if nav <= 0:
            nav = sum(
                (p.market_value if p.market_value is not None else p.position * p.avg_cost)
                for p in rows
            ) or 1.0
        return rows, nav
    finally:
        adapter.disconnect()


def run_reconciliation(
    target_weights: dict[str, float],
    report_dir: Path | None = None,
    positions: list[PositionRow] | None = None,
    nav: float | None = None,
    client_id_override: int | None = None,
) -> ReconciliationReport:
    """Compute executed weights from positions/NAV (or fetch from broker), compare to target, return report."""
    if positions is None or nav is None:
        live = fetch_post_trade_positions_and_nav(client_id_override=client_id_override)
        if live is None:
            raise RuntimeError("Could not fetch post-trade positions; broker unavailable")
        positions, nav = live
        logger.info("Fetched post-trade positions: %d, NAV=%.2f", len(positions), nav)
    executed_weights = positions_to_current_weights(positions, nav)
    # Align keys
    all_symbols = set(target_weights) | set(executed_weights)
    weight_delta = {s: (target_weights.get(s, 0.0) - executed_weights.get(s, 0.0)) for s in all_symbols}
    max_abs_delta = max(abs(d) for d in weight_delta.values()) if weight_delta else 0.0
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = ReconciliationReport(
        nav=nav,
        target_weights=dict(target_weights),
        executed_weights=executed_weights,
        weight_delta=weight_delta,
        max_abs_delta=max_abs_delta,
        run_id=run_id,
    )
    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"reconcile_{run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "nav": report.nav,
                    "target_weights": report.target_weights,
                    "executed_weights": report.executed_weights,
                    "weight_delta": report.weight_delta,
                    "max_abs_delta": report.max_abs_delta,
                },
                f,
                indent=2,
            )
        logger.info("Saved reconciliation report: %s", path)
    return report
