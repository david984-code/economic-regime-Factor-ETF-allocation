"""Paper order submission via IBKR (paper only). No live trading support.

Submits orders from a RebalancePreview; supports market or limit order type from config.
Collects order IDs and statuses; structured logging. Requires explicit enable and paper-only config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Paper gateway port; live = 7496 — we never use it
PAPER_PORT = 4002


@dataclass
class SubmittedOrderResult:
    """Result of one submitted order: symbol, side, shares, order_id, status, message."""

    symbol: str
    side: str
    shares: int
    order_id: int
    status: str
    message: str = ""
    perm_id: int = 0


def _load_submission_config() -> dict[str, Any]:
    """Load paper_trading.yaml; require paper_only and return rebalance section."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: uv add pyyaml") from None
    config_path = PROJECT_ROOT / "config" / "paper_trading.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Paper trading config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not raw.get("paper_only", True):
        raise ValueError("paper_only must be true; live trading is not supported")
    port = int(raw.get("port", PAPER_PORT))
    if port != PAPER_PORT:
        logger.warning(
            "Config port %s is not paper port %s; submission may be disabled", port, PAPER_PORT
        )
    return raw


def submit_paper_orders(
    proposed_orders: list[Any],
    order_type: str = "MKT",
    account: str = "",
) -> list[SubmittedOrderResult]:
    """Submit paper orders to IBKR. proposed_orders are OrderPreviewRow-like (symbol, side, shares, price_used).

    order_type: MKT = market, LMT = limit (uses price_used).
    Returns list of SubmittedOrderResult with order_id and status. Paper port only.
    """
    try:
        from ib_insync import IB, LimitOrder, MarketOrder
        from ib_insync.contract import Stock
        from ib_insync.order import OrderStatus
    except ImportError as e:
        raise ImportError(
            "ib_insync required for order submission. Install with: uv add ib-insync"
        ) from e

    config = _load_submission_config()
    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", PAPER_PORT))
    client_id = int(config.get("client_id", 1))
    acc = account or (config.get("account") or "").strip()
    if port != PAPER_PORT:
        raise ValueError(f"Order submission is paper-only. Port must be {PAPER_PORT}, got {port}")

    ib = IB()
    try:
        # Non-readonly connection required for order placement (paper only)
        ib.connect(
            host=host, port=port, clientId=client_id, timeout=10, readonly=False, account=acc
        )
    except Exception as e:
        logger.exception("Failed to connect to IB Gateway for order submission: %s", e)
        raise ConnectionError(f"Cannot connect to IB Gateway at {host}:{port}: {e}") from e

    results: list[SubmittedOrderResult] = []
    trades: list[Any] = []
    try:
        for i, o in enumerate(proposed_orders):
            symbol = getattr(o, "symbol", None) or (o.get("symbol") if isinstance(o, dict) else "")
            side = getattr(o, "side", None) or (o.get("side") if isinstance(o, dict) else "")
            shares = int(
                getattr(o, "shares", 0) or (o.get("shares", 0) if isinstance(o, dict) else 0)
            )
            price_used = float(
                getattr(o, "price_used", 0)
                or (o.get("price_used", 0) if isinstance(o, dict) else 0)
            )
            if not symbol or shares <= 0:
                logger.warning("Skipping invalid order row: symbol=%s shares=%s", symbol, shares)
                continue
            action = "BUY" if side.upper() == "BUY" else "SELL"
            contract = Stock(symbol, "SMART", "USD")
            ot = (order_type or "MKT").upper()
            if ot == "LMT" and price_used > 0:
                order = LimitOrder(action, float(shares), price_used)
            else:
                order = MarketOrder(action, float(shares))
            if acc:
                order.account = acc
            try:
                trade = ib.placeOrder(contract, order)
                trades.append(trade)
                order_id = trade.order.orderId
                status = trade.orderStatus.status
                perm_id = getattr(trade.order, "permId", 0) or 0
                logger.info(
                    "Submitted paper order: symbol=%s side=%s shares=%s orderId=%s status=%s",
                    symbol,
                    action,
                    shares,
                    order_id,
                    status,
                )
                results.append(
                    SubmittedOrderResult(
                        symbol=symbol,
                        side=action,
                        shares=shares,
                        order_id=order_id,
                        status=status,
                        message="",
                        perm_id=perm_id,
                    )
                )
            except Exception as e:
                logger.exception(
                    "Failed to place order for %s %s %s: %s", action, shares, symbol, e
                )
                results.append(
                    SubmittedOrderResult(
                        symbol=symbol,
                        side=action,
                        shares=shares,
                        order_id=0,
                        status="Error",
                        message=str(e),
                    )
                )
        # Poll order status for a short window to capture Submitted / Filled / Cancelled
        if trades:
            poll_seconds = 10
            interval = 2
            for elapsed in range(0, poll_seconds, interval):
                ib.sleep(interval)
                all_done = True
                for idx, trade in enumerate(trades):
                    if idx >= len(results):
                        break
                    status = trade.orderStatus.status
                    results[idx] = SubmittedOrderResult(
                        symbol=results[idx].symbol,
                        side=results[idx].side,
                        shares=results[idx].shares,
                        order_id=results[idx].order_id,
                        status=status,
                        message=results[idx].message,
                        perm_id=results[idx].perm_id,
                    )
                    if status not in OrderStatus.DoneStates:
                        all_done = False
                    logger.debug("Order %s status: %s", results[idx].order_id, status)
                if all_done:
                    logger.info("All orders in terminal state after %.0fs", elapsed + interval)
                    break
            for r in results:
                logger.info("Final order state: orderId=%s status=%s", r.order_id, r.status)
    finally:
        try:
            ib.disconnect()
            logger.info("Disconnected from IB Gateway after order submission")
        except Exception as e:
            logger.warning("Disconnect warning: %s", e)

    return results
