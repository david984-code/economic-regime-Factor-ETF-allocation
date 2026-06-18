"""Dry-run order creation: target vs current weights, tau filter, share quantities.

Loads target weights, accepts current positions and NAV, computes drift, applies
tau = 0.015 no-trade band, converts to proposed share orders. No broker writes.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default tau for frozen strategy (monthly rebalance)
DEFAULT_TAU = 0.015


@dataclass
class PositionRow:
    """Single position: symbol, quantity, avg cost; optional market value/price for valuation-correct weights."""

    symbol: str
    position: float
    avg_cost: float
    market_value: float | None = None
    market_price: float | None = None


@dataclass
class OrderPreviewRow:
    """Proposed order (preview only): symbol, side, shares, approximate dollar, weight delta."""

    symbol: str
    side: str  # "BUY" | "SELL"
    shares: int
    approximate_dollar: float
    weight_delta: float
    price_used: float


@dataclass
class RebalancePreview:
    """Full dry-run rebalance result: weights comparison, proposed orders, turnover."""

    nav: float
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    tau: float
    weight_drift: dict[str, float]
    tau_filtered_delta: dict[str, float]
    proposed_orders: list[OrderPreviewRow]
    turnover_one_way: float
    symbols_missing_price: list[str] = field(default_factory=list)


def load_target_weights(path: Path) -> dict[str, float]:
    """Load target weights from CSV or JSON. Returns symbol -> weight (sum typically 1.0)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target weights file not found: {path}")

    suffix = path.suffix.lower()
    out: dict[str, float] = {}

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                out[str(k).strip()] = float(v)
        elif isinstance(data, list):
            for row in data:
                s = (row.get("symbol") or row.get("ticker") or "").strip()
                w = float(row.get("weight", row.get("target_weight", 0)))
                if s:
                    out[s] = w
        else:
            raise ValueError("Target weights JSON must be object or list of {symbol, weight}")
    elif suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = (row.get("symbol") or row.get("ticker") or "").strip()
                w_str = row.get("weight") or row.get("target_weight") or "0"
                if s:
                    out[s] = float(w_str)
    else:
        raise ValueError(f"Unsupported target weights format: {suffix}. Use .csv or .json")

    logger.info("Loaded target weights for %d symbols from %s", len(out), path)
    return out


def validate_target_weights_for_execution(
    weights: dict[str, float],
    *,
    allow_leverage: bool = False,
    tolerance: float = 0.02,
    max_gross_without_leverage: float = 1.0,
) -> float:
    """Ensure target weights sum to ~1.0 unless leverage is explicitly allowed.

    Returns the sum for logging. Raises ValueError if invalid.
    """
    total = sum(float(v) for v in weights.values())
    if allow_leverage:
        if total < max_gross_without_leverage - tolerance:
            raise ValueError(
                f"Target weights sum to {total:.6f}; below minimum {max_gross_without_leverage - tolerance:.6f} "
                f"(allow_leverage=true)."
            )
        return total
    if total > max_gross_without_leverage + tolerance:
        raise ValueError(
            f"Target weights sum to {total:.6f}, expected ~1.0 (tolerance {tolerance}). "
            "This often indicates cash was added without scaling equity sleeves, or unintended leverage. "
            "Fix the weights file or set allow_leverage=true in paper trading config if intentional."
        )
    if total < max_gross_without_leverage - tolerance:
        raise ValueError(
            f"Target weights sum to {total:.6f}, expected ~1.0 (tolerance {tolerance})."
        )
    return total


def load_prices(path: Path | None) -> dict[str, float]:
    """Load optional symbol -> price from JSON. Used for share conversion when no position."""
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k).strip(): float(v) for k, v in data.items() if v is not None}


def positions_to_current_weights(
    positions: list[PositionRow],
    nav: float,
) -> dict[str, float]:
    """Compute current portfolio weights from positions and NAV. Uses market value when available, else cost."""
    if nav <= 0:
        logger.warning("NAV <= 0; returning empty current weights")
        return {}

    value_by_symbol: dict[str, float] = {}
    for p in positions:
        if p.market_value is not None and p.market_value >= 0:
            val = p.market_value
            logger.debug("Using market value for %s: %.2f", p.symbol, val)
        else:
            val = p.position * p.avg_cost
            logger.debug("Using cost basis for %s: %.2f (no market value)", p.symbol, val)
        value_by_symbol[p.symbol] = value_by_symbol.get(p.symbol, 0.0) + val

    total_invested = sum(value_by_symbol.values())
    cash_value = nav - total_invested
    if cash_value < 0:
        logger.warning("NAV < sum(position value); treating cash as 0")
        cash_value = 0.0

    weights: dict[str, float] = {}
    for sym, val in value_by_symbol.items():
        weights[sym] = val / nav
    weights["cash"] = cash_value / nav
    return weights


def apply_tau_filter(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    tau: float,
) -> dict[str, float]:
    """Return weight delta to trade: only where |target - current| > tau."""
    all_symbols = set(target_weights) | set(current_weights)
    delta: dict[str, float] = {}
    for s in all_symbols:
        t = target_weights.get(s, 0.0)
        c = current_weights.get(s, 0.0)
        d = t - c
        if abs(d) > tau:
            delta[s] = d
        else:
            delta[s] = 0.0
    return delta


def compute_turnover_one_way(weight_delta: dict[str, float]) -> float:
    """One-way turnover as fraction of NAV (sum of positive deltas, or half of sum of abs)."""
    return sum(abs(d) for d in weight_delta.values()) / 2.0


def _resolve_price_per_symbol(
    current_positions: list[PositionRow],
    supplied_prices: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Price hierarchy: (1) broker market price from position, (2) supplied prices. Returns (price_map, missing)."""
    price_map: dict[str, float] = {}
    # 1) Broker: market price if available, else avg cost for held positions
    for p in current_positions:
        if p.symbol in price_map:
            continue
        if p.market_price is not None and p.market_price > 0:
            price_map[p.symbol] = p.market_price
            logger.debug("Price for %s: broker market %.2f", p.symbol, p.market_price)
        elif p.avg_cost and p.avg_cost > 0:
            price_map[p.symbol] = p.avg_cost
            logger.debug("Price for %s: broker avgCost %.2f (no market price)", p.symbol, p.avg_cost)
    # 2) Supplied prices file for symbols not from broker
    for sym, px in (supplied_prices or {}).items():
        if sym and px is not None and px > 0 and sym not in price_map:
            price_map[sym] = float(px)
            logger.debug("Price for %s: supplied file %.2f", sym, px)
    # Missing = symbols we need but have no price (filled by caller for symbols that need orders)
    return price_map, []


def create_order_preview(
    target_weights: dict[str, float],
    current_positions: list[PositionRow],
    nav: float,
    tau: float = DEFAULT_TAU,
    prices: dict[str, float] | None = None,
    fail_on_missing_price: bool = True,
) -> RebalancePreview:
    """Build dry-run order preview: current vs target, tau filter, proposed share orders.

    Price hierarchy for order sizing: (1) broker market price from position, (2) supplied prices dict.
    If any symbol with non-zero tau-filtered delta has no price, either adds to symbols_missing_price
    (when fail_on_missing_price=False) or raises ValueError (when fail_on_missing_price=True, default).
    """
    supplied_prices = prices or {}
    current_weights = positions_to_current_weights(current_positions, nav)

    # Normalize target to same universe (include cash if present in target)
    target = dict(target_weights)
    if "cash" not in target and current_weights:
        # Infer cash target so weights sum to 1
        other = sum(target.get(s, 0.0) for s in target)
        target["cash"] = max(0.0, 1.0 - other)

    weight_drift = {s: target.get(s, 0.0) - current_weights.get(s, 0.0) for s in set(target) | set(current_weights)}
    tau_filtered_delta = apply_tau_filter(target, current_weights, tau)
    turnover_one_way = compute_turnover_one_way(tau_filtered_delta)

    price_map, _ = _resolve_price_per_symbol(current_positions, supplied_prices)

    proposed: list[OrderPreviewRow] = []
    symbols_missing_price: list[str] = []

    # Tradable symbols = all in tau_filtered_delta except cash (cash is never an order)
    tradable_with_delta = [
        s for s, d in tau_filtered_delta.items()
        if s != "cash" and abs(d) >= 1e-9
    ]
    for symbol in tradable_with_delta:
        delta = tau_filtered_delta[symbol]
        price = price_map.get(symbol)
        if price is None or price <= 0:
            symbols_missing_price.append(symbol)
            continue
        dollar = delta * nav
        shares = int(round(dollar / price))
        if shares == 0:
            continue
        side = "BUY" if shares > 0 else "SELL"
        proposed.append(
            OrderPreviewRow(
                symbol=symbol,
                side=side,
                shares=abs(shares),
                approximate_dollar=abs(dollar),
                weight_delta=delta,
                price_used=price,
            )
        )

    if fail_on_missing_price and symbols_missing_price:
        raise ValueError(
            "Order sizing requires a price for every tradable symbol with non-zero drift. "
            "Missing prices for: "
            + ", ".join(sorted(symbols_missing_price))
            + ". Provide broker market data (connect to IBKR so portfolio returns market prices) or a prices file (--prices)."
        )

    return RebalancePreview(
        nav=nav,
        current_weights=current_weights,
        target_weights=target,
        tau=tau,
        weight_drift=weight_drift,
        tau_filtered_delta=tau_filtered_delta,
        proposed_orders=proposed,
        turnover_one_way=turnover_one_way,
        symbols_missing_price=symbols_missing_price,
    )
