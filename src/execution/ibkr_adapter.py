"""IBKR paper trading adapter via IB Gateway.

Connects to IB Gateway (paper), fetches account summary, cash, NAV, and positions.
No order placement. Config-driven; credentials via environment variables.
"""

import logging
import os
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


# Optional: load yaml only when needed to avoid hard dependency if not used
def _load_paper_config() -> dict[str, Any]:
    """Load paper_trading.yaml with env overrides. No credentials in file."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for execution config. Install with: pip install pyyaml"
        ) from None
    config_path = PROJECT_ROOT / "config" / "paper_trading.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Paper trading config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Env overrides (no hardcoded credentials)
    out = {
        "host": os.environ.get("IBKR_HOST", raw.get("host", "127.0.0.1")),
        "port": int(os.environ.get("IBKR_PORT", raw.get("port", 4002))),
        "client_id": int(os.environ.get("IBKR_CLIENT_ID", raw.get("client_id", 1))),
        "account": os.environ.get("IBKR_ACCOUNT", raw.get("account") or "").strip(),
        "trading_enabled": raw.get("trading_enabled", False),
        "dry_run": raw.get("dry_run", True),
    }
    if not out["account"]:
        logger.warning("IBKR_ACCOUNT not set; account summary may list all accounts")
    return out


class IBKRPaperAdapter:
    """Adapter for IBKR paper trading via IB Gateway. Read-only connectivity and account data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or _load_paper_config()
        self._ib: Any = None
        self._connected = False

    def connect(self, timeout: float = 10.0) -> None:
        """Connect to IB Gateway (paper). Fails safely if unavailable."""
        if self._connected and self._ib is not None and self._ib.isConnected():
            logger.info("Already connected to IB Gateway")
            return
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError(
                "ib_insync is required for IBKR adapter. Install with: pip install ib_insync"
            ) from None
        host = self._config["host"]
        port = self._config["port"]
        client_id = self._config["client_id"]
        account = self._config["account"]
        logger.info(
            "Connecting to IB Gateway paper host=%s port=%s client_id=%s", host, port, client_id
        )
        self._ib = IB()
        try:
            self._ib.connect(
                host=host,
                port=port,
                clientId=client_id,
                timeout=timeout,
                readonly=True,
                account=account or "",
            )
            self._connected = True
            logger.info("Connected to IB Gateway successfully")
        except Exception as e:
            self._ib = None
            self._connected = False
            logger.exception("IB Gateway connection failed: %s", e)
            raise ConnectionError(f"Cannot connect to IB Gateway at {host}:{port}: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IB Gateway and clear session state."""
        if self._ib is None:
            return
        try:
            if self._ib.isConnected():
                self._ib.disconnect()
                logger.info("Disconnected from IB Gateway")
        except Exception as e:
            logger.warning("Disconnect warning: %s", e)
        finally:
            self._ib = None
            self._connected = False

    def get_account_summary(self) -> list[dict[str, str]]:
        """Return account summary (tag, value, currency, account) for the configured account."""
        self._require_connected()
        account = self._config["account"] or ""
        rows = self._ib.accountSummary(account=account)
        out = []
        for r in rows:
            out.append(
                {
                    "tag": getattr(r, "tag", ""),
                    "value": getattr(r, "value", ""),
                    "currency": getattr(r, "currency", ""),
                    "account": getattr(r, "account", ""),
                }
            )
        logger.debug("Account summary rows: %d", len(out))
        return out

    def get_positions(self) -> list[dict[str, Any]]:
        """Return current positions (account, symbol, quantity, avgCost, etc.)."""
        self._require_connected()
        account = self._config["account"] or ""
        positions = self._ib.positions(account=account)
        out = []
        for p in positions:
            contract = getattr(p, "contract", None)
            symbol = ""
            if contract is not None:
                symbol = getattr(contract, "symbol", "") or ""
            out.append(
                {
                    "account": getattr(p, "account", ""),
                    "symbol": symbol,
                    "position": getattr(p, "position", 0),
                    "avgCost": getattr(p, "avgCost", 0.0),
                }
            )
        logger.debug("Positions count: %d", len(out))
        return out

    def get_portfolio(self) -> list[dict[str, Any]]:
        """Return portfolio items with market value and price for valuation-correct weights and order sizing.

        Each item has: account, symbol, position, avgCost, marketValue, marketPrice.
        Prefer this over get_positions() when available; if portfolio list is empty (e.g. before
        account updates), falls back to get_positions() with marketValue/marketPrice set from cost.
        """
        self._require_connected()
        account = self._config["account"] or ""
        items: list[dict[str, Any]] = []
        try:
            portfolio = self._ib.portfolio(account=account)
            for item in portfolio:
                contract = getattr(item, "contract", None)
                symbol = getattr(contract, "symbol", "") if contract else ""
                if not symbol:
                    continue
                position = float(getattr(item, "position", 0))
                avg_cost = float(getattr(item, "averageCost", 0.0))
                market_value = float(getattr(item, "marketValue", 0.0))
                market_price = float(getattr(item, "marketPrice", 0.0))
                if market_price <= 0 and position != 0 and avg_cost != 0:
                    market_price = avg_cost
                    market_value = position * avg_cost
                items.append(
                    {
                        "account": account,
                        "symbol": symbol,
                        "position": position,
                        "avgCost": avg_cost,
                        "marketValue": market_value,
                        "marketPrice": market_price,
                    }
                )
        except Exception as e:
            logger.warning("get_portfolio failed, falling back to positions(): %s", e)
        if not items:
            # Fallback: positions only (no market data)
            for p in self.get_positions():
                symbol = (p.get("symbol") or "").strip()
                if not symbol:
                    continue
                pos = float(p.get("position", 0))
                avg = float(p.get("avgCost", 0.0))
                items.append(
                    {
                        "account": p.get("account", ""),
                        "symbol": symbol,
                        "position": pos,
                        "avgCost": avg,
                        "marketValue": pos * avg if (pos and avg) else 0.0,
                        "marketPrice": avg if avg else 0.0,
                    }
                )
            logger.info(
                "Using positions fallback (no portfolio market data); current weights will use cost basis"
            )
        else:
            logger.debug("Portfolio items: %d with market value/price", len(items))
        return items

    def health_check(self) -> bool:
        """Return True if connected and we can reach the broker (e.g. account summary or positions)."""
        if self._ib is None or not self._ib.isConnected():
            return False
        try:
            self.get_account_summary()
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    def _require_connected(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("Not connected to IB Gateway. Call connect() first.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    def __enter__(self) -> "IBKRPaperAdapter":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
