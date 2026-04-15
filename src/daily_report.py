"""Daily portfolio report: market moves, P&L, regime status, position check.

Generates a concise SMS + detailed JSON report after each pipeline run.
Optionally fetches live IBKR positions for reconciliation.

Requires env vars for SMS: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
                           TWILIO_FROM_NUMBER, NOTIFY_TO_NUMBER.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LOGS_DIR, OUTPUTS_DIR, TICKERS

logger = logging.getLogger(__name__)


def _fetch_todays_moves() -> dict[str, float]:
    """Fetch today's return for each portfolio ticker using cached price data."""
    try:
        from src.data.market_ingestion import fetch_prices

        prices = fetch_prices(start="2026-01-01")
        if prices.empty or len(prices) < 2:
            return {}
        latest = prices.iloc[-1]
        prev = prices.iloc[-2]
        moves: dict[str, float] = {}
        for t in TICKERS:
            if t in latest.index and t in prev.index:
                p0 = float(prev[t])
                p1 = float(latest[t])
                if p0 > 0:
                    moves[t] = (p1 / p0) - 1
        return moves
    except Exception as e:
        logger.warning("Could not fetch today's moves: %s", e)
        return {}


def _compute_portfolio_pnl(
    moves: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute portfolio daily P&L from ticker moves and weights."""
    pnl = 0.0
    for ticker, move in moves.items():
        w = weights.get(ticker, 0.0)
        pnl += w * move
    return pnl


def _load_current_weights() -> dict[str, float]:
    """Load latest target weights from rebalance output."""
    csv_path = OUTPUTS_DIR / "rebalance" / "target_weights.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return dict(zip(df["symbol"], df["weight"], strict=True))
    json_path = OUTPUTS_DIR / "rebalance" / "target_weights.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def _load_regime_status() -> dict[str, Any]:
    """Load latest regime label and risk_on from DB."""
    try:
        from src.utils.database import Database

        db = Database()
        df = db.load_regime_labels()
        bt = db.get_latest_backtest_results()
        db.close()
        if df.empty:
            return {"regime": "Unknown", "risk_on": None}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        result: dict[str, Any] = {
            "regime": str(latest["regime"]),
            "risk_on": float(latest["risk_on"])
            if pd.notna(latest["risk_on"])
            else None,
            "prev_regime": str(prev["regime"]),
            "regime_changed": str(latest["regime"]) != str(prev["regime"]),
        }
        if bt:
            result["backtest"] = bt["portfolio"]
        return result
    except Exception as e:
        logger.warning("Could not load regime status: %s", e)
        return {"regime": "Unknown", "risk_on": None}


def _fetch_ibkr_positions() -> dict[str, Any] | None:
    """Try to fetch live IBKR positions. Returns None if unavailable."""
    try:
        from src.execution.ibkr_adapter import IBKRPaperAdapter

        adapter = IBKRPaperAdapter()
        adapter.connect(timeout=5)
        summary = adapter.get_account_summary()
        portfolio = adapter.get_portfolio()
        adapter.disconnect()

        nav = 0.0
        for row in summary:
            if (row.get("tag") or "").strip() == "NetLiquidation":
                nav = float(row.get("value", 0))
                break

        positions = {}
        for p in portfolio:
            sym = (p.get("symbol") or "").strip()
            if sym:
                positions[sym] = {
                    "qty": float(p.get("position", 0)),
                    "market_value": float(p.get("marketValue", 0)),
                    "market_price": float(p.get("marketPrice", 0)),
                }
        return {"nav": nav, "positions": positions}
    except Exception:
        return None


def generate_daily_report() -> dict[str, Any]:
    """Generate the full daily report as a structured dict."""
    now = datetime.now()
    report: dict[str, Any] = {
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.isoformat(),
    }

    # Market moves
    moves = _fetch_todays_moves()
    report["market_moves"] = {k: round(v * 100, 2) for k, v in moves.items()}
    report["spy_return"] = round(moves.get("SPY", 0) * 100, 2)

    # Current weights and portfolio P&L
    weights = _load_current_weights()
    report["target_weights"] = {k: round(v, 4) for k, v in weights.items() if v > 0.001}

    if moves and weights:
        pnl = _compute_portfolio_pnl(moves, weights)
        report["portfolio_return_pct"] = round(pnl * 100, 2)
        spy_ret = moves.get("SPY", 0)
        report["vs_spy_pct"] = round((pnl - spy_ret) * 100, 2)
    else:
        report["portfolio_return_pct"] = None
        report["vs_spy_pct"] = None

    # Regime status
    regime = _load_regime_status()
    report["regime"] = regime.get("regime", "Unknown")
    report["risk_on"] = regime.get("risk_on")
    report["regime_changed"] = regime.get("regime_changed", False)
    report["prev_regime"] = regime.get("prev_regime")
    if "backtest" in regime:
        report["backtest_sharpe"] = round(regime["backtest"]["Sharpe"], 3)

    # IBKR positions (if available)
    ibkr = _fetch_ibkr_positions()
    if ibkr:
        report["ibkr_connected"] = True
        report["ibkr_nav"] = ibkr["nav"]
        report["ibkr_positions"] = ibkr["positions"]
    else:
        report["ibkr_connected"] = False

    return report


def format_sms(report: dict[str, Any]) -> str:
    """Format report as a concise SMS (aim for ≤320 chars, 2 segments max).

    Example:
        04/01 Contraction ro:0.51
        Port:-0.12% SPY:-0.45% +0.33%
        IEF 36% SPY 20% TLT 17%
        Sharpe:0.97
    """
    date = report.get("date", "??")
    regime = report.get("regime", "?")
    ro = report.get("risk_on")
    ro_str = f"{ro:.2f}" if ro is not None else "N/A"

    port_ret = report.get("portfolio_return_pct")
    spy_ret = report.get("spy_return", 0)
    vs_spy = report.get("vs_spy_pct")

    if port_ret is not None:
        pnl_line = f"Port:{port_ret:+.2f}% SPY:{spy_ret:+.2f}%"
        if vs_spy is not None:
            pnl_line += f" {vs_spy:+.2f}%"
    else:
        pnl_line = "No market data"

    # Top 3 weights
    weights = report.get("target_weights", {})
    top = sorted(
        ((k, v) for k, v in weights.items() if k != "cash"),
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    weights_line = " ".join(f"{k} {v:.0%}" for k, v in top)

    sharpe = report.get("backtest_sharpe")
    sharpe_str = f"Sharpe:{sharpe:.2f}" if sharpe else ""

    regime_alert = ""
    if report.get("regime_changed"):
        regime_alert = f"\n!! REGIME CHANGE: {report.get('prev_regime')} -> {regime}"

    ibkr_str = ""
    if report.get("ibkr_connected"):
        ibkr_str = f"\nIBKR NAV:${report['ibkr_nav']:,.0f}"

    body = f"{date} {regime} ro:{ro_str}\n{pnl_line}\n{weights_line}"
    if sharpe_str:
        body += f"\n{sharpe_str}"
    if regime_alert:
        body += regime_alert
    if ibkr_str:
        body += ibkr_str

    return body[:320]


def save_report(report: dict[str, Any]) -> Path:
    """Save detailed report to logs/ as JSON."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = report.get("date", datetime.now().strftime("%Y%m%d"))
    path = LOGS_DIR / f"daily_report_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Daily report saved: %s", path)
    return path


def send_daily_report() -> None:
    """Generate report, save JSON, and send SMS.

    Non-fatal: logs warnings on failure.
    """
    try:
        report = generate_daily_report()
        save_report(report)

        sms_body = format_sms(report)
        logger.info("Daily SMS (%d chars):\n%s", len(sms_body), sms_body)

        try:
            from src.notify import send_sms

            send_sms(sms_body)
        except OSError:
            logger.warning("Twilio not configured — skipping SMS.")
        except Exception as e:
            logger.warning("SMS send failed: %s", e)

    except Exception as e:
        logger.warning("Daily report generation failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    send_daily_report()
