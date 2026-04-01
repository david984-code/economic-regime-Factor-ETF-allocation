"""On-demand system status check.

Run anytime to see: is the pipeline healthy, what's the current regime,
how did the portfolio perform today/this week/YTD, is IBKR connected.

Usage:
    uv run python -m src.status           # full status
    uv run python -m src.status --quick   # just today's P&L
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import LOGS_DIR, OUTPUTS_DIR, TICKERS

logger = logging.getLogger(__name__)


def _load_weights() -> dict[str, float]:
    """Load current target weights."""
    csv_path = OUTPUTS_DIR / "rebalance" / "target_weights.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return dict(zip(df["symbol"], df["weight"], strict=True))
    return {}


def _regime_status() -> dict[str, Any]:
    """Get regime info from DB."""
    try:
        from src.utils.database import Database

        db = Database()
        df = db.load_regime_labels()
        bt = db.get_latest_backtest_results()
        forecast = None
        try:
            next_month = (pd.Timestamp.today().to_period("M") + 1).strftime("%Y-%m")
            forecast = db.load_latest_regime_forecast(next_month)
        except Exception:
            pass
        db.close()

        if df.empty:
            return {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        out: dict[str, Any] = {
            "regime": str(latest["regime"]),
            "risk_on": float(latest["risk_on"]) if pd.notna(latest["risk_on"]) else None,
            "date": str(latest.name),
            "prev_regime": str(prev["regime"]),
            "changed": str(latest["regime"]) != str(prev["regime"]),
        }
        if bt:
            out["backtest"] = bt["portfolio"]
        if forecast:
            out["forecast"] = forecast
        return out
    except Exception as e:
        return {"error": str(e)}


def _pipeline_health() -> dict[str, Any]:
    """Check when the pipeline last ran and if it succeeded."""
    health: dict[str, Any] = {"status": "unknown"}

    # Check latest log file
    if LOGS_DIR.exists():
        logs = sorted(LOGS_DIR.glob("daily_update_*.log"), reverse=True)
        if logs:
            latest_log = logs[0]
            health["last_log"] = str(latest_log.name)
            health["last_log_time"] = datetime.fromtimestamp(
                latest_log.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")

            # Check if it completed successfully
            content = latest_log.read_text(encoding="utf-8", errors="replace")
            if "SUMMARY: All steps completed" in content:
                health["status"] = "OK"
            elif "[FAIL]" in content:
                health["status"] = "FAILED"
                # Find the failure line
                for line in content.split("\n"):
                    if "[FAIL]" in line:
                        health["failure"] = line.strip()[-100:]
                        break
            else:
                health["status"] = "incomplete"

            # Check age
            age_hours = (
                datetime.now() - datetime.fromtimestamp(latest_log.stat().st_mtime)
            ).total_seconds() / 3600
            health["age_hours"] = round(age_hours, 1)
            if age_hours > 28:
                health["stale"] = True

    # Check latest daily report
    if LOGS_DIR.exists():
        reports = sorted(LOGS_DIR.glob("daily_report_*.json"), reverse=True)
        if reports:
            health["last_report"] = str(reports[0].name)
            with open(reports[0]) as f:
                health["last_report_data"] = json.load(f)

    return health


def _market_performance(weights: dict[str, float]) -> dict[str, Any]:
    """Compute portfolio performance for today, this week, MTD, YTD."""
    try:
        from src.data.market_ingestion import fetch_prices

        prices = fetch_prices(start="2025-12-01")
        if prices.empty:
            return {"error": "No price data"}

        spy_rets = prices["SPY"].pct_change().dropna()

        # Portfolio returns using weights
        port_rets = pd.Series(0.0, index=spy_rets.index)
        for ticker, w in weights.items():
            if ticker in prices.columns and ticker != "cash" and w > 0.001:
                t_rets = prices[ticker].pct_change().dropna()
                common = port_rets.index.intersection(t_rets.index)
                port_rets.loc[common] += w * t_rets.loc[common]

        today = prices.index[-1]
        periods: dict[str, Any] = {}

        # Today
        if len(port_rets) >= 1:
            periods["today"] = {
                "date": today.strftime("%Y-%m-%d"),
                "portfolio": round(float(port_rets.iloc[-1]) * 100, 2),
                "spy": round(float(spy_rets.iloc[-1]) * 100, 2),
                "vs_spy": round(float(port_rets.iloc[-1] - spy_rets.iloc[-1]) * 100, 2),
            }

        # This week (last 5 trading days)
        week = port_rets.iloc[-5:]
        spy_week = spy_rets.reindex(week.index)
        if len(week) >= 2:
            periods["week"] = {
                "portfolio": round(float((1 + week).prod() - 1) * 100, 2),
                "spy": round(float((1 + spy_week).prod() - 1) * 100, 2),
            }

        # MTD
        month_start = today.replace(day=1)
        mtd_mask = port_rets.index >= month_start
        if mtd_mask.sum() >= 1:
            mtd_p = port_rets.loc[mtd_mask]
            mtd_s = spy_rets.reindex(mtd_p.index)
            periods["mtd"] = {
                "portfolio": round(float((1 + mtd_p).prod() - 1) * 100, 2),
                "spy": round(float((1 + mtd_s).prod() - 1) * 100, 2),
            }

        # YTD
        ytd_mask = port_rets.index >= "2026-01-01"
        if ytd_mask.sum() >= 1:
            ytd_p = port_rets.loc[ytd_mask]
            ytd_s = spy_rets.reindex(ytd_p.index)
            periods["ytd"] = {
                "portfolio": round(float((1 + ytd_p).prod() - 1) * 100, 2),
                "spy": round(float((1 + ytd_s).prod() - 1) * 100, 2),
            }

        # Top movers today
        movers: list[dict[str, Any]] = []
        for t in TICKERS:
            if t in prices.columns:
                ret = float(prices[t].pct_change().iloc[-1])
                w = weights.get(t, 0.0)
                if w > 0.001:
                    movers.append({
                        "ticker": t,
                        "return": round(ret * 100, 2),
                        "weight": round(w * 100, 1),
                        "contribution": round(ret * w * 100, 3),
                    })
        movers.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        periods["movers"] = movers

        return periods
    except Exception as e:
        return {"error": str(e)}


def _ibkr_status() -> dict[str, Any]:
    """Check IBKR connectivity and positions."""
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

        positions: dict[str, dict[str, float]] = {}
        for p in portfolio:
            sym = (p.get("symbol") or "").strip()
            if sym:
                positions[sym] = {
                    "qty": float(p.get("position", 0)),
                    "value": float(p.get("marketValue", 0)),
                }

        return {"connected": True, "nav": nav, "positions": positions}
    except Exception:
        return {"connected": False}


def print_status(quick: bool = False) -> None:
    """Print the full system status to console."""
    weights = _load_weights()

    # Header
    print()
    print("=" * 60)
    print(f"  SYSTEM STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Pipeline health
    health = _pipeline_health()
    status_icon = {"OK": "[OK]", "FAILED": "[!!]", "incomplete": "[??]"}.get(
        health.get("status", ""), "[??]"
    )
    print(f"\n  Pipeline: {status_icon} {health.get('status', 'unknown')}")
    if health.get("last_log"):
        print(f"  Last run: {health.get('last_log_time', '?')} ({health.get('age_hours', '?')}h ago)")
    if health.get("stale"):
        print("  !! WARNING: Pipeline has not run in >28 hours")
    if health.get("failure"):
        print(f"  Failure: {health['failure']}")

    # Regime
    regime = _regime_status()
    if regime:
        ro = regime.get("risk_on")
        ro_str = f"{ro:.3f}" if ro is not None else "N/A"
        print(f"\n  Regime: {regime.get('regime', '?')} | risk_on: {ro_str}")
        if regime.get("changed"):
            print(f"  !! REGIME CHANGED from {regime.get('prev_regime')}")
        if regime.get("forecast"):
            fc = regime["forecast"]
            print(f"  Forecast: risk_on={fc.get('risk_on_forecast', '?'):.3f} (next month)")

    # Current weights
    if weights:
        print("\n  Portfolio weights:")
        for ticker, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if w > 0.005:
                print(f"    {ticker:6s} {w:.1%}")

    if quick:
        # Just today's P&L
        perf = _market_performance(weights)
        if "today" in perf:
            t = perf["today"]
            print(f"\n  Today ({t['date']}):")
            print(f"    Portfolio: {t['portfolio']:+.2f}%")
            print(f"    SPY:       {t['spy']:+.2f}%")
            print(f"    vs SPY:    {t['vs_spy']:+.2f}%")
        print()
        return

    # Full performance
    perf = _market_performance(weights)
    if "today" in perf:
        t = perf["today"]
        print(f"\n  Today ({t['date']}):")
        print(f"    Portfolio: {t['portfolio']:+.2f}%  SPY: {t['spy']:+.2f}%  vs SPY: {t['vs_spy']:+.2f}%")
    if "week" in perf:
        w = perf["week"]
        print(f"  This week:  Port: {w['portfolio']:+.2f}%  SPY: {w['spy']:+.2f}%")
    if "mtd" in perf:
        m = perf["mtd"]
        print(f"  MTD:        Port: {m['portfolio']:+.2f}%  SPY: {m['spy']:+.2f}%")
    if "ytd" in perf:
        y = perf["ytd"]
        print(f"  YTD:        Port: {y['portfolio']:+.2f}%  SPY: {y['spy']:+.2f}%")

    # Top movers
    if "movers" in perf and perf["movers"]:
        print("\n  Today's movers (by contribution):")
        for m in perf["movers"][:5]:
            print(
                f"    {m['ticker']:6s} {m['return']:+.2f}%  "
                f"(wt: {m['weight']:.1f}%, contrib: {m['contribution']:+.3f}%)"
            )

    # Backtest metrics
    if regime.get("backtest"):
        bt = regime["backtest"]
        print(f"\n  Backtest: Sharpe={bt.get('Sharpe', 0):.3f}  "
              f"CAGR={bt.get('CAGR', 0):.2%}  MaxDD={bt.get('Max Drawdown', 0):.2%}")

    # IBKR
    ibkr = _ibkr_status()
    if ibkr.get("connected"):
        print(f"\n  IBKR: Connected | NAV: ${ibkr['nav']:,.2f}")
        if ibkr.get("positions"):
            print("  Positions:")
            for sym, pos in sorted(ibkr["positions"].items()):
                print(f"    {sym:6s} qty={pos['qty']:.0f}  val=${pos['value']:,.0f}")
    else:
        print("\n  IBKR: Not connected (IB Gateway not running)")

    print()
    print("=" * 60)
    print()


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="System status check.")
    p.add_argument("--quick", action="store_true", help="Just today's P&L.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_cli()
    print_status(quick=args.quick)
