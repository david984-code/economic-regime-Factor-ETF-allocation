"""Tests for src/daily_report.py — daily portfolio report."""

from __future__ import annotations

import pytest

from src.daily_report import _compute_portfolio_pnl, format_sms


class TestComputePortfolioPnl:
    def test_simple_pnl(self) -> None:
        moves = {"SPY": 0.01, "IEF": -0.005}
        weights = {"SPY": 0.6, "IEF": 0.4}
        pnl = _compute_portfolio_pnl(moves, weights)
        assert pnl == pytest.approx(0.006 - 0.002)

    def test_zero_weights(self) -> None:
        moves = {"SPY": 0.05}
        weights = {"SPY": 0.0}
        assert _compute_portfolio_pnl(moves, weights) == 0.0

    def test_missing_ticker(self) -> None:
        moves = {"SPY": 0.01, "GLD": 0.02}
        weights = {"SPY": 0.5}
        pnl = _compute_portfolio_pnl(moves, weights)
        assert pnl == pytest.approx(0.005)


class TestFormatSms:
    def test_basic_format(self) -> None:
        report = {
            "date": "2026-04-01",
            "regime": "Contraction",
            "risk_on": 0.51,
            "portfolio_return_pct": -0.12,
            "spy_return": -0.45,
            "vs_spy_pct": 0.33,
            "target_weights": {"IEF": 0.35, "SPY": 0.20, "TLT": 0.17, "cash": 0.05},
            "backtest_sharpe": 0.97,
            "regime_changed": False,
            "ibkr_connected": False,
        }
        sms = format_sms(report)
        assert "Contraction" in sms
        assert "ro:0.51" in sms
        assert "Port:" in sms
        assert len(sms) <= 320

    def test_regime_change_alert(self) -> None:
        report = {
            "date": "2026-04-01",
            "regime": "Recovery",
            "risk_on": 0.72,
            "portfolio_return_pct": 1.5,
            "spy_return": 1.2,
            "vs_spy_pct": 0.3,
            "target_weights": {"SPY": 0.60},
            "regime_changed": True,
            "prev_regime": "Contraction",
            "ibkr_connected": False,
        }
        sms = format_sms(report)
        assert "REGIME CHANGE" in sms

    def test_no_market_data(self) -> None:
        report = {
            "date": "2026-04-01",
            "regime": "Unknown",
            "risk_on": None,
            "portfolio_return_pct": None,
            "spy_return": 0,
            "vs_spy_pct": None,
            "target_weights": {},
            "ibkr_connected": False,
        }
        sms = format_sms(report)
        assert "No market data" in sms

    def test_ibkr_connected(self) -> None:
        report = {
            "date": "2026-04-01",
            "regime": "Recovery",
            "risk_on": 0.70,
            "portfolio_return_pct": 0.5,
            "spy_return": 0.8,
            "vs_spy_pct": -0.3,
            "target_weights": {"SPY": 0.60},
            "ibkr_connected": True,
            "ibkr_nav": 105000,
            "regime_changed": False,
        }
        sms = format_sms(report)
        assert "IBKR NAV:$105,000" in sms
