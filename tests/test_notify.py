"""Tests for src/notify.py — SMS notifier."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.notify import SMS_MAX_LEN, build_sms_body, notify, send_sms


@pytest.fixture()
def mock_db(tmp_path: Path) -> Path:
    """Create a temporary allocations.db with test data."""
    db_path = tmp_path / "allocations.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE regime_labels (
            date TEXT PRIMARY KEY, regime TEXT NOT NULL, risk_on REAL
        )
    """)
    conn.execute("""
        CREATE TABLE optimal_allocations (
            regime TEXT NOT NULL, asset TEXT NOT NULL, weight REAL NOT NULL,
            PRIMARY KEY (regime, asset)
        )
    """)
    conn.execute("""
        CREATE TABLE backtest_results (
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            portfolio_cagr REAL, portfolio_volatility REAL,
            portfolio_sharpe REAL, portfolio_max_drawdown REAL,
            benchmark_cagr REAL, benchmark_volatility REAL,
            benchmark_sharpe REAL, benchmark_max_drawdown REAL
        )
    """)
    conn.execute("INSERT INTO regime_labels VALUES ('2026-03-31', 'Recovery', 0.72)")
    allocations = [
        ("Recovery", "SPY", 0.28),
        ("Recovery", "MTUM", 0.18),
        ("Recovery", "QUAL", 0.15),
        ("Recovery", "VLUE", 0.12),
        ("Recovery", "IJR", 0.10),
        ("Recovery", "VIG", 0.07),
        ("Recovery", "USMV", 0.05),
        ("Recovery", "cash", 0.05),
    ]
    conn.executemany("INSERT INTO optimal_allocations VALUES (?, ?, ?)", allocations)
    conn.execute(
        "INSERT INTO backtest_results "
        "(portfolio_cagr, portfolio_volatility, portfolio_sharpe, portfolio_max_drawdown, "
        "benchmark_cagr, benchmark_volatility, benchmark_sharpe, benchmark_max_drawdown) "
        "VALUES (0.075, 0.12, 0.51, -0.075, 0.10, 0.15, 0.55, -0.20)"
    )
    conn.commit()
    conn.close()
    return db_path


class TestBuildSmsBody:
    def test_body_under_160_chars(self, mock_db: Path) -> None:
        with patch("src.notify.Database") as mock_cls:
            from src.utils.database import Database

            db = Database(db_path=mock_db)
            mock_cls.return_value = db
            body = build_sms_body()
            assert len(body) <= SMS_MAX_LEN

    def test_body_contains_regime(self, mock_db: Path) -> None:
        with patch("src.notify.Database") as mock_cls:
            from src.utils.database import Database

            db = Database(db_path=mock_db)
            mock_cls.return_value = db
            body = build_sms_body()
            assert "Recovery" in body

    def test_body_contains_top_assets(self, mock_db: Path) -> None:
        with patch("src.notify.Database") as mock_cls:
            from src.utils.database import Database

            db = Database(db_path=mock_db)
            mock_cls.return_value = db
            body = build_sms_body()
            assert "SPY" in body
            assert "MTUM" in body

    def test_body_contains_sharpe(self, mock_db: Path) -> None:
        with patch("src.notify.Database") as mock_cls:
            from src.utils.database import Database

            db = Database(db_path=mock_db)
            mock_cls.return_value = db
            body = build_sms_body()
            assert "Sharpe" in body

    def test_empty_db_returns_message(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE regime_labels (
                date TEXT PRIMARY KEY, regime TEXT NOT NULL, risk_on REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE optimal_allocations (
                regime TEXT, asset TEXT, weight REAL, created_at TIMESTAMP,
                PRIMARY KEY (regime, asset)
            )
        """)
        conn.execute("""
            CREATE TABLE current_weights (
                date TEXT, asset TEXT, weight REAL, created_at TIMESTAMP,
                PRIMARY KEY (date, asset)
            )
        """)
        conn.execute("""
            CREATE TABLE regime_forecast (
                forecast_date TEXT, target_month TEXT, risk_on_forecast REAL,
                regime_forecast TEXT, accuracy_1m REAL, created_at TIMESTAMP,
                PRIMARY KEY (forecast_date, target_month)
            )
        """)
        conn.execute("""
            CREATE TABLE backtest_results (
                run_date TIMESTAMP, portfolio_cagr REAL, portfolio_volatility REAL,
                portfolio_sharpe REAL, portfolio_max_drawdown REAL,
                benchmark_cagr REAL, benchmark_volatility REAL,
                benchmark_sharpe REAL, benchmark_max_drawdown REAL
            )
        """)
        conn.commit()
        conn.close()
        with patch("src.notify.Database") as mock_cls:
            from src.utils.database import Database

            db = Database(db_path=db_path)
            mock_cls.return_value = db
            body = build_sms_body()
            assert "no regime data" in body.lower()


class TestSendSms:
    def test_missing_env_vars_raises(self) -> None:
        from src.notify import _get_twilio_config

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="Missing Twilio"):
                _get_twilio_config()

    def test_send_calls_twilio(self) -> None:
        import sys

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = MagicMock(sid="SM123")
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # Create fake twilio module so the import inside send_sms works
        fake_twilio = MagicMock()
        fake_twilio.rest.Client = mock_client_cls
        sys.modules["twilio"] = fake_twilio
        sys.modules["twilio.rest"] = fake_twilio.rest

        config = {
            "TWILIO_ACCOUNT_SID": "AC_TEST",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+1111",
            "NOTIFY_TO_NUMBER": "+2222",
        }
        try:
            with patch("src.notify._get_twilio_config", return_value=config):
                sid = send_sms("hello")
            assert sid == "SM123"
            mock_client_instance.messages.create.assert_called_once()
        finally:
            sys.modules.pop("twilio", None)
            sys.modules.pop("twilio.rest", None)


class TestNotify:
    def test_notify_logs_warning_on_missing_config(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch("src.notify.build_sms_body", return_value="test body"):
            with patch("src.notify.send_sms", side_effect=OSError("no config")):
                notify()
        assert any(
            "not configured" in r.message or "failed" in r.message
            for r in caplog.records
        )
