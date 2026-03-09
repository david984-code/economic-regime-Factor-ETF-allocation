"""Database module for storing and retrieving regime and allocation data.

Uses SQLite for lightweight, serverless relational storage with ACID guarantees.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class Database:
    """SQLite database handler for economic regime and allocation data."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to outputs/allocations.db
        """
        if db_path is None:
            OUTPUTS_DIR.mkdir(exist_ok=True)
            db_path = OUTPUTS_DIR / "allocations.db"

        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._create_tables()
        logger.debug("Database connected: %s", db_path)

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_labels (
                date TEXT PRIMARY KEY,
                regime TEXT NOT NULL,
                risk_on REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimal_allocations (
                regime TEXT NOT NULL,
                asset TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (regime, asset)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_weights (
                date TEXT NOT NULL,
                asset TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, asset)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_forecast (
                forecast_date TEXT NOT NULL,
                target_month TEXT NOT NULL,
                risk_on_forecast REAL NOT NULL,
                regime_forecast TEXT,
                accuracy_1m REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (forecast_date, target_month)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                portfolio_cagr REAL,
                portfolio_volatility REAL,
                portfolio_sharpe REAL,
                portfolio_max_drawdown REAL,
                benchmark_cagr REAL,
                benchmark_volatility REAL,
                benchmark_sharpe REAL,
                benchmark_max_drawdown REAL
            )
        """)

        self.conn.commit()

    def save_regime_labels(self, df: pd.DataFrame) -> None:
        """Save regime labels to database."""
        df_copy = df.reset_index()
        df_copy["date"] = df_copy["date"].astype(str)
        df_copy.to_sql(
            "regime_labels",
            self.conn,
            if_exists="replace",
            index=False,
            dtype={"date": "TEXT", "regime": "TEXT", "risk_on": "REAL"},
        )
        self.conn.commit()
        logger.info("Saved regime labels to database")

    def load_regime_labels(self) -> pd.DataFrame:
        """Load regime labels from database."""
        df = pd.read_sql(
            "SELECT date, regime, risk_on FROM regime_labels ORDER BY date",
            self.conn,
            parse_dates=["date"],
        )
        df.set_index("date", inplace=True)
        return df

    def save_optimal_allocations(self, allocations: dict[str, dict[str, float]]) -> None:
        """Save optimal allocations to database."""
        rows = []
        for regime, weights in allocations.items():
            for asset, weight in weights.items():
                rows.append({"regime": regime, "asset": asset, "weight": float(weight)})
        df = pd.DataFrame(rows)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM optimal_allocations")
        self.conn.commit()
        df.to_sql("optimal_allocations", self.conn, if_exists="append", index=False)
        self.conn.commit()
        logger.info("Saved optimal allocations to database")

    def load_optimal_allocations(self) -> dict[str, dict[str, float]]:
        """Load optimal allocations from database."""
        df = pd.read_sql("SELECT regime, asset, weight FROM optimal_allocations", self.conn)
        allocations: dict[str, dict[str, float]] = {}
        for regime in df["regime"].unique():
            regime_df = df[df["regime"] == regime]
            allocations[regime] = dict(zip(regime_df["asset"], regime_df["weight"]))
        return allocations

    def save_current_weights(self, date: str, weights: pd.Series) -> None:
        """Save current portfolio weights."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM current_weights WHERE date = ?", (date,))
        self.conn.commit()
        df = pd.DataFrame({"date": date, "asset": weights.index, "weight": weights.values})
        df.to_sql("current_weights", self.conn, if_exists="append", index=False)
        self.conn.commit()

    def save_regime_forecast(
        self,
        forecast_date: str,
        target_month: str,
        risk_on_forecast: float,
        regime_forecast: str | None = None,
        accuracy_1m: float | None = None,
    ) -> None:
        """Save regime forecast for a target month."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO regime_forecast
            (forecast_date, target_month, risk_on_forecast, regime_forecast, accuracy_1m)
            VALUES (?, ?, ?, ?, ?)
            """,
            (forecast_date, target_month, risk_on_forecast, regime_forecast, accuracy_1m),
        )
        self.conn.commit()
        logger.info("Saved regime forecast for %s", target_month)

    def load_latest_regime_forecast(self, target_month: str) -> dict[str, float | str | None] | None:
        """Load most recent forecast for a target month."""
        df = pd.read_sql(
            """
            SELECT forecast_date, risk_on_forecast, regime_forecast
            FROM regime_forecast WHERE target_month = ?
            ORDER BY forecast_date DESC LIMIT 1
            """,
            self.conn,
            params=(target_month,),
        )
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "forecast_date": str(row["forecast_date"]),
            "risk_on_forecast": float(row["risk_on_forecast"]),
            "regime_forecast": str(row["regime_forecast"]) if pd.notna(row["regime_forecast"]) else None,
        }

    def save_backtest_results(self, metrics: dict[str, float], bench_metrics: dict[str, float]) -> None:
        """Save backtest performance metrics."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO backtest_results (
                portfolio_cagr, portfolio_volatility, portfolio_sharpe, portfolio_max_drawdown,
                benchmark_cagr, benchmark_volatility, benchmark_sharpe, benchmark_max_drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics["CAGR"],
                metrics["Volatility"],
                metrics["Sharpe"],
                metrics["Max Drawdown"],
                bench_metrics["CAGR"],
                bench_metrics["Volatility"],
                bench_metrics["Sharpe"],
                bench_metrics["Max Drawdown"],
            ),
        )
        self.conn.commit()

    def get_latest_backtest_results(self) -> dict[str, Any] | None:
        """Get most recent backtest results."""
        df = pd.read_sql(
            "SELECT * FROM backtest_results ORDER BY run_date DESC LIMIT 1",
            self.conn,
        )
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "run_date": row["run_date"],
            "portfolio": {
                "CAGR": row["portfolio_cagr"],
                "Volatility": row["portfolio_volatility"],
                "Sharpe": row["portfolio_sharpe"],
                "Max Drawdown": row["portfolio_max_drawdown"],
            },
            "benchmark": {
                "CAGR": row["benchmark_cagr"],
                "Volatility": row["benchmark_volatility"],
                "Sharpe": row["benchmark_sharpe"],
                "Max Drawdown": row["benchmark_max_drawdown"],
            },
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
