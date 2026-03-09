"""Central configuration for the quantitative research pipeline.

Single source of truth for tickers, dates, paths, and regime constraints.
"""

from datetime import datetime
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
MACRO_CACHE_DIR = OUTPUTS_DIR / "macro_cache"

# --- Date range ---
START_DATE = "2010-01-01"


def get_end_date() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.today().strftime("%Y-%m-%d")


# --- Asset universe ---
TICKERS = [
    "SPY",
    "GLD",
    "MTUM",
    "VLUE",
    "USMV",
    "QUAL",
    "IJR",
    "VIG",
    "IEF",
    "TLT",
]
ASSETS = TICKERS + ["cash"]

# --- Backtest / vol scaling ---
VOL_LOOKBACK = 63  # ~3 months of trading days
VOL_EPS = 1e-8
MIN_VOL = 0.05
MAX_VOL = 2.00
COST_BPS = 0.0008  # 8 bps per dollar traded (one-way)

# --- Regime-specific cash constraints (min, max) ---
REGIME_CASH: dict[str, tuple[float, float]] = {
    "Recovery": (0.05, 0.10),
    "Overheating": (0.05, 0.12),
    "Stagflation": (0.10, 0.20),
    "Contraction": (0.15, 0.30),
    "Unknown": (0.10, 0.20),
}
DEFAULT_MIN_CASH = 0.05
DEFAULT_MAX_CASH = 0.15

# --- Regime-specific min asset allocations ---
REGIME_MIN_ASSETS: dict[str, dict[str, float]] = {
    "Stagflation": {"GLD": 0.08},
    "Contraction": {"IEF": 0.05, "TLT": 0.05},
}

# --- FRED series (optional high-frequency) ---
FRED_SERIES_OPTIONAL: dict[str, str] = {
    "pmi": "NAPM",
    "indpro": "INDPRO",
    "claims": "ICSA",
    "hy_spread": "BAMLH0A0HYM2",
}

# --- Regime aliases for blending ---
REGIME_ALIASES: dict[str, str] = {
    "Expansion": "Overheating",
    "Slowdown": "Contraction",
}
RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
