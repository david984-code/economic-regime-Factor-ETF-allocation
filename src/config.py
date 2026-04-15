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

# Expanded universe with sector ETFs
TICKERS_EXPANDED = TICKERS + [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Healthcare
    "XLI",  # Industrials
    "XLP",  # Consumer Staples
    "XLY",  # Consumer Discretionary
    "XLU",  # Utilities
    "XLB",  # Materials
    "XLRE",  # Real Estate
    "XLC",  # Communication Services
]
ASSETS_EXPANDED = TICKERS_EXPANDED + ["cash"]

# Sleeve definitions for current universe
RISK_ON_ASSETS_BASE = ["SPY", "MTUM", "VLUE", "QUAL", "USMV", "IJR", "VIG"]
EQUITY_TICKERS = RISK_ON_ASSETS_BASE
RISK_OFF_ASSETS_BASE = ["IEF", "TLT", "GLD"]

# Sleeve definitions for expanded universe (sector ETFs)
RISK_ON_ASSETS_EXPANDED = RISK_ON_ASSETS_BASE + [
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
]
RISK_OFF_ASSETS_EXPANDED = RISK_OFF_ASSETS_BASE  # Same defensive assets

# Diverse multi-asset universe (v2): global equity + real assets + inflation hedges
RISK_ON_ASSETS_DIVERSE = RISK_ON_ASSETS_BASE + ["EFA", "EEM", "VNQ", "XLE"]
RISK_OFF_ASSETS_DIVERSE = RISK_OFF_ASSETS_BASE + ["TIP", "SHY", "DBC", "UUP"]
TICKERS_DIVERSE = RISK_ON_ASSETS_DIVERSE + RISK_OFF_ASSETS_DIVERSE
ASSETS_DIVERSE = TICKERS_DIVERSE + ["cash"]

# --- Backtest / vol scaling ---
VOL_LOOKBACK = 63  # ~3 months of trading days
VOL_EPS = 1e-8
MIN_VOL = 0.05
MAX_VOL = 2.00
COST_BPS = 0.0008  # 8 bps per dollar traded (one-way)
TOLERANCE_TAU = 0.015  # rebalance only when |target - current| > tau

# --- VIX override (caps risk_on when VIX elevated) ---
VIX_THRESHOLD = 25.0
VIX_RISK_ON_CAP = 0.25

# --- 200-day MA trend filter (disabled -- tested and rejected, see PROJECT_CONTEXT) ---
MA_LOOKBACK = 200
MA_EQUITY_CAP = 0.30

# --- Intramonth rebalance triggers (checked every Friday close) ---
VIX_INTRAMONTH_THRESHOLD = 27.0
SPY_WEEKLY_DROP_THRESHOLD = 0.99  # effectively disabled (too many false positives)

# --- HYG credit stress trigger (Exp D/E -- accepted) ---
ENABLE_CREDIT_TRIGGER = True
HYG_LOOKBACK_DAYS = 10
HYG_STRESS_THRESHOLD = -0.02  # HYG 10d return < -2% = stress
HYG_RECOVERY_THRESHOLD = 0.00  # HYG 10d return > 0% after stress = recovery
HYG_STRESS_MEMORY_DAYS = 30  # look-back window for "recent stress" state

# --- HYG/LQD z-score credit spread trigger (Exp F -- accepted) ---
ENABLE_CREDIT_ZSCORE = True
CREDIT_ZSCORE_LOOKBACK = 60  # rolling window for z-score
CREDIT_ZSCORE_STRESS = -1.5  # z < -1.5 = credit stress
CREDIT_ZSCORE_RECOVERY = -0.5  # z > -0.5 after stress = recovery

# --- Credit standalone mode: z-score + HYG replaces VIX as primary trigger ---
USE_CREDIT_STANDALONE = True

# --- Cross-sectional momentum tilt (Exp B -- monitoring, paper trading) ---
ENABLE_MOMENTUM_TILT = True
MOMENTUM_LOOKBACK_DAYS = 10
MOMENTUM_STRENGTH = 0.20
MOMENTUM_MAX_TILT = 2.0

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
    "Recovery": {"SPY": 0.15},
    "Overheating": {"SPY": 0.10},
    "Stagflation": {"GLD": 0.08},
    "Contraction": {"IEF": 0.05, "TLT": 0.05},
}

# --- Regime-specific max asset allocations (cap single-asset concentration) ---
REGIME_MAX_ASSETS: dict[str, dict[str, float]] = {
    "Recovery": {"GLD": 0.20, "TLT": 0.10, "IEF": 0.10},
    "Overheating": {"GLD": 0.25, "TLT": 0.10, "IEF": 0.10},
    "Stagflation": {"IJR": 0.15, "MTUM": 0.15},
}

# --- Min aggregate risk-on sleeve weight per regime ---
REGIME_MIN_RISK_ON: dict[str, float] = {
    "Recovery": 0.55,
    "Overheating": 0.45,
}

# --- Max aggregate risk-on sleeve weight per regime ---
REGIME_MAX_RISK_ON: dict[str, float] = {
    "Stagflation": 0.40,
    "Contraction": 0.30,
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
