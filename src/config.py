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


# --- Risk-free (single source of truth for Sharpe/Sortino and synthetic cash) ---
RF_ANNUAL = 0.045
RF_DAILY = float((1.0 + RF_ANNUAL) ** (1.0 / 252.0) - 1.0)
RF_MONTHLY = float((1.0 + RF_ANNUAL) ** (1.0 / 12.0) - 1.0)
# Separate from RF: softness term in optimizer objective (-Sortino + k*cash_weight)
OPTIMIZER_CASH_PREFERENCE = 0.05


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
    "XLRE", # Real Estate
    "XLC",  # Communication Services
]
ASSETS_EXPANDED = TICKERS_EXPANDED + ["cash"]

# Sleeve definitions for current universe
RISK_ON_ASSETS_BASE = ["SPY", "MTUM", "VLUE", "QUAL", "USMV", "IJR", "VIG"]
RISK_OFF_ASSETS_BASE = ["IEF", "TLT", "GLD"]

# Sleeve definitions for expanded universe (sector ETFs)
RISK_ON_ASSETS_EXPANDED = RISK_ON_ASSETS_BASE + [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"
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

# --- Production walk-forward baseline (single source of truth) ---
BASELINE_WALK_FORWARD: dict[str, object] = {
    # Signal knobs
    "market_lookback_months": 24,
    "sigmoid_scale": 0.25,
    "tolerance": 0.015,
    "trend_filter_type": "none",
    # Regime/portfolio construction
    "portfolio_construction_method": "equal_weight",
    "use_stagflation_override": False,
    "use_stagflation_risk_on_cap": False,
    "use_regime_smoothing": False,
    "use_hybrid_signal": True,
    "hybrid_macro_weight": 0.0,
    "use_momentum": True,
    "momentum_12m_weight": 0.0,
    "momentum_6m_weight": 0.0,
    "vol_scaling_method": "none",
    "use_post_blend_inv_vol": True,
    # Walk-forward design
    "min_train_months": 60,
    "test_months": 12,
    "expanding": True,
    "quarterly_rebalance": False,
}

# --- Regime-specific cash constraints (min, max) ---
REGIME_CASH: dict[str, tuple[float, float]] = {
    "Recovery": (0.05, 0.10),
    "Overheating": (0.05, 0.12),
    "Stagflation": (0.10, 0.20),
    # TODO(walk-forward): Contraction bounds are undocumented heuristic; an in-sample screen
    # suggested revisiting minimum cash — do NOT change without WF validation vs baseline.
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
