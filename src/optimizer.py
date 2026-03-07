import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Load data
import yfinance as yf
from scipy.optimize import minimize

from .database import Database

ROOT_DIR = (
    Path(__file__).resolve().parents[1]
)  # .../Economic-Regime-Asset-Allocation-With-Fred-main
OUTPUTS_DIR = ROOT_DIR / "outputs"


TICKERS = [
    "SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG",
    "IEF", "TLT",
]
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


# Load regimes (monthly labels) from database
def load_regimes() -> pd.DataFrame:
    """Load regime labels from database, fallback to CSV if needed."""
    db = Database()
    try:
        df = db.load_regime_labels()
        db.close()
        return df
    except Exception as e:
        db.close()
        # Fallback to CSV for backwards compatibility
        path = OUTPUTS_DIR / "regime_labels_expanded.csv"
        if not path.exists():
            raise FileNotFoundError(
                "No regime data found in database or CSV. "
                "Run `python -m src.economic_regime` first."
            ) from e
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df


# Download daily prices and convert to monthly returns
def main() -> None:

    px = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(px.columns, pd.MultiIndex):
        px = px["Adj Close"] if "Adj Close" in px.columns.levels[0] else px["Close"]
    else:
        px = px["Adj Close"] if "Adj Close" in px.columns else px["Close"]

    px = px.dropna()

    # month-end prices -> monthly returns
    monthly_px = px.resample("ME").last()
    returns = monthly_px.pct_change().dropna()
    returns.index.name = "Date"

    # Add Period column
    returns["Period"] = returns.index.to_period("M")

    # Load regimes and add Period column
    regimes = load_regimes()
    regimes["Period"] = pd.to_datetime(regimes.index).to_period("M")

    # Ensure required columns exist

    if "cash" not in returns.columns:
        print("[WARNING] Adding synthetic 'cash' column (~5% annualized).")
        cash_annual_yield = 0.05
        cash_monthly = (1 + cash_annual_yield) ** (1 / 12) - 1
        returns["cash"] = cash_monthly

    # Define asset groups
    all_assets = [col for col in returns.columns if col not in ["Period"]]
    risky_assets = [a for a in all_assets if a not in ["cash"]]
    full_asset_list = risky_assets + ["cash"]

    # Merge datasets
    merged = pd.merge(returns, regimes, on="Period", how="inner")
    merged.set_index("Period", inplace=True)

    def negative_sortino(
        weights: np.ndarray,
        returns_risky: np.ndarray,
        risk_free: float = 0.0,
    ) -> float:
        """
        Sortino ratio: return / downside_deviation (penalizes only downside vol).
        weights: full vector [risky..., cash]
        returns_risky: (T x n) array of monthly returns for risky assets
        """
        risky_weights = np.asarray(weights[:-1], dtype=float)
        cash_weight = float(weights[-1])
        risky_sum = risky_weights.sum()
        if risky_sum <= 1e-10:
            return 1e9

        w = risky_weights / risky_sum
        port_rets = returns_risky @ w
        mean_ret = float(np.mean(port_rets))
        downside_rets = np.minimum(port_rets - risk_free, 0.0)
        downside_var = float(np.mean(downside_rets**2))
        downside_vol = np.sqrt(downside_var) if downside_var > 1e-12 else 1e-8

        sortino = (mean_ret - risk_free) / downside_vol
        if not np.isfinite(sortino):
            return 1e9

        cash_penalty = 0.05 * cash_weight
        return float(-sortino + cash_penalty)

    # Regime-specific cash constraints (risk-off = more cash)
    REGIME_CASH = {
        "Recovery": (0.05, 0.10),
        "Overheating": (0.05, 0.12),
        "Stagflation": (0.10, 0.20),
        "Contraction": (0.15, 0.30),
        "Unknown": (0.10, 0.20),
    }
    default_min, default_max = 0.05, 0.15

    # Regime-specific min allocations: e.g. min gold in Stagflation, min bonds in Contraction
    REGIME_MIN_ASSETS: dict[str, dict[str, float]] = {
        "Stagflation": {"GLD": 0.08},
        "Contraction": {"IEF": 0.05, "TLT": 0.05},
    }

    def get_constraints(
        num_assets: int, regime: str, asset_list: list[str]
    ) -> list[dict[str, object]]:
        min_cash, max_cash = REGIME_CASH.get(
            regime, (default_min, default_max)
        )
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w, mc=min_cash: w[-1] - mc},
            {"type": "ineq", "fun": lambda w, mx=max_cash: mx - w[-1]},
        ]
        min_assets = REGIME_MIN_ASSETS.get(regime, {})
        for asset, min_w in min_assets.items():
            if asset in asset_list:
                idx = asset_list.index(asset)
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, i=idx, m=min_w: w[i] - m,
                })
        return constraints

    # Store results
    optimal_allocations = {}

    # Optimization loop (Sortino ratio, regime-specific min allocations)
    for regime in merged["regime"].unique():
        print(f"\n[OPTIMIZE] Optimizing for regime: {regime} (Sortino)")
        subset = merged[merged["regime"] == regime]
        subset_risky = subset[risky_assets].fillna(0)

        if len(subset_risky) < 2:
            print(f"[WARNING] Skipping regime {regime}: not enough data.")
            continue

        returns_risky = subset_risky.values.astype(float)
        n = len(full_asset_list)
        min_cash, max_cash = REGIME_CASH.get(
            regime, (default_min, default_max)
        )

        # Feasible init: start cash at midpoint of [min_cash, max_cash]
        init_guess = np.full(n, 0.0)
        start_cash = (min_cash + max_cash) / 2
        init_guess[:-1] = (1 - start_cash) / (n - 1)
        init_guess[-1] = start_cash

        bounds = [(0.0, 1.0)] * len(full_asset_list)

        result = minimize(
            negative_sortino,
            init_guess,
            args=(returns_risky,),
            method="SLSQP",
            bounds=bounds,
            constraints=get_constraints(n, regime, risky_assets),
        )

        if result.success:
            allocation = dict(zip(full_asset_list, result.x))
            optimal_allocations[regime] = allocation
            print(f"[SUCCESS] Optimized: {regime}")
        else:
            print(f"[ERROR] Failure for {regime}: {result.message}")

    # Save results to database and CSV
    if optimal_allocations:
        # Save to database (primary storage)
        db = Database()
        db.save_optimal_allocations(optimal_allocations)
        db.close()
        print("[SUCCESS] Saved optimal allocations to database")

        # Also save to CSV for backwards compatibility
        df_opt = pd.DataFrame(optimal_allocations).T
        df_opt.index.name = "regime"
        OUTPUTS_DIR.mkdir(exist_ok=True)
        out_csv = OUTPUTS_DIR / "optimal_allocations.csv"
        df_opt.to_csv(out_csv)
        print(f"[SUCCESS] Saved CSV backup: {out_csv}")

        print("[FORMAT] Formatting allocations into Excel...")
        subprocess.run(
            [sys.executable, str(ROOT_DIR / "src" / "format_allocations.py")], check=True
        )
        print("[SUCCESS] Saved optimal_allocations_formatted.xlsx")

    else:
        print("[WARNING] No successful optimizations. CSV not saved.")


if __name__ == "__main__":
    main()
