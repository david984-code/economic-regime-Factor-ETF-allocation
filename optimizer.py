import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

# Load data
import yfinance as yf
import subprocess
import sys


TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Load regimes (monthly labels)
regimes = pd.read_csv("regime_labels_expanded.csv", parse_dates=["date"], index_col="date")

# Download daily prices and convert to monthly returns
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
regimes["Period"] = regimes.index.to_period("M")

# Ensure required columns exist

if "cash" not in returns.columns:
    print("⚠️ Adding synthetic 'cash' column (~5% annualized).")
    CASH_ANNUAL_YIELD = 0.05
    CASH_MONTHLY = (1 + CASH_ANNUAL_YIELD) ** (1/12) - 1
    returns["cash"] = CASH_MONTHLY


# Define asset groups
all_assets = [col for col in returns.columns if col not in ["Period"]]
risky_assets = [a for a in all_assets if a not in ["cash"]]
full_asset_list = risky_assets + ["cash"]


# Merge datasets
merged = pd.merge(returns, regimes, on="Period", how="inner")
merged.set_index("Period", inplace=True)

def negative_sharpe(weights, mean_returns_risky, cov_matrix_risky, risk_free=0.0):
    """
    weights: full vector [risky..., cash]
    mean_returns_risky: array for risky assets only (no cash)
    cov_matrix_risky: covariance for risky assets only (no cash)
    """
    # Split risky vs cash
    risky_weights = np.asarray(weights[:-1], dtype=float)
    cash_weight = float(weights[-1])

    # If there is essentially no risky exposure, reject
    risky_sum = risky_weights.sum()
    if risky_sum <= 1e-10:
        return 1e9

    # Normalize risky weights so Sharpe is based on the MIX, not the exposure size
    w = risky_weights / risky_sum

    mu = np.asarray(mean_returns_risky, dtype=float)
    cov = np.asarray(cov_matrix_risky, dtype=float)

    port_return = float(np.dot(w, mu))
    port_var = float(w.T @ cov @ w)

    if not np.isfinite(port_var) or port_var <= 1e-12:
        return 1e9

    port_vol = np.sqrt(port_var)
    sharpe = (port_return - risk_free) / port_vol

    if not np.isfinite(sharpe):
        return 1e9

    # Small penalty so optimizer doesn't always hug max_cash boundary
    cash_penalty = 0.05 * cash_weight   # tune 0.01–0.10 if needed

    return -sharpe + cash_penalty


# Constraints & bounds
min_cash = 0.05
max_cash = 0.15



def get_constraints(num_assets):
    return [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},                 # full allocation
        {"type": "ineq", "fun": lambda w: w[-1] - min_cash},            # cash >= min_cash
        {"type": "ineq", "fun": lambda w: max_cash - w[-1]}             # cash <= max_cash
    ]


# Store results
optimal_allocations = {}

# Optimization loop
for regime in merged["regime"].unique():
    print(f"\n🔍 Optimizing for regime: {regime}")
    subset = merged[merged["regime"] == regime]
    subset_risky = subset[risky_assets].fillna(0)

    if len(subset_risky) < 2:
        print(f"⚠️ Skipping regime {regime}: not enough data.")
        continue

    # Use risky-only mean/cov for the Sharpe objective (negative_sharpe normalizes risky weights)
    mean_returns_risky = subset_risky.mean().values.astype(float)
    cov_matrix_risky = subset_risky.cov().values.astype(float)

    # Ridge regularization to prevent singular / near-singular covariance
    eps = 1e-6
    cov_matrix_risky = cov_matrix_risky + eps * np.eye(cov_matrix_risky.shape[0])


    n = len(full_asset_list)

    # Feasible init: start cash at midpoint of [min_cash, max_cash]
    init_guess = np.full(n, 0.0)
    start_cash = (min_cash + max_cash) / 2
    init_guess[:-1] = (1 - start_cash) / (n - 1)
    init_guess[-1] = start_cash


    bounds = [(0.0, 1.0)] * len(full_asset_list)

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns_risky, cov_matrix_risky,),
        method="SLSQP",
        bounds=bounds,
        constraints=get_constraints(len(full_asset_list))
    )

    if result.success:
        allocation = dict(zip(full_asset_list, result.x))
        optimal_allocations[regime] = allocation
        print(f"✅ Success: {regime}")
    else:
        print(f"❌ Failure for {regime}: {result.message}")

# Save results
if optimal_allocations:
    df_opt = pd.DataFrame(optimal_allocations).T
    df_opt.index.name = "regime"
    df_opt.to_csv("optimal_allocations.csv")
    print("✅ Saved optimal_allocations.csv")

    print("📄 Formatting allocations into Excel...")
    subprocess.run([sys.executable, "format_allocations.py"], check=True)
    print("✅ Saved optimal_allocations_formatted.xlsx")

else:
    print("⚠️ No successful optimizations. CSV not saved.")

