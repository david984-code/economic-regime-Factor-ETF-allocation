from src.optimizer import optimize_weights
from src.backtest import run_backtest
from src.format_allocations import format_allocations


import pandas as pd
from fredapi import Fred
import os
from datetime import datetime
import numpy as np


# FRED API (use env var if set)

FRED_API_KEY = os.getenv("FRED_API_KEY", "c17bfdfc6b7586911695813f32c0157a")
fred = Fred(api_key=FRED_API_KEY)

def to_month_end(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("M").to_timestamp("M")  # force month-end timestamps
    # If multiple points map into same month-end, keep the last one
    s = s[~s.index.duplicated(keep="last")]
    return s

# Fetch Data (force pulls to today)

end = datetime.today().strftime("%Y-%m-%d")
gdp = fred.get_series("GDPC1", observation_end=end)        # real GDP (quarterly)
cpi = fred.get_series("CPIAUCSL", observation_end=end)     # CPI (monthly)
yield_10y = fred.get_series("GS10", observation_end=end)   # 10Y (daily)
yield_3m = fred.get_series("GS3M", observation_end=end)    # 3M (daily)
m2 = fred.get_series("M2SL", observation_end=end)          # M2 (weekly)
velocity = fred.get_series("M2V", observation_end=end)     # velocity (quarterly)

# Index Setup
for s in (gdp, cpi, yield_10y, yield_3m, m2, velocity):
    s.index = pd.to_datetime(s.index)

print("\n🧾 Latest available dates pulled from FRED:")
print("GDP:", gdp.index.max())
print("CPI:", cpi.index.max())
print("10Y:", yield_10y.index.max())
print("3M:", yield_3m.index.max())
print("M2:", m2.index.max())
print("Velocity (M2V):", velocity.index.max())


# Resample to Monthly (month-end)

# Note: some series are not monthly; forward-fill to month-end
gdp = gdp.resample("ME").ffill()
yield_10y = yield_10y.resample("ME").ffill()
yield_3m = yield_3m.resample("ME").ffill()
m2 = m2.resample("ME").ffill()
velocity = velocity.resample("ME").ffill()
# CPI is already monthly but sometimes lands on month start; force to month-end
cpi.index = (cpi.index + pd.offsets.MonthEnd(0))

# Yield Curve (level)
# Normalize indexes to month-end for clean merging
gdp = to_month_end(gdp)
cpi = to_month_end(cpi)
m2 = to_month_end(m2)
velocity = to_month_end(velocity)

# Rates are daily — resample to month-end first, then normalize
yield_curve = (yield_10y - yield_3m).rename("yield_curve")


# Helper: rolling z-score on MoM changes
def rolling_z(x: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
    """Rolling z-score of a series (x - rolling_mean) / rolling_std."""
    m = x.rolling(window=window, min_periods=min_periods).mean()
    s = x.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (x - m) / s.replace(0, np.nan)

def sigmoid(a: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-a))


# Force rates to clean month-end index too
yield_10y = to_month_end(yield_10y)
yield_3m  = to_month_end(yield_3m)

# Master month-end index (continuous timeline)
start = min(
    gdp.index.min(), cpi.index.min(), yield_10y.index.min(),
    yield_3m.index.min(), m2.index.min(), velocity.index.min()
)
end = max(
    gdp.index.max(), cpi.index.max(), yield_10y.index.max(),
    yield_3m.index.max(), m2.index.max(), velocity.index.max()
)

monthly_index = pd.date_range(
    start=start.to_period("M").to_timestamp("M"),
    end=end.to_period("M").to_timestamp("M"),
    freq="ME"
)

# Reindex + forward-fill (stale data is OK per your choice A)
gdp_m      = gdp.reindex(monthly_index).ffill()
cpi_m      = cpi.reindex(monthly_index).ffill()
y10_m      = yield_10y.reindex(monthly_index).ffill()
y3m_m      = yield_3m.reindex(monthly_index).ffill()
m2_m       = m2.reindex(monthly_index).ffill()
velocity_m = velocity.reindex(monthly_index).ffill()

yield_curve_m = (y10_m - y3m_m).rename("yield_curve")

# Build df directly (no merges, no missing months)
df = pd.DataFrame(index=monthly_index)
df.index.name = "date"

df["gdp_mom"] = gdp_m.pct_change()
df["cpi_mom"] = cpi_m.pct_change()
df["yield_curve"] = yield_curve_m
df["m2_mom"] = m2_m.pct_change()
df["vel_mom"] = velocity_m.pct_change()

df = df.reset_index()


# Yield curve can be used as a level signal; we z-score the level itself
df["yield_level_z"] = rolling_z(df["yield_curve"], window=60, min_periods=24)

# Z-scores of MoM changes (this is the key change vs your old code)
df["gdp_z"] = rolling_z(df["gdp_mom"], window=60, min_periods=24)
df["infl_z"] = rolling_z(df["cpi_mom"], window=60, min_periods=24)
df["m2_z"] = rolling_z(df["m2_mom"], window=60, min_periods=24)
df["vel_z"] = rolling_z(df["vel_mom"], window=60, min_periods=24)

# Regime classification USING z-scores

def classify_regime_z(row):
    # Require enough data for at least inflation + liquidity proxy (m2/vel)
    if pd.isna(row["infl_z"]) or (pd.isna(row["m2_z"]) and pd.isna(row["vel_z"])):
        return None

    gdp_z = row["gdp_z"]
    infl_z = row["infl_z"]
    liq_z = np.nanmean([row["m2_z"], row["vel_z"]])
    yc_z = row["yield_level_z"]

    # If GDP z not available, fall back to yield curve + liquidity
    has_gdp = not pd.isna(gdp_z)

    # Simple, consistent rules:
    # - Overheating: inflation high AND (growth or liquidity strong)
    # - Recovery: inflation low AND (growth or liquidity strong)
    # - Stagflation: inflation high AND (growth weak)
    # - Contraction: inflation low AND (liquidity weak OR yield curve very negative)
    if has_gdp:
        if infl_z > 0.5 and (gdp_z > 0.0 or liq_z > 0.0):
            return "Overheating"
        if infl_z <= 0.5 and (gdp_z > 0.0 or liq_z > 0.0):
            return "Recovery"
        if infl_z > 0.5 and gdp_z <= 0.0:
            return "Stagflation"
        # inflation not high + growth not strong => contraction-ish
        return "Contraction"
    else:
        # No GDP: use yield curve (inversion) and liquidity as growth proxy
        if infl_z > 0.5 and liq_z > 0.0:
            return "Overheating"
        if infl_z <= 0.5 and liq_z > 0.0:
            return "Recovery"
        if infl_z > 0.5 and (liq_z <= 0.0 or (not pd.isna(yc_z) and yc_z < -0.5)):
            return "Stagflation"
        return "Contraction"

df["regime"] = df.apply(classify_regime_z, axis=1)

# Continuous risk-on score (0..1): higher growth, lower inflation, steeper curve => more risk-on
macro_components = pd.concat(
    [df["gdp_z"], df["yield_level_z"], df["m2_z"], df["vel_z"]],
    axis=1
)
df["macro_score"] = macro_components.mean(axis=1, skipna=True)

# If ALL components are NaN for a month, treat it as neutral (0 => sigmoid=0.5)
df["macro_score"] = df["macro_score"].fillna(0.0)

df["risk_on"] = sigmoid(df["macro_score"])  # 0..1


# Print Recent Results

classified = df[df["regime"].notna()]
classified_recent = classified[classified["date"] >= pd.to_datetime("2020-01-01")]

print("\n🗓️ Regime Classification from 2020 to Latest:")
print(classified_recent[["date", "regime", "risk_on"]].tail(24))

latest = classified_recent.iloc[-1]
print("\n🔎 Latest Classified Month:")
print("Date:", latest["date"].strftime("%Y-%m-%d"))
print("Regime:", latest["regime"])
print("Risk-on score (0..1):", round(float(latest["risk_on"]), 3))
print("Suggested Allocation: (generated in optimizer.py using factor ETFs)")

print("\n📊 Regime Count Since 2020:")
print(classified_recent["regime"].value_counts())

# Include regime + risk_on + z-scores so backtest can scale weights within regime.
out_cols = ["date", "regime", "risk_on", "macro_score", "gdp_z", "infl_z", "m2_z", "vel_z", "yield_level_z"]
regime_df = df[out_cols].dropna(subset=["regime"]).copy()

project_folder = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(project_folder, "regime_labels_expanded.csv")

try:
    regime_df.to_csv(save_path, index=False)
    print("\n✅ Saved:", save_path)
except PermissionError:
    fallback = os.path.join(project_folder, f"regime_labels_expanded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    regime_df.to_csv(fallback, index=False)
    print("\n⚠️ File was locked. Saved instead to:", fallback)
