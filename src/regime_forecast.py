"""Regime Forecast Module.

Predicts next month's regime (risk_on score) using:
- Macro z-scores (GDP, inflation, yield curve, etc.)
- Market momentum (SPY 1/3/6 month returns)
- Seasonality (month-of-year historical return patterns)
- Gradient Boosting ML model

Designed for active trading: forecast-driven allocation instead of
pure backtest/reactive regime labeling.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .database import Database

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
START_DATE = "2010-01-01"


def load_regime_features() -> pd.DataFrame:
    """Load regime labels with macro features from CSV (from economic_regime output)."""
    path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.economic_regime` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["regime", "risk_on"])
    return df


def build_market_features(end_date: str) -> pd.DataFrame:
    """Fetch SPY returns and compute momentum features."""
    px = yf.download("SPY", start=START_DATE, end=end_date, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Adj Close"] if "Adj Close" in px.columns.levels[0] else px["Close"]
    else:
        px = px["Adj Close"] if "Adj Close" in px.columns else px["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.squeeze()
    px = pd.Series(px).dropna()
    monthly = px.resample("ME").last()
    ret_1m = monthly.pct_change(1)
    ret_3m = monthly.pct_change(3)
    ret_6m = monthly.pct_change(6)
    df = pd.DataFrame(
        {"spy_1m": ret_1m, "spy_3m": ret_3m, "spy_6m": ret_6m},
        index=monthly.index,
    )
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def build_seasonality_features(regime_df: pd.DataFrame) -> pd.Series:
    """
    Month-of-year seasonality: average historical risk_on by month.
    Months with historically higher risk_on = more bullish seasonality.
    """
    regime_df = regime_df.copy()
    regime_df["month"] = pd.to_datetime(regime_df["date"]).dt.month
    seasonal = regime_df.groupby("month")["risk_on"].mean()
    return seasonal


def build_feature_matrix(
    regime_df: pd.DataFrame,
    market_df: pd.DataFrame,
    seasonal: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y (next month risk_on).
    Features: macro z-scores, momentum, seasonality.
    """
    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_df["Period"] = regime_df["date"].dt.to_period("M")
    regime_df = regime_df.set_index("Period")

    feature_cols = ["gdp_z", "infl_z", "m2_z", "vel_z", "yield_level_z"]
    for c in ("pmi_z", "claims_z", "hy_spread_z"):
        if c in regime_df.columns:
            feature_cols.append(c)
    avail = [c for c in feature_cols if c in regime_df.columns]

    X_list = []
    y_list = []
    dates = []

    for i in range(len(regime_df) - 1):
        row = regime_df.iloc[i]
        next_row = regime_df.iloc[i + 1]
        period = regime_df.index[i]
        next_period = regime_df.index[i + 1]

        macro = [float(row.get(c, 0.0)) for c in avail]
        if any(pd.isna(macro)):
            continue

        month = next_period.to_timestamp().month
        seas = float(seasonal.get(month, 0.5))

        period_ts = period.to_timestamp("M")
        if period_ts in market_df.index:
            mkt = market_df.loc[period_ts]
            spy_1m = float(mkt.get("spy_1m", 0.0)) if pd.notna(mkt.get("spy_1m")) else 0.0
            spy_3m = float(mkt.get("spy_3m", 0.0)) if pd.notna(mkt.get("spy_3m")) else 0.0
            spy_6m = float(mkt.get("spy_6m", 0.0)) if pd.notna(mkt.get("spy_6m")) else 0.0
        else:
            spy_1m = spy_3m = spy_6m = 0.0

        feat = macro + [spy_1m, spy_3m, spy_6m, seas, month / 12.0]
        X_list.append(feat)
        y_list.append(float(next_row["risk_on"]))
        dates.append(next_period)

    col_names = avail + ["spy_1m", "spy_3m", "spy_6m", "seasonality", "month_norm"]
    X = pd.DataFrame(X_list, index=dates, columns=col_names)
    y = pd.Series(y_list, index=dates)
    return X, y


def evaluate_forecast(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Walk-forward cross-validation: train on past, predict next month."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = []
    actuals = []
    scaler = StandardScaler()
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_scaled = scaler.fit_transform(X_train.fillna(0))
        X_test_scaled = scaler.transform(X_test.fillna(0))
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        preds.extend(pred)
        actuals.extend(y_test.values)
    preds = np.array(preds)
    actuals = np.array(actuals)
    mae = float(np.mean(np.abs(preds - actuals)))
    return mae


def predict_next_month(
    X: pd.DataFrame,
    y: pd.Series,
    scaler: StandardScaler | None = None,
    model: GradientBoostingRegressor | None = None,
) -> tuple[float, StandardScaler, GradientBoostingRegressor]:
    """Train on full history and predict next month's risk_on."""
    if scaler is None:
        scaler = StandardScaler()
    if model is None:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    X_scaled = scaler.fit_transform(X.fillna(0))
    model.fit(X_scaled, y.values)
    last = X.iloc[[-1]].fillna(0)
    last_scaled = scaler.transform(last)
    pred = float(model.predict(last_scaled)[0])
    pred = float(np.clip(pred, 0.0, 1.0))
    return pred, scaler, model


def main() -> None:
    """Run regime forecast: train ML model, predict next month, save to DB."""
    end_date = datetime.today().strftime("%Y-%m-%d")

    regime_df = load_regime_features()
    market_df = build_market_features(end_date)
    seasonal = build_seasonality_features(regime_df)

    X, y = build_feature_matrix(regime_df, market_df, seasonal)
    if len(X) < 24:
        print("[WARNING] Not enough data for forecast (need 24+ months). Skipping.")
        return

    mae = evaluate_forecast(X, y)
    print(f"\n[FORECAST] Walk-forward MAE (risk_on): {mae:.3f}")

    risk_on_pred, _, _ = predict_next_month(X, y)

    next_month = (pd.Timestamp.today().to_period("M") + 1).strftime("%Y-%m")
    forecast_date = datetime.today().strftime("%Y-%m-%d")

    db = Database()
    db.save_regime_forecast(
        forecast_date=forecast_date,
        target_month=next_month,
        risk_on_forecast=risk_on_pred,
        regime_forecast=None,
        accuracy_1m=mae,
    )
    db.close()

    print(f"[FORECAST] Next month ({next_month}) predicted risk_on: {risk_on_pred:.3f}")
    print("[SUCCESS] Saved forecast to database")


if __name__ == "__main__":
    main()
