"""ML regime forecast: Gradient Boosting predictor for next month risk_on."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import OUTPUTS_DIR, get_end_date
from src.features.market_features import (
    build_momentum_features,
    build_seasonality_features,
)
from src.utils.database import Database

if TYPE_CHECKING:
    from src.data.pipeline_data import PipelineData

logger = logging.getLogger(__name__)


def load_regime_features() -> pd.DataFrame:
    """Load regime labels with macro features from CSV."""
    path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run regime classification first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").dropna(subset=["regime", "risk_on"])


def build_feature_matrix(
    regime_df: pd.DataFrame,
    market_df: pd.DataFrame,
    seasonal: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target y (next month risk_on)."""
    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_df["Period"] = regime_df["date"].dt.to_period("M")
    regime_df = regime_df.set_index("Period")

    feature_cols = ["gdp_z", "infl_z", "m2_z", "vel_z", "yield_level_z"]
    for c in ("pmi_z", "claims_z", "hy_spread_z"):
        if c in regime_df.columns:
            feature_cols.append(c)
    avail = [c for c in feature_cols if c in regime_df.columns]

    X_list: list[list[float]] = []
    y_list: list[float] = []
    dates: list[pd.Period] = []

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
            spy_1m = float(mkt.get("spy_1m", 0.0) or 0.0)
            spy_3m = float(mkt.get("spy_3m", 0.0) or 0.0)
            spy_6m = float(mkt.get("spy_6m", 0.0) or 0.0)
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
    """Walk-forward CV; return MAE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds: list[float] = []
    actuals: list[float] = []
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
        preds.extend(pred.tolist())
        actuals.extend(y_test.values.tolist())
    mae = float(np.mean(np.abs(np.array(preds) - np.array(actuals))))
    return mae


def predict_next_month(X: pd.DataFrame, y: pd.Series) -> float:
    """Train on full history and predict next month risk_on."""
    scaler = StandardScaler()
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
    return float(np.clip(pred, 0.0, 1.0))


def main(pipeline_data: "PipelineData | None" = None) -> None:
    """Run regime forecast and save to database.

    Args:
        pipeline_data: If provided, use cached market data. Otherwise fetch via build_momentum_features.
    """
    regime_df = load_regime_features()
    if pipeline_data is not None:
        market_df = pipeline_data.get_momentum_features(ticker="SPY")
        logger.debug("[DATA] Regime forecast using shared pipeline_data (cache hit)")
    else:
        market_df = build_momentum_features(end=get_end_date())
    seasonal = build_seasonality_features(regime_df)

    X, y = build_feature_matrix(regime_df, market_df, seasonal)
    if len(X) < 24:
        logger.warning("Not enough data for forecast (need 24+ months). Skipping.")
        return

    mae = evaluate_forecast(X, y)
    logger.info("Walk-forward MAE (risk_on): %.3f", mae)

    risk_on_pred = predict_next_month(X, y)
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

    logger.info("Next month (%s) predicted risk_on: %.3f", next_month, risk_on_pred)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
