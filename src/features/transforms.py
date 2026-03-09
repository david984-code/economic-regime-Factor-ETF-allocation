"""Time series transforms: month-end, rolling z-score, sigmoid."""

import numpy as np
import pandas as pd


def to_month_end(series: pd.Series) -> pd.Series:
    """Convert series index to month-end timestamps.

    Args:
        series: Input time series.

    Returns:
        Series with month-end index, duplicates dropped (keep last).
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("M").to_timestamp("M")
    s = s[~s.index.duplicated(keep="last")]
    return s


def rolling_z_score(
    series: pd.Series,
    window: int = 60,
    min_periods: int = 24,
) -> pd.Series:
    """Compute rolling z-score.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum periods required.

    Returns:
        Rolling z-scores.
    """
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def sigmoid(series: pd.Series) -> pd.Series:
    """Apply sigmoid transformation: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-series))  # type: ignore[no-any-return]
