"""Macro feature engineering: resample, z-scores, macro score."""

from typing import Any

import numpy as np
import pandas as pd

from src.features.transforms import rolling_z_score, sigmoid, to_month_end


def resample_to_monthly(
    gdp: pd.Series,
    cpi: pd.Series,
    yield_10y: pd.Series,
    yield_3m: pd.Series,
    m2: pd.Series,
    velocity: pd.Series,
) -> tuple[pd.Series, ...]:
    """Resample all series to monthly frequency (month-end)."""
    gdp = gdp.resample("ME").ffill()
    yield_10y = yield_10y.resample("ME").ffill()
    yield_3m = yield_3m.resample("ME").ffill()
    m2 = m2.resample("ME").ffill()
    velocity = velocity.resample("ME").ffill()
    cpi.index = cpi.index + pd.offsets.MonthEnd(0)

    gdp = to_month_end(gdp)
    cpi = to_month_end(cpi)
    m2 = to_month_end(m2)
    velocity = to_month_end(velocity)
    yield_10y = to_month_end(yield_10y)
    yield_3m = to_month_end(yield_3m)
    return gdp, cpi, yield_10y, yield_3m, m2, velocity


def resample_high_freq_to_monthly(hf: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Resample high-frequency series to month-end."""
    out: dict[str, pd.Series] = {}
    for name, s in hf.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        if name == "claims":
            s = s.resample("ME").mean()
        else:
            s = s.resample("ME").last().ffill()
        s = to_month_end(s)
        out[name] = s
    return out


def build_macro_dataframe(
    gdp: pd.Series,
    cpi: pd.Series,
    yield_10y: pd.Series,
    yield_3m: pd.Series,
    m2: pd.Series,
    velocity: pd.Series,
    hf_monthly: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build dataframe with raw indicators and month-over-month changes (no z-scores)."""
    gdp_mom = gdp.pct_change()
    cpi_mom = cpi.pct_change()
    m2_mom = m2.pct_change()
    vel_mom = velocity.pct_change()
    yield_curve = (yield_10y - yield_3m).rename("yield_curve")

    data: dict[str, pd.Series] = {
        "gdp": gdp,
        "cpi": cpi,
        "gdp_mom": gdp_mom,
        "cpi_mom": cpi_mom,
        "m2_mom": m2_mom,
        "vel_mom": vel_mom,
        "yield_curve": yield_curve,
    }

    if hf_monthly:
        for name, s in hf_monthly.items():
            if name == "pmi":
                data["pmi"] = s
                data["pmi_mom"] = s.pct_change()
            elif name == "claims":
                data["claims"] = s
                data["claims_mom"] = -s.pct_change()
            elif name == "hy_spread":
                data["hy_spread"] = s
                data["hy_spread_mom"] = -s.pct_change()

    return pd.DataFrame(data)


def add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling z-scores to macro dataframe."""
    df["gdp_z"] = rolling_z_score(df["gdp_mom"])
    df["infl_z"] = rolling_z_score(df["cpi_mom"])
    df["m2_z"] = rolling_z_score(df["m2_mom"])
    df["vel_z"] = rolling_z_score(df["vel_mom"])
    df["yield_level_z"] = rolling_z_score(df["yield_curve"])
    if "pmi_mom" in df.columns:
        df["pmi_z"] = rolling_z_score(df["pmi_mom"])
    if "claims_mom" in df.columns:
        df["claims_z"] = rolling_z_score(df["claims_mom"])
    if "hy_spread_mom" in df.columns:
        df["hy_spread_z"] = rolling_z_score(df["hy_spread_mom"])
    return df


def calculate_macro_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add macro_score and risk_on columns. Requires z-score columns."""
    """Add macro_score and risk_on columns to dataframe."""
    base = df["gdp_z"] + df["vel_z"] - df["infl_z"] + df["m2_z"] + df["yield_level_z"]
    if "pmi_z" in df.columns:
        base = base + df["pmi_z"].fillna(0)
    if "claims_z" in df.columns:
        base = base + df["claims_z"].fillna(0)
    if "hy_spread_z" in df.columns:
        base = base + df["hy_spread_z"].fillna(0)
    df["macro_score"] = base
    df["risk_on"] = sigmoid(df["macro_score"] * 0.25)
    return df
