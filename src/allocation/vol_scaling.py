"""Inverse-volatility weight scaling."""

import numpy as np
import polars as pl

from src.config import MAX_VOL, MIN_VOL, VOL_EPS


def vol_scaled_weights_from_std(
    raw_w: dict[str, float],
    std_by_asset: dict[str, float],
    risky_assets: list[str],
) -> dict[str, float]:
    """Scale weights by inverse volatility using precomputed std. No Polars.

    Same logic as vol_scaled_weights but takes std dict instead of trailing returns.
    """
    w = raw_w.copy()
    if len(risky_assets) == 0:
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    vol_dict = dict(std_by_asset)
    valid_vols = [v for v in vol_dict.values() if v is not None and not np.isnan(v)]
    if not valid_vols:
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}
    vol_median = float(np.median(valid_vols))

    for asset in risky_assets:
        v = vol_dict.get(asset)
        if v is None or not np.isfinite(v):
            v = vol_median if vol_median > 0 else VOL_EPS
        else:
            v = max(MIN_VOL * vol_median, min(MAX_VOL * vol_median, v))
            if v == 0.0:
                v = VOL_EPS
        vol_dict[asset] = v

    for asset in risky_assets:
        w[asset] = w[asset] / vol_dict[asset]
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def vol_scaled_weights(
    raw_w: dict[str, float],
    trailing_rets_pl: pl.DataFrame,
    risky_assets: list[str],
) -> dict[str, float]:
    """Scale weights by inverse volatility using Polars.

    Args:
        raw_w: Dict mapping asset -> weight (includes cash).
        trailing_rets_pl: Polars DataFrame with daily returns.
        risky_assets: List of risky assets (excludes cash).

    Returns:
        Dict of volatility-scaled weights.
    """
    w = raw_w.copy()
    if len(risky_assets) == 0:
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    vol_pl = trailing_rets_pl.select([pl.col(col).std() for col in risky_assets])
    vol_values = vol_pl.row(0)
    vol_dict = dict(zip(risky_assets, vol_values, strict=False))
    valid_vols = [v for v in vol_dict.values() if v is not None and not np.isnan(v)]
    if not valid_vols:
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}
    vol_median = float(np.median(valid_vols))

    for asset in risky_assets:
        v = vol_dict[asset]
        if v is None or not np.isfinite(v):
            v = vol_median if vol_median > 0 else VOL_EPS
        else:
            v = max(MIN_VOL * vol_median, min(MAX_VOL * vol_median, v))
            if v == 0.0:
                v = VOL_EPS
        vol_dict[asset] = v

    for asset in risky_assets:
        w[asset] = w[asset] / vol_dict[asset]
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}
