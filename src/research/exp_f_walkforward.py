"""Baseline walk-forward OOS return series for bootstrap significance testing."""

from __future__ import annotations

import pandas as pd

from src.config import BASELINE_WALK_FORWARD, START_DATE, get_end_date
from src.evaluation.walk_forward import collect_walk_forward_oos_returns


def exp_f_walkforward(
    start: str | None = None,
    end: str | None = None,
    *,
    fast_mode: bool = False,
    max_segments: int | None = None,
) -> pd.Series:
    """Run production-baseline walk-forward; return stitched non-overlapping OOS daily returns."""
    return collect_walk_forward_oos_returns(
        start=start or START_DATE,
        end=end or get_end_date(),
        fast_mode=fast_mode,
        max_segments=max_segments,
        use_vol_regime=False,
        **BASELINE_WALK_FORWARD,
    )
