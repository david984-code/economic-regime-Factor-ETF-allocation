"""Tests for src/research/bootstrap_significance.py — block bootstrap inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.bootstrap_significance import paired_block_bootstrap_delta_sharpe


def _identical_series(n: int = 60) -> tuple[pd.Series, pd.Series]:
    """Identical strategy and benchmark — true delta-Sharpe is exactly zero."""
    rng = np.random.default_rng(0)
    idx = pd.period_range("2015-01", periods=n, freq="M")
    r = pd.Series(rng.normal(0.008, 0.04, size=n), index=idx)
    return r.copy(), r.copy()


def _strong_outperformance(n: int = 120) -> tuple[pd.Series, pd.Series]:
    """Strategy ≈ benchmark + 1.5%/mo with identical noise — large, real delta."""
    rng = np.random.default_rng(1)
    idx = pd.period_range("2015-01", periods=n, freq="M")
    noise = rng.normal(0.0, 0.035, size=n)
    bench = pd.Series(0.005 + noise, index=idx)
    strat = pd.Series(0.020 + noise, index=idx)
    return strat, bench


class TestCenteredPValue:
    """The p-value must test H0:delta=0 by centering the bootstrap distribution.

    Regression guard: an earlier version used
        p = mean(|boot| >= |obs|)
    which measures distributional symmetry around obs_delta, not deviation from
    zero. That formulation returned p≈0.5 regardless of effect size.
    """

    def test_identical_series_high_p(self) -> None:
        s, b = _identical_series()
        res = paired_block_bootstrap_delta_sharpe(s, b, n_iterations=500, seed=0)
        # True delta is exactly zero on every resample → obs_delta == 0 and the
        # centered |boot - mean| >= 0 condition holds on every iteration → p == 1.
        assert res.observed_delta_sharpe == pytest.approx(0.0, abs=1e-9)
        assert res.p_value_two_sided > 0.5
        assert res.verdict == "fail_to_reject_H0"

    def test_large_effect_low_p(self) -> None:
        s, b = _strong_outperformance()
        res = paired_block_bootstrap_delta_sharpe(s, b, n_iterations=1000, seed=0)
        # 1.5%/mo persistent excess with matched noise is a huge Sharpe gap;
        # a correct centered test must reject H0.
        assert res.observed_delta_sharpe > 0.5
        assert res.p_value_two_sided < 0.05, (
            f"Got p={res.p_value_two_sided} for a huge true delta — "
            "looks like the bootstrap is not centered."
        )
        assert res.verdict == "reject_H0"

    def test_p_value_reasonable_for_modest_effect(self) -> None:
        """A modest noisy effect should give a p-value that's neither stuck at
        ~0.5 (uncentered bug) nor pathologically low/high."""
        rng = np.random.default_rng(2)
        n = 120
        idx = pd.period_range("2015-01", periods=n, freq="M")
        noise = rng.normal(0.0, 0.04, size=n)
        bench = pd.Series(0.006 + noise, index=idx)
        strat = pd.Series(0.009 + noise, index=idx)  # small ~0.3%/mo excess
        res = paired_block_bootstrap_delta_sharpe(strat, bench, n_iterations=1000, seed=0)
        # Should not be the suspiciously flat p~0.5 the uncentered bug produced.
        assert res.p_value_two_sided != pytest.approx(0.5, abs=0.05) or (
            res.observed_delta_sharpe == pytest.approx(0.0, abs=0.01)
        )
