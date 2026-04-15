"""Tests for config loading, risk_on score, weight normalization, and cache staleness."""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import (
    COST_BPS,
    DEFAULT_MAX_CASH,
    DEFAULT_MIN_CASH,
    EQUITY_TICKERS,
    REGIME_CASH,
    REGIME_MAX_RISK_ON,
    REGIME_MIN_RISK_ON,
    RISK_OFF_ASSETS_BASE,
    RISK_ON_ASSETS_BASE,
    TICKERS,
    TOLERANCE_TAU,
    VOL_LOOKBACK,
)
from src.features.transforms import sigmoid


class TestConfigDefaults:
    """Verify config module has expected defaults and types."""

    def test_tickers_is_list(self) -> None:
        assert isinstance(TICKERS, list)
        assert len(TICKERS) >= 8

    def test_vol_lookback_positive(self) -> None:
        assert VOL_LOOKBACK > 0
        assert isinstance(VOL_LOOKBACK, int)

    def test_tolerance_tau_range(self) -> None:
        assert 0.0 < TOLERANCE_TAU < 0.10

    def test_cost_bps_reasonable(self) -> None:
        assert 0 < COST_BPS < 0.01

    def test_regime_cash_has_four_regimes(self) -> None:
        for regime in ("Recovery", "Overheating", "Stagflation", "Contraction"):
            assert regime in REGIME_CASH
            lo, hi = REGIME_CASH[regime]
            assert 0.0 <= lo < hi <= 1.0

    def test_default_cash_bounds(self) -> None:
        assert 0.0 < DEFAULT_MIN_CASH < DEFAULT_MAX_CASH < 1.0

    def test_risk_on_off_sleeves_partition(self) -> None:
        on_set = set(RISK_ON_ASSETS_BASE)
        off_set = set(RISK_OFF_ASSETS_BASE)
        assert on_set.isdisjoint(off_set)
        assert on_set | off_set == set(TICKERS)

    def test_equity_tickers_alias(self) -> None:
        assert EQUITY_TICKERS == RISK_ON_ASSETS_BASE

    def test_regime_constraints_consistent(self) -> None:
        for regime, min_ro in REGIME_MIN_RISK_ON.items():
            assert 0.0 < min_ro <= 1.0, f"{regime} min_risk_on out of range"
        for regime, max_ro in REGIME_MAX_RISK_ON.items():
            assert 0.0 < max_ro <= 1.0, f"{regime} max_risk_on out of range"


class TestRiskOnScore:
    """Verify sigmoid-based risk_on score properties."""

    def test_output_range_zero_to_one(self) -> None:
        scores = pd.Series(np.linspace(-10, 10, 1000))
        result = sigmoid(scores)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_monotonically_increasing(self) -> None:
        scores = pd.Series(np.linspace(-5, 5, 200))
        result = sigmoid(scores)
        diffs = result.diff().dropna()
        assert (diffs >= 0).all()

    def test_zero_input_gives_half(self) -> None:
        result = sigmoid(pd.Series([0.0]))
        assert abs(float(result.iloc[0]) - 0.5) < 1e-6

    def test_extreme_positive_near_one(self) -> None:
        result = sigmoid(pd.Series([100.0]))
        assert float(result.iloc[0]) > 0.99

    def test_extreme_negative_near_zero(self) -> None:
        result = sigmoid(pd.Series([-100.0]))
        assert float(result.iloc[0]) < 0.01

    def test_symmetry(self) -> None:
        pos = sigmoid(pd.Series([2.0]))
        neg = sigmoid(pd.Series([-2.0]))
        assert abs(float(pos.iloc[0]) + float(neg.iloc[0]) - 1.0) < 1e-6


class TestWeightNormalization:
    """Verify portfolio weight properties from the optimizer."""

    def test_weights_sum_to_one(self) -> None:
        from src.allocation.optimizer import _optimize_single_regime

        np.random.seed(42)
        n_months = 36
        assets = ["SPY", "GLD", "IEF"]
        returns = pd.DataFrame(
            np.random.randn(n_months, len(assets)) * 0.02,
            columns=assets,
        )
        full_list = assets + ["cash"]
        result = _optimize_single_regime("Recovery", returns, assets, full_list)
        if result is not None:
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_non_negative(self) -> None:
        from src.allocation.optimizer import _optimize_single_regime

        np.random.seed(42)
        n_months = 36
        assets = ["SPY", "GLD", "IEF"]
        returns = pd.DataFrame(
            np.random.randn(n_months, len(assets)) * 0.02,
            columns=assets,
        )
        full_list = assets + ["cash"]
        result = _optimize_single_regime("Recovery", returns, assets, full_list)
        if result is not None:
            for asset, w in result.items():
                assert w >= -1e-8, f"Negative weight for {asset}: {w}"

    def test_cash_within_regime_bounds(self) -> None:
        from src.allocation.optimizer import _optimize_single_regime

        np.random.seed(42)
        n_months = 48
        assets = ["SPY", "GLD", "IEF"]
        returns = pd.DataFrame(
            np.random.randn(n_months, len(assets)) * 0.02,
            columns=assets,
        )
        full_list = assets + ["cash"]
        for regime in ("Recovery", "Contraction"):
            result = _optimize_single_regime(regime, returns, assets, full_list)
            if result is not None:
                lo, hi = REGIME_CASH.get(regime, (DEFAULT_MIN_CASH, DEFAULT_MAX_CASH))
                cash_w = result.get("cash", 0.0)
                assert cash_w >= lo - 1e-6, f"{regime}: cash {cash_w:.4f} < min {lo}"
                assert cash_w <= hi + 1e-6, f"{regime}: cash {cash_w:.4f} > max {hi}"


class TestCacheStaleness:
    """Verify cache staleness detection for prices and FRED."""

    def test_price_cache_missing_file_is_stale(self, tmp_path: Path) -> None:
        from src.data.market_ingestion import _is_stale

        assert _is_stale(tmp_path / "nonexistent.parquet") is True

    def test_price_cache_fresh_file_not_stale(self, tmp_path: Path) -> None:
        from src.data.market_ingestion import _is_stale

        f = tmp_path / "prices.parquet"
        f.write_text("dummy")
        assert _is_stale(f, max_age_hours=24) is False

    def test_price_cache_old_file_is_stale(self, tmp_path: Path) -> None:
        from src.data.market_ingestion import _is_stale

        f = tmp_path / "prices.parquet"
        f.write_text("dummy")
        old_time = time.time() - 25 * 3600
        import os

        os.utime(f, (old_time, old_time))
        assert _is_stale(f, max_age_hours=24) is True

    def test_fred_quarterly_staleness(self) -> None:
        from src.data.fred_cache import STALENESS_DAYS

        assert STALENESS_DAYS.get("GDP") == 30
        assert STALENESS_DAYS.get("M2V") == 30

    def test_fred_daily_staleness(self) -> None:
        from src.data.fred_cache import STALENESS_DAYS

        assert STALENESS_DAYS.get("DGS10") == 1
        assert STALENESS_DAYS.get("ICSA") == 1

    def test_fred_monthly_default_staleness(self) -> None:
        from src.data.fred_cache import DEFAULT_STALENESS_DAYS

        assert DEFAULT_STALENESS_DAYS == 7

    def test_fred_stale_detection(self, tmp_path: Path) -> None:
        from src.data.fred_cache import _is_stale

        with patch("src.data.fred_cache.MACRO_CACHE_DIR", tmp_path):
            with patch(
                "src.data.fred_cache._cache_path", return_value=tmp_path / "GDP.csv"
            ):
                assert _is_stale("GDP") is True

                f = tmp_path / "GDP.csv"
                f.write_text("date,value\n2024-01-01,100")
                assert _is_stale("GDP") is False

                old_time = time.time() - 31 * 86400
                import os

                os.utime(f, (old_time, old_time))
                assert _is_stale("GDP") is True


class TestRetryWithBackoff:
    """Verify retry_with_backoff behavior."""

    def test_succeeds_on_first_try(self) -> None:
        from src.utils.retry import retry_with_backoff

        result = retry_with_backoff(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self) -> None:
        from src.utils.retry import retry_with_backoff

        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network down")
            return "ok"

        result = retry_with_backoff(
            flaky,
            max_attempts=3,
            base_wait=0.01,
            exceptions=(ConnectionError,),
        )
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_exhausting_attempts(self) -> None:
        from src.utils.retry import retry_with_backoff

        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(ValueError("always fails")),
                max_attempts=2,
                base_wait=0.01,
                exceptions=(ValueError,),
            )
