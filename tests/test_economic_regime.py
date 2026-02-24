"""Unit tests for EconomicRegimeClassifier."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.economic_regime import EconomicRegimeClassifier


class TestEconomicRegimeClassifier:
    """Test suite for EconomicRegimeClassifier."""

    @pytest.fixture
    def classifier(self) -> EconomicRegimeClassifier:
        """Create a classifier instance for testing."""
        return EconomicRegimeClassifier(api_key="test_api_key")

    def test_initialization(self, classifier: EconomicRegimeClassifier) -> None:
        """Test classifier initializes correctly."""
        assert classifier.fred is not None
        assert classifier.end_date == datetime.today().strftime("%Y-%m-%d")

    def test_to_month_end_conversion(self) -> None:
        """Test month-end timestamp conversion."""
        dates = pd.date_range("2020-01-15", periods=3, freq="D")
        series = pd.Series([1, 2, 3], index=dates)

        result = EconomicRegimeClassifier.to_month_end(series)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert all(result.index == result.index.to_period("M").to_timestamp("M"))

    def test_rolling_z_score_calculation(self) -> None:
        """Test rolling z-score normalization."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)

        result = EconomicRegimeClassifier.rolling_z_score(series, window=10, min_periods=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        assert not result.dropna().isnull().all()

    def test_sigmoid_transformation(self) -> None:
        """Test sigmoid function output range."""
        test_values = pd.Series([-10, -1, 0, 1, 10])

        result = EconomicRegimeClassifier.sigmoid(test_values)

        assert (result >= 0).all() and (result <= 1).all()
        assert result.iloc[2] == pytest.approx(0.5, abs=0.01)

    def test_classify_regime_recovery(self) -> None:
        """Test regime classification: Recovery (GDP+, Inflation-)."""
        row = pd.Series({"gdp_z": 1.5, "infl_z": -0.5})

        result = EconomicRegimeClassifier.classify_regime(row)

        assert result == "Recovery"

    def test_classify_regime_overheating(self) -> None:
        """Test regime classification: Overheating (GDP+, Inflation+)."""
        row = pd.Series({"gdp_z": 1.2, "infl_z": 0.8})

        result = EconomicRegimeClassifier.classify_regime(row)

        assert result == "Overheating"

    def test_classify_regime_stagflation(self) -> None:
        """Test regime classification: Stagflation (GDP-, Inflation+)."""
        row = pd.Series({"gdp_z": -1.0, "infl_z": 1.5})

        result = EconomicRegimeClassifier.classify_regime(row)

        assert result == "Stagflation"

    def test_classify_regime_contraction(self) -> None:
        """Test regime classification: Contraction (GDP-, Inflation-)."""
        row = pd.Series({"gdp_z": -0.8, "infl_z": -1.2})

        result = EconomicRegimeClassifier.classify_regime(row)

        assert result == "Contraction"

    def test_classify_regime_unknown_with_nan(self) -> None:
        """Test regime classification: Unknown when data is missing."""
        row = pd.Series({"gdp_z": np.nan, "infl_z": 1.0})

        result = EconomicRegimeClassifier.classify_regime(row)

        assert result == "Unknown"

    def test_classify_regimes_dataframe(self, classifier: EconomicRegimeClassifier) -> None:
        """Test regime classification on full dataframe."""
        df = pd.DataFrame(
            {
                "gdp_z": [1.5, -0.5, 1.0, -1.0, np.nan],
                "infl_z": [-0.5, 1.0, 1.0, -1.0, 0.5],
            }
        )

        result = classifier.classify_regimes(df)

        expected_regimes = ["Recovery", "Stagflation", "Overheating", "Contraction", "Unknown"]
        assert list(result["regime"]) == expected_regimes

    def test_calculate_macro_score(self, classifier: EconomicRegimeClassifier) -> None:
        """Test macro score calculation and risk_on transformation."""
        df = pd.DataFrame(
            {
                "gdp_z": [1.0],
                "infl_z": [-0.5],
                "m2_z": [0.2],
                "vel_z": [0.3],
                "yield_level_z": [0.1],
            }
        )

        result = classifier.calculate_macro_score(df)

        assert "macro_score" in result.columns
        assert "risk_on" in result.columns
        assert 0 <= result["risk_on"].iloc[0] <= 1

    @patch("src.economic_regime.Path.mkdir")
    def test_save_results_success(
        self, mock_mkdir: Mock, classifier: EconomicRegimeClassifier, tmp_path: Path
    ) -> None:
        """Test successful save without retries."""
        df = pd.DataFrame(
            {
                "gdp_z": [1.0],
                "infl_z": [-0.5],
                "m2_z": [0.2],
                "vel_z": [0.3],
                "yield_level_z": [0.1],
                "macro_score": [2.1],
                "risk_on": [0.6],
                "regime": ["Recovery"],
            }
        )

        classifier.save_results(df, tmp_path)

        saved_file = tmp_path / "regime_labels_expanded.csv"
        assert saved_file.exists()
        saved_df = pd.read_csv(saved_file)
        assert "regime" in saved_df.columns
        assert saved_df["regime"].iloc[0] == "Recovery"

    @patch("time.sleep")
    def test_save_results_retry_succeeds(
        self, mock_sleep: Mock, classifier: EconomicRegimeClassifier, tmp_path: Path
    ) -> None:
        """Test save succeeds after retry with exponential backoff."""
        df = pd.DataFrame(
            {
                "gdp_z": [1.0],
                "infl_z": [-0.5],
                "m2_z": [0.2],
                "vel_z": [0.3],
                "yield_level_z": [0.1],
                "macro_score": [2.1],
                "risk_on": [0.6],
                "regime": ["Recovery"],
            }
        )

        call_count = 0

        def mock_to_csv(path: Path, **kwargs: dict) -> None:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise PermissionError("File locked")

        with patch.object(pd.DataFrame, "to_csv", side_effect=mock_to_csv):
            classifier.save_results(df, tmp_path)

        assert call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)

    def test_save_results_retry_exhausted(
        self, classifier: EconomicRegimeClassifier, tmp_path: Path
    ) -> None:
        """Test save fails after exhausting all retries."""
        df = pd.DataFrame(
            {
                "gdp_z": [1.0],
                "infl_z": [-0.5],
                "m2_z": [0.2],
                "vel_z": [0.3],
                "yield_level_z": [0.1],
                "macro_score": [2.1],
                "risk_on": [0.6],
                "regime": ["Recovery"],
            }
        )

        with patch.object(
            pd.DataFrame, "to_csv", side_effect=PermissionError("Permanently locked")
        ):
            with pytest.raises(PermissionError, match="Failed to save after 4 attempts"):
                classifier.save_results(df, tmp_path)

    def test_build_dataframe_structure(self, classifier: EconomicRegimeClassifier) -> None:
        """Test dataframe construction from economic indicators."""
        dates = pd.date_range("2020-01-01", periods=5, freq="ME")
        gdp = pd.Series([100, 102, 103, 105, 104], index=dates)
        cpi = pd.Series([250, 252, 255, 258, 260], index=dates)
        yield_10y = pd.Series([2.0, 2.1, 2.2, 2.3, 2.4], index=dates)
        yield_3m = pd.Series([1.5, 1.6, 1.7, 1.8, 1.9], index=dates)
        m2 = pd.Series([15000, 15100, 15200, 15300, 15400], index=dates)
        velocity = pd.Series([1.4, 1.41, 1.42, 1.43, 1.44], index=dates)

        result = classifier.build_dataframe(gdp, cpi, yield_10y, yield_3m, m2, velocity)

        expected_cols = ["gdp", "cpi", "gdp_mom", "cpi_mom", "m2_mom", "vel_mom", "yield_curve"]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == 5

    def test_add_z_scores(self, classifier: EconomicRegimeClassifier) -> None:
        """Test z-score calculation adds correct columns."""
        df = pd.DataFrame(
            {
                "gdp_mom": np.random.randn(100) * 0.02,
                "cpi_mom": np.random.randn(100) * 0.01,
                "m2_mom": np.random.randn(100) * 0.03,
                "vel_mom": np.random.randn(100) * 0.02,
                "yield_curve": np.random.randn(100) * 0.5,
            }
        )

        result = classifier.add_z_scores(df)

        z_score_cols = ["gdp_z", "infl_z", "m2_z", "vel_z", "yield_level_z"]
        assert all(col in result.columns for col in z_score_cols)
