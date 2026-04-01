"""Tests for src/research/sensitivity.py — parameter sweep scanner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.research.sensitivity import (
    BASELINE,
    DEFAULT_GRID,
    SensitivityReport,
    SweepResult,
    _extract_overall_metrics,
    _save_summary_csv,
    build_param_combos,
)


class TestBuildParamCombos:
    def test_default_grid_count(self) -> None:
        combos = build_param_combos()
        expected = 1
        for v in DEFAULT_GRID.values():
            expected *= len(v)
        assert len(combos) == expected

    def test_all_combos_contain_baseline_keys(self) -> None:
        combos = build_param_combos()
        for combo in combos:
            for key in BASELINE:
                assert key in combo

    def test_single_param_grid(self) -> None:
        grid = {"sigmoid_scale": [0.15, 0.25, 0.50]}
        combos = build_param_combos(grid)
        assert len(combos) == 3
        # Non-swept params should be at baseline
        for combo in combos:
            assert combo["market_lookback_months"] == BASELINE["market_lookback_months"]
            assert combo["tolerance"] == BASELINE["tolerance"]

    def test_custom_grid_values_present(self) -> None:
        grid = {"trend_filter_type": ["none", "200dma"]}
        combos = build_param_combos(grid)
        filter_values = {c["trend_filter_type"] for c in combos}
        assert filter_values == {"none", "200dma"}

    def test_empty_grid_uses_default(self) -> None:
        combos = build_param_combos({})
        # Empty dict falls through to DEFAULT_GRID
        expected = 1
        for v in DEFAULT_GRID.values():
            expected *= len(v)
        assert len(combos) == expected


class TestExtractOverallMetrics:
    def test_extracts_overall_row(self) -> None:
        data = {
            "segment": ["OVERALL", 0, 1],
            "Strategy_CAGR": [0.075, 0.08, 0.07],
            "Strategy_Sharpe": [0.51, 0.55, 0.47],
            "Strategy_MaxDD": [-0.075, -0.05, -0.10],
            "Strategy_Vol": [0.12, 0.11, 0.13],
            "Strategy_Turnover": [0.90, 0.85, 0.95],
            "Strategy_HitRate": [0.52, 0.53, 0.51],
        }
        df = pd.DataFrame(data)
        metrics = _extract_overall_metrics(df)
        assert metrics["Strategy_CAGR"] == pytest.approx(0.075)
        assert metrics["Strategy_Sharpe"] == pytest.approx(0.51)

    def test_missing_overall_returns_nan(self) -> None:
        df = pd.DataFrame({"segment": [0, 1], "Strategy_CAGR": [0.08, 0.07]})
        metrics = _extract_overall_metrics(df)
        assert np.isnan(metrics["Strategy_CAGR"])

    def test_missing_columns_return_nan(self) -> None:
        df = pd.DataFrame({"segment": ["OVERALL"], "Strategy_CAGR": [0.075]})
        metrics = _extract_overall_metrics(df)
        assert metrics["Strategy_CAGR"] == pytest.approx(0.075)
        assert np.isnan(metrics["Strategy_Sharpe"])


class TestSaveSummaryCsv:
    def test_csv_created(self, tmp_path: pd.Timestamp) -> None:
        report = SensitivityReport(
            sweep_id="test_001",
            created_at="2026-04-01T00:00:00",
            grid={"sigmoid_scale": [0.15, 0.25]},
            baseline=dict(BASELINE),
            results=[
                SweepResult(
                    params={**BASELINE, "sigmoid_scale": 0.15},
                    metrics={"Strategy_Sharpe": 0.48, "Strategy_CAGR": 0.06},
                    run_id="r1",
                    elapsed_sec=10.0,
                    n_segments=5,
                ),
                SweepResult(
                    params={**BASELINE, "sigmoid_scale": 0.25},
                    metrics={"Strategy_Sharpe": 0.51, "Strategy_CAGR": 0.075},
                    run_id="r2",
                    elapsed_sec=10.0,
                    n_segments=5,
                ),
            ],
        )
        baseline_metrics = {"Strategy_Sharpe": 0.51, "Strategy_CAGR": 0.075}

        with patch("src.research.sensitivity.OUTPUTS_DIR", tmp_path):
            _save_summary_csv(report, baseline_metrics)

        csv_path = tmp_path / "sensitivity_test_001.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "Sharpe_Delta" in df.columns
        # First row should be the better Sharpe (sorted descending)
        assert df.iloc[0]["Strategy_Sharpe"] >= df.iloc[1]["Strategy_Sharpe"]


class TestSweepResultDataclass:
    def test_sweep_result_fields(self) -> None:
        r = SweepResult(
            params={"sigmoid_scale": 0.25},
            metrics={"Strategy_Sharpe": 0.51},
            run_id="abc",
            elapsed_sec=5.0,
            n_segments=10,
        )
        assert r.params["sigmoid_scale"] == 0.25
        assert r.metrics["Strategy_Sharpe"] == 0.51
        assert r.n_segments == 10
