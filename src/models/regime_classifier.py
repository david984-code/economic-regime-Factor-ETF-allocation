"""Rule-based economic regime classification."""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from src.config import OUTPUTS_DIR, get_end_date
from src.data.fred_ingestion import fetch_fred_core, fetch_fred_optional
from src.features.macro_features import (
    add_z_scores,
    build_macro_dataframe,
    calculate_macro_score,
    resample_high_freq_to_monthly,
    resample_to_monthly,
)
from src.features.transforms import rolling_z_score, sigmoid, to_month_end
from src.utils.database import Database
from src.utils.retry import retry_on_permission_error

logger = logging.getLogger(__name__)


def classify_regime_row(row: pd.Series) -> str:
    """Original row-wise implementation. Used by run_parity_check for validation.
    Can be removed after parity is verified; kept for regression testing."""
    gdp_z = row["gdp_z"]
    infl_z = row["infl_z"]
    if pd.isna(gdp_z) or pd.isna(infl_z):
        return "Unknown"
    if gdp_z > 0 and infl_z <= 0:
        return "Recovery"
    if gdp_z > 0 and infl_z > 0:
        return "Overheating"
    if gdp_z <= 0 and infl_z > 0:
        return "Stagflation"
    return "Contraction"


def classify_regime(row: pd.Series) -> str:
    """Classify regime from GDP and inflation z-scores. For single-row use (tests)."""
    return classify_regime_row(row)


def classify_regimes_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized regime classification using np.select.

    Logic (first match wins):
      1. NaN in gdp_z or infl_z -> Unknown
      2. gdp_z > 0 and infl_z <= 0 -> Recovery
      3. gdp_z > 0 and infl_z > 0 -> Overheating
      4. gdp_z <= 0 and infl_z > 0 -> Stagflation
      5. else -> Contraction
    """
    gdp_z = np.asarray(df["gdp_z"], dtype=float)
    infl_z = np.asarray(df["infl_z"], dtype=float)
    nan_mask = np.isnan(gdp_z) | np.isnan(infl_z)
    cond_recovery = (gdp_z > 0) & (infl_z <= 0)
    cond_overheating = (gdp_z > 0) & (infl_z > 0)
    cond_stagflation = (gdp_z <= 0) & (infl_z > 0)
    result = np.select(
        [nan_mask, cond_recovery, cond_overheating, cond_stagflation],
        ["Unknown", "Recovery", "Overheating", "Stagflation"],
        default="Contraction",
    )
    return pd.Series(result, index=df.index)


class RegimeClassifier:
    """Orchestrates data fetch, feature engineering, and regime classification."""

    # Static methods for backward compatibility with tests
    to_month_end = staticmethod(to_month_end)
    rolling_z_score = staticmethod(rolling_z_score)
    sigmoid = staticmethod(sigmoid)
    classify_regime = staticmethod(classify_regime)

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
        self.end_date = get_end_date()

    def build_dataframe(
        self,
        gdp: pd.Series,
        cpi: pd.Series,
        yield_10y: pd.Series,
        yield_3m: pd.Series,
        m2: pd.Series,
        velocity: pd.Series,
        hf_monthly: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """Build macro dataframe (backward compat)."""
        return build_macro_dataframe(
            gdp, cpi, yield_10y, yield_3m, m2, velocity, hf_monthly=hf_monthly
        )

    def add_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add z-scores (backward compat)."""
        return add_z_scores(df)

    def calculate_macro_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate macro score (backward compat)."""
        return calculate_macro_score(df)

    def classify_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply regime classification (vectorized)."""
        df = df.copy()
        df["regime"] = classify_regimes_vectorized(df)
        return df

    def save_results(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Save results (backward compat for tests)."""
        self._save_results(df, output_dir)

    def run(self, use_local_cache: bool = True) -> None:
        """Run full pipeline and save. See run_and_return_df for parity testing."""
        df = self.run_and_return_df(use_local_cache=use_local_cache)
        self._save_results(df, OUTPUTS_DIR)
        self._log_recent(df)

    def run_and_return_df(self, use_local_cache: bool = True) -> pd.DataFrame:
        """Run full pipeline: fetch -> features -> classify -> save.

        Args:
            use_local_cache: If True, use local macro cache and incremental FRED fetch.
        """
        end_date = get_end_date()
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # --- FRED retrieval ---
        t0 = time.perf_counter()
        if use_local_cache:
            from src.data.fred_ingestion import fetch_fred_core_cached, fetch_fred_optional_cached
            gdp, cpi, yield_10y, yield_3m, m2, velocity = fetch_fred_core_cached(
                self.api_key, end_date=end_date
            )
            hf_raw = fetch_fred_optional_cached(self.api_key, end_date=end_date)
        else:
            gdp, cpi, yield_10y, yield_3m, m2, velocity = fetch_fred_core(
                self.api_key, end_date=end_date
            )
            hf_raw = fetch_fred_optional(self.api_key, end_date=end_date)
        timings["fred_retrieval"] = time.perf_counter() - t0

        # --- Merge / resample ---
        t0 = time.perf_counter()
        hf_monthly = resample_high_freq_to_monthly(hf_raw) if hf_raw else None
        gdp, cpi, yield_10y, yield_3m, m2, velocity = resample_to_monthly(
            gdp, cpi, yield_10y, yield_3m, m2, velocity
        )
        timings["merge_resample"] = time.perf_counter() - t0

        # --- Feature calculation ---
        t0 = time.perf_counter()
        df = build_macro_dataframe(
            gdp, cpi, yield_10y, yield_3m, m2, velocity, hf_monthly=hf_monthly
        )
        df = add_z_scores(df)
        df = calculate_macro_score(df)
        timings["feature_calculation"] = time.perf_counter() - t0

        # --- Regime labeling ---
        t0 = time.perf_counter()
        df["regime"] = classify_regimes_vectorized(df)
        timings["regime_labeling"] = time.perf_counter() - t0

        elapsed = time.perf_counter() - t_total
        logger.info(
            "[REGIME] Sub-step timing: fred=%.2fs merge=%.2fs features=%.2fs label=%.2fs total=%.2fs",
            timings["fred_retrieval"],
            timings["merge_resample"],
            timings["feature_calculation"],
            timings["regime_labeling"],
            elapsed,
        )
        return df

    def _save_results(self, df: pd.DataFrame, output_dir: Path | None = None) -> None:
        """Save regime labels to database and CSV."""
        out_dir = output_dir or OUTPUTS_DIR
        out_cols = [
            "date",
            "regime",
            "risk_on",
            "macro_score",
            "gdp_z",
            "infl_z",
            "m2_z",
            "vel_z",
            "yield_level_z",
        ]
        for c in ("pmi_z", "claims_z", "hy_spread_z"):
            if c in df.columns:
                out_cols.append(c)
        df_reset = df.reset_index().rename(columns={"index": "date"})
        avail = [c for c in out_cols if c in df_reset.columns]
        regime_df = df_reset[avail].dropna(subset=["regime"]).copy()

        db = Database()
        regime_df_for_db = regime_df[["date", "regime", "risk_on"]].copy()
        regime_df_for_db.set_index("date", inplace=True)
        db.save_regime_labels(regime_df_for_db)
        db.close()

        out_dir.mkdir(exist_ok=True)
        save_path = out_dir / "regime_labels_expanded.csv"

        def _write_csv() -> None:
            regime_df.to_csv(save_path, index=False)
            logger.info("Saved CSV backup: %s", save_path)

        try:
            retry_on_permission_error(_write_csv, logger=logger)
        except PermissionError:
            logger.warning("Could not save CSV backup; data is in database.")

    def _log_recent(self, df: pd.DataFrame) -> None:
        """Log recent regime classifications."""
        df_reset = df.reset_index().rename(columns={"index": "date"})
        classified = df_reset.dropna(subset=["regime"])
        recent = classified[classified["date"] >= pd.to_datetime("2020-01-01")]
        latest = recent.iloc[-1]
        logger.info(
            "Latest regime: %s | risk_on=%.3f",
            latest["regime"],
            latest["risk_on"],
        )


def run_parity_check(df: pd.DataFrame) -> tuple[bool, dict]:
    """Compare old (apply) vs new (vectorized) regime classification.

    Returns:
        (all_match, report_dict) with keys: match_count, total, mismatches, old_time_sec, new_time_sec
    """
    if "gdp_z" not in df.columns or "infl_z" not in df.columns:
        raise ValueError("DataFrame must have gdp_z and infl_z columns")

    # Old: row-wise apply
    t0 = time.perf_counter()
    old_regimes = df.apply(classify_regime_row, axis=1)
    old_time = time.perf_counter() - t0

    # New: vectorized
    t0 = time.perf_counter()
    new_regimes = classify_regimes_vectorized(df)
    new_time = time.perf_counter() - t0

    match = old_regimes == new_regimes
    match_count = int(match.sum())
    total = len(df)
    mismatches = df[~match].copy() if not match.all() else pd.DataFrame()
    if not mismatches.empty:
        mismatches["old_regime"] = old_regimes[~match].values
        mismatches["new_regime"] = new_regimes[~match].values

    report = {
        "match_count": match_count,
        "total": total,
        "all_match": match_count == total,
        "mismatches": mismatches,
        "old_time_sec": old_time,
        "new_time_sec": new_time,
        "speedup": old_time / new_time if new_time > 0 else float("inf"),
    }
    return report["all_match"], report


def main() -> None:
    """Entry point for regime classification."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. Create .env with FRED_API_KEY=your_key"
        )
    classifier = RegimeClassifier(api_key)
    classifier.run()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
