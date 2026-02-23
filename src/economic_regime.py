"""Economic Regime Classification Module.

This module fetches macroeconomic data from FRED and classifies economic regimes
based on GDP growth, inflation, money supply, velocity, and yield curve signals.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


class EconomicRegimeClassifier:
    """Classifies economic regimes based on macroeconomic indicators."""

    def __init__(self, api_key: str) -> None:
        """Initialize the classifier with FRED API credentials.

        Args:
            api_key: FRED API key for data access
        """
        self.fred = Fred(api_key=api_key)
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def fetch_fred_data(self) -> tuple[pd.Series, ...]:
        """Fetch macroeconomic data from FRED.

        Returns:
            Tuple of (gdp, cpi, yield_10y, yield_3m, m2, velocity) series
        """
        gdp = self.fred.get_series("GDP", observation_end=self.end_date)
        cpi = self.fred.get_series("CPIAUCSL", observation_end=self.end_date)
        yield_10y = self.fred.get_series("DGS10", observation_end=self.end_date)
        yield_3m = self.fred.get_series("DGS3MO", observation_end=self.end_date)
        m2 = self.fred.get_series("M2SL", observation_end=self.end_date)
        velocity = self.fred.get_series("M2V", observation_end=self.end_date)

        # Convert indices to datetime
        for series in (gdp, cpi, yield_10y, yield_3m, m2, velocity):
            series.index = pd.to_datetime(series.index)

        return gdp, cpi, yield_10y, yield_3m, m2, velocity

    def print_data_summary(
        self,
        gdp: pd.Series,
        cpi: pd.Series,
        yield_10y: pd.Series,
        yield_3m: pd.Series,
        m2: pd.Series,
        velocity: pd.Series,
    ) -> None:
        """Print summary of fetched data dates."""
        print("\n[DATA] Latest available dates pulled from FRED:")
        print(f"GDP: {gdp.index.max()}")
        print(f"CPI: {cpi.index.max()}")
        print(f"10Y: {yield_10y.index.max()}")
        print(f"3M: {yield_3m.index.max()}")
        print(f"M2: {m2.index.max()}")
        print(f"Velocity (M2V): {velocity.index.max()}")

    @staticmethod
    def to_month_end(series: pd.Series) -> pd.Series:
        """Convert series to month-end timestamps.

        Args:
            series: Input time series

        Returns:
            Series with month-end index
        """
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        s.index = s.index.to_period("M").to_timestamp("M")
        s = s[~s.index.duplicated(keep="last")]
        return s  # type: ignore[no-any-return]

    def resample_to_monthly(
        self,
        gdp: pd.Series,
        cpi: pd.Series,
        yield_10y: pd.Series,
        yield_3m: pd.Series,
        m2: pd.Series,
        velocity: pd.Series,
    ) -> tuple[pd.Series, ...]:
        """Resample all series to monthly frequency."""
        gdp = gdp.resample("ME").ffill()
        yield_10y = yield_10y.resample("ME").ffill()
        yield_3m = yield_3m.resample("ME").ffill()
        m2 = m2.resample("ME").ffill()
        velocity = velocity.resample("ME").ffill()
        cpi.index = cpi.index + pd.offsets.MonthEnd(0)

        # Normalize to month-end
        gdp = self.to_month_end(gdp)
        cpi = self.to_month_end(cpi)
        m2 = self.to_month_end(m2)
        velocity = self.to_month_end(velocity)
        yield_10y = self.to_month_end(yield_10y)
        yield_3m = self.to_month_end(yield_3m)

        return gdp, cpi, yield_10y, yield_3m, m2, velocity

    @staticmethod
    def rolling_z_score(series: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
        """Calculate rolling z-score.

        Args:
            series: Input series
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Rolling z-scores
        """
        mean = series.rolling(window=window, min_periods=min_periods).mean()
        std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
        return (series - mean) / std.replace(0, np.nan)  # type: ignore[no-any-return]

    @staticmethod
    def sigmoid(series: pd.Series) -> pd.Series:
        """Apply sigmoid transformation."""
        return 1.0 / (1.0 + np.exp(-series))  # type: ignore[no-any-return]

    def build_dataframe(
        self,
        gdp: pd.Series,
        cpi: pd.Series,
        yield_10y: pd.Series,
        yield_3m: pd.Series,
        m2: pd.Series,
        velocity: pd.Series,
    ) -> pd.DataFrame:
        """Build dataframe with raw indicators and month-over-month changes."""
        gdp_mom = gdp.pct_change()
        cpi_mom = cpi.pct_change()
        m2_mom = m2.pct_change()
        vel_mom = velocity.pct_change()
        yield_curve = (yield_10y - yield_3m).rename("yield_curve")

        return pd.DataFrame(
            {
                "gdp": gdp,
                "cpi": cpi,
                "gdp_mom": gdp_mom,
                "cpi_mom": cpi_mom,
                "m2_mom": m2_mom,
                "vel_mom": vel_mom,
                "yield_curve": yield_curve,
            }
        )

    def add_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling z-scores to the dataframe."""
        df["gdp_z"] = self.rolling_z_score(df["gdp_mom"])
        df["infl_z"] = self.rolling_z_score(df["cpi_mom"])
        df["m2_z"] = self.rolling_z_score(df["m2_mom"])
        df["vel_z"] = self.rolling_z_score(df["vel_mom"])
        df["yield_level_z"] = self.rolling_z_score(df["yield_curve"])
        return df

    def calculate_macro_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate macro risk score from z-scores."""
        df["macro_score"] = (
            df["gdp_z"] + df["vel_z"] - df["infl_z"] + df["m2_z"] + df["yield_level_z"]
        )
        df["risk_on"] = self.sigmoid(df["macro_score"] * 0.25)
        return df

    @staticmethod
    def classify_regime(row: pd.Series) -> str:
        """Classify regime based on GDP and inflation z-scores."""
        gdp_z = row["gdp_z"]
        infl_z = row["infl_z"]

        if pd.isna(gdp_z) or pd.isna(infl_z):
            return "Unknown"

        if gdp_z > 0 and infl_z <= 0:
            return "Recovery"
        elif gdp_z > 0 and infl_z > 0:
            return "Overheating"
        elif gdp_z <= 0 and infl_z > 0:
            return "Stagflation"
        else:
            return "Contraction"

    def classify_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply regime classification to dataframe."""
        df["regime"] = df.apply(self.classify_regime, axis=1)
        return df

    def save_results(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Save regime classification results."""
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
        df_reset = df.reset_index().rename(columns={"index": "date"})
        regime_df = df_reset[out_cols].dropna(subset=["regime"]).copy()

        save_path = output_dir / "regime_labels_expanded.csv"

        try:
            regime_df.to_csv(save_path, index=False)
            print(f"\n[SUCCESS] Saved: {save_path}")
        except PermissionError:
            fallback = output_dir / f"regime_labels_expanded_{datetime.now():%Y%m%d_%H%M%S}.csv"
            regime_df.to_csv(fallback, index=False)
            print(f"\n[WARNING] File was locked. Saved instead to: {fallback}")

    def print_recent_regimes(self, df: pd.DataFrame) -> None:
        """Print recent regime classifications."""
        df_reset = df.reset_index().rename(columns={"index": "date"})
        classified = df_reset.dropna(subset=["regime"])
        recent = classified[classified["date"] >= pd.to_datetime("2020-01-01")]

        print("\n[REGIMES] Regime Classification from 2020 to Latest:")
        print(recent[["date", "regime", "risk_on"]].tail(24))

        latest = recent.iloc[-1]
        print("\n[CURRENT] Latest Classified Month:")
        print(f"Date: {latest['date']:%Y-%m-%d}")
        print(f"Regime: {latest['regime']}")
        print(f"Risk-on score (0..1): {latest['risk_on']:.3f}")
        print("Suggested Allocation: (generated in optimizer.py using factor ETFs)")

        print("\n[STATS] Regime Count Since 2020:")
        print(recent["regime"].value_counts())

    def run(self) -> None:
        """Run the complete regime classification pipeline."""
        # Fetch data
        gdp, cpi, yield_10y, yield_3m, m2, velocity = self.fetch_fred_data()
        self.print_data_summary(gdp, cpi, yield_10y, yield_3m, m2, velocity)

        # Resample to monthly
        gdp, cpi, yield_10y, yield_3m, m2, velocity = self.resample_to_monthly(
            gdp, cpi, yield_10y, yield_3m, m2, velocity
        )

        # Build dataframe and calculate z-scores
        df = self.build_dataframe(gdp, cpi, yield_10y, yield_3m, m2, velocity)
        df = self.add_z_scores(df)

        # Calculate scores
        df = self.calculate_macro_score(df)

        # Classify regimes
        df = self.classify_regimes(df)

        # Print results
        self.print_recent_regimes(df)

        # Save results
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        self.save_results(df, output_dir)


def main() -> None:
    """Main entry point for economic regime classification."""
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ FRED_API_KEY not found!\n"
            "Please create a .env file in the project root with:\n"
            "FRED_API_KEY=your_api_key_here\n\n"
            "Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    # Run classification
    classifier = EconomicRegimeClassifier(api_key)
    classifier.run()


if __name__ == "__main__":
    main()
