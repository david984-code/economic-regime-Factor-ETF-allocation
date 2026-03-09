"""Data ingestion modules."""

from src.data.fred_ingestion import (
    fetch_fred_core,
    fetch_fred_core_cached,
    fetch_fred_optional,
    fetch_fred_optional_cached,
)
from src.data.market_ingestion import fetch_prices, fetch_monthly_returns
from src.data.pipeline_data import PipelineData

__all__ = [
    "fetch_fred_core",
    "fetch_fred_core_cached",
    "fetch_fred_optional",
    "fetch_fred_optional_cached",
    "fetch_prices",
    "fetch_monthly_returns",
    "PipelineData",
]
