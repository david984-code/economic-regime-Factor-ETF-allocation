"""Tests for FRED key resolution and ticker CLI parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import fred_key
from src.utils import ticker_universe as tickers_mod


def test_parse_tickers_arg_basic() -> None:
    assert tickers_mod.parse_tickers_arg("spy,gld") == ["SPY", "GLD"]


def test_parse_tickers_arg_empty() -> None:
    assert tickers_mod.parse_tickers_arg("") is None
    assert tickers_mod.parse_tickers_arg("  ") is None


def test_resolve_tickers_prefers_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPELINE_TICKERS", raising=False)
    out = tickers_mod.resolve_tickers(["XLE", "XLF"])
    assert out == ["XLE", "XLF"]


def test_resolve_tickers_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPELINE_TICKERS", raising=False)
    with pytest.raises(ValueError, match="Invalid"):
        tickers_mod.resolve_tickers(["123"])


_TEST_FRED_KEY = "0123456789abcdef0123456789abcdef"


def test_get_fred_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.utils.fred_key.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("FRED_API_KEY", _TEST_FRED_KEY)
    monkeypatch.delenv("FRED_API_KEY_FILE", raising=False)
    assert fred_key.get_fred_api_key() == _TEST_FRED_KEY


def test_get_fred_api_key_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.utils.fred_key.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    p = tmp_path / "k.txt"
    p.write_text(f"{_TEST_FRED_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("FRED_API_KEY_FILE", str(p))
    assert fred_key.get_fred_api_key() == _TEST_FRED_KEY


def test_get_fred_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.utils.fred_key.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY_FILE", raising=False)
    assert fred_key.get_fred_api_key() is None


def test_fetch_prices_mocked_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import market_ingestion

    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {("Adj Close", "SPY"): [100.0, 101.0, 102.0]},
        index=idx,
    )
    raw.columns = pd.MultiIndex.from_tuples([("Adj Close", "SPY")])

    def fake_download(syms, start, end, progress=False, auto_adjust=False):
        assert list(syms) == ["SPY"]
        return raw

    monkeypatch.setattr(market_ingestion.yf, "download", fake_download)

    out = market_ingestion.fetch_prices(
        tickers=["SPY"], start="2024-01-01", end="2024-01-10"
    )
    assert list(out.columns) == ["SPY"]
    assert len(out) == 3
