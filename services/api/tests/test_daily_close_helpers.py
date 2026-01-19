import pytest

from services.api.daily_close import (
    get_yfinance_tickers_to_try,
    normalize_ticker,
    normalize_twd_ticker,
)


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("2330", "2330.TW"),
        ("0050", "0050.TW"),
        ("00983A", "00983A.TW"),
        ("2330.TW", "2330.TW"),  # Already has suffix
        (" 0050 ", "0050.TW"),   # With whitespace
        (2330, "2330.TW"),       # As integer
    ],
)
def test_normalize_twd_ticker(symbol, expected):
    """Test that TWD tickers are correctly normalized with .TW suffix."""
    assert normalize_twd_ticker(symbol) == expected


@pytest.mark.parametrize(
    "symbol, asset_ccy, expected",
    [
        # TWD assets
        ("2330", "TWD", "2330.TW"),
        ("0050", "TWD", "0050.TW"),
        ("00878", "twd", "00878.TW"), # Lowercase currency
        ("AAPL", "TWD", "AAPL.TW"),   # US stock traded in TWD context (less common)

        # USD assets
        ("AAPL", "USD", "AAPL"),
        ("GOOG", "USD", "GOOG"),
        ("VT", "usd", "VT"),     # Lowercase currency
        ("BRK-B", "USD", "BRK-B"),

        # Edge cases
        (" 2330 ", "TWD", "2330.TW"), # Whitespace
        ("2330.TW", "TWD", "2330.TW"), # Already normalized
        ("AAPL.O", "USD", "AAPL.O"),   # Already has suffix
    ],
)
def test_normalize_ticker(symbol, asset_ccy, expected):
    """Test universal ticker normalization for different currencies."""
    assert normalize_ticker(symbol, asset_ccy) == expected


@pytest.mark.parametrize(
    "symbol, asset_ccy, expected",
    [
        # TWD listed stock (.TW)
        ("2330", "TWD", ["2330.TW", "2330.TWO"]),
        # TWD OTC stock (.TWO)
        ("6446", "TWD", ["6446.TW", "6446.TWO"]),
        # TWD with existing suffix
        ("2330.TW", "TWD", ["2330.TW"]),
        # TWD with existing .TWO suffix
        ("6446.TWO", "TWD", ["6446.TWO"]),
        # USD stock
        ("AAPL", "USD", ["AAPL"]),
        # USD stock with suffix
        ("BRK-B", "USD", ["BRK-B"]),
    ],
)
def test_get_yfinance_tickers_to_try(symbol, asset_ccy, expected):
    """Test the list of tickers to try for yfinance lookup."""
    assert get_yfinance_tickers_to_try(symbol, asset_ccy) == expected
