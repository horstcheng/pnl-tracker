import pytest

from services.api.daily_close import validate_symbol, get_yfinance_tickers_to_try, normalize_ticker


class TestValidateSymbol:
    def test_valid_symbol_with_leading_zeros(self):
        """Symbols with leading zeros are preserved as-is."""
        assert validate_symbol("0050", "row 2") == "0050"
        assert validate_symbol("0052", "row 3") == "0052"
        assert validate_symbol("00687B", "row 4") == "00687B"
        assert validate_symbol("00953B", "row 5") == "00953B"
        assert validate_symbol("00713", "row 6") == "00713"
        assert validate_symbol("00965", "row 7") == "00965"
        assert validate_symbol("009805", "row 8") == "009805"
        assert validate_symbol("00972", "row 9") == "00972"
        assert validate_symbol("009812", "row 10") == "009812"

    def test_valid_symbol_alphanumeric(self):
        """Alphanumeric symbols are preserved."""
        assert validate_symbol("00983A", "row 2") == "00983A"
        assert validate_symbol("00984A", "row 3") == "00984A"
        assert validate_symbol("00988A", "row 4") == "00988A"

    def test_valid_symbol_numeric_no_leading_zero(self):
        """Numeric symbols without leading zeros are preserved."""
        assert validate_symbol("1519", "row 2") == "1519"
        assert validate_symbol("4979", "row 3") == "4979"
        assert validate_symbol("6442", "row 4") == "6442"
        assert validate_symbol("6789", "row 5") == "6789"

    def test_valid_symbol_us_stock(self):
        """US stock symbols are preserved."""
        assert validate_symbol("AAPL", "row 2") == "AAPL"
        assert validate_symbol("GOOGL", "row 3") == "GOOGL"
        assert validate_symbol("MSFT", "row 4") == "MSFT"

    def test_strips_whitespace(self):
        """Whitespace is stripped from symbols."""
        assert validate_symbol("  0050  ", "row 2") == "0050"
        assert validate_symbol(" AAPL ", "row 3") == "AAPL"
        assert validate_symbol("\t2330\n", "row 4") == "2330"

    def test_empty_symbol_raises_error(self):
        """Empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            validate_symbol("", "row 2")
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            validate_symbol("   ", "row 3")

    def test_non_string_raises_error(self):
        """Non-string symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol must be a string"):
            validate_symbol(123, "row 2")
        with pytest.raises(ValueError, match="Symbol must be a string"):
            validate_symbol(None, "row 3")


class TestGetYfinanceTickersToTry:
    def test_twd_symbol_without_dot_tries_tw_then_two(self):
        """TWD symbols without dot should try .TW then .TWO."""
        assert get_yfinance_tickers_to_try("0050", "TWD") == ["0050.TW", "0050.TWO"]
        assert get_yfinance_tickers_to_try("4979", "TWD") == ["4979.TW", "4979.TWO"]
        assert get_yfinance_tickers_to_try("00687B", "TWD") == ["00687B.TW", "00687B.TWO"]

    def test_twd_symbol_with_dot_used_as_is(self):
        """TWD symbols with dot are used as-is (already have suffix)."""
        assert get_yfinance_tickers_to_try("2330.TW", "TWD") == ["2330.TW"]
        assert get_yfinance_tickers_to_try("4979.TWO", "TWD") == ["4979.TWO"]

    def test_usd_symbol_used_as_is(self):
        """USD symbols are used as-is (no suffix needed)."""
        assert get_yfinance_tickers_to_try("AAPL", "USD") == ["AAPL"]
        assert get_yfinance_tickers_to_try("GOOGL", "USD") == ["GOOGL"]
        assert get_yfinance_tickers_to_try("MSFT", "USD") == ["MSFT"]

    def test_other_currency_used_as_is(self):
        """Other currency symbols are used as-is."""
        assert get_yfinance_tickers_to_try("VOO", "EUR") == ["VOO"]


class TestNormalizeTicker:
    """Tests for normalize_ticker() function."""

    def test_twd_symbol_with_letters_preserves_suffix(self):
        """
        TWD symbols with letters (e.g., 00983A) should:
        1) Keep leading zeros
        2) Keep trailing letters
        3) Append .TW suffix
        """
        assert normalize_ticker("00983A", "TWD") == "00983A.TW"
        assert normalize_ticker("00984A", "TWD") == "00984A.TW"
        assert normalize_ticker("00988A", "TWD") == "00988A.TW"
        assert normalize_ticker("00687B", "TWD") == "00687B.TW"
        assert normalize_ticker("00953B", "TWD") == "00953B.TW"

    def test_twd_symbol_with_leading_zeros_preserved(self):
        """TWD symbols with leading zeros should preserve them."""
        assert normalize_ticker("0050", "TWD") == "0050.TW"
        assert normalize_ticker("0052", "TWD") == "0052.TW"
        assert normalize_ticker("00713", "TWD") == "00713.TW"
        assert normalize_ticker("00965", "TWD") == "00965.TW"

    def test_twd_symbol_numeric_no_leading_zero(self):
        """TWD numeric symbols without leading zeros get .TW suffix."""
        assert normalize_ticker("2330", "TWD") == "2330.TW"
        assert normalize_ticker("1519", "TWD") == "1519.TW"
        assert normalize_ticker("4979", "TWD") == "4979.TW"

    def test_twd_symbol_with_existing_suffix_unchanged(self):
        """TWD symbols already having a dot (suffix) are returned as-is."""
        assert normalize_ticker("2330.TW", "TWD") == "2330.TW"
        assert normalize_ticker("4979.TWO", "TWD") == "4979.TWO"

    def test_usd_symbol_unchanged(self):
        """USD symbols are returned as-is (no suffix added)."""
        assert normalize_ticker("AAPL", "USD") == "AAPL"
        assert normalize_ticker("GOOGL", "USD") == "GOOGL"
        assert normalize_ticker("MSFT", "USD") == "MSFT"

    def test_symbol_treated_as_string(self):
        """
        Symbol is always treated as string, preserving leading zeros
        even if passed as other types (defensive handling).
        """
        # String input
        assert normalize_ticker("0050", "TWD") == "0050.TW"
        # Ensure str() conversion happens (symbol should be string but be defensive)
        assert normalize_ticker("  00983A  ", "TWD") == "00983A.TW"

    def test_whitespace_stripped(self):
        """Whitespace around symbol is stripped."""
        assert normalize_ticker("  2330  ", "TWD") == "2330.TW"
        assert normalize_ticker(" AAPL ", "USD") == "AAPL"
