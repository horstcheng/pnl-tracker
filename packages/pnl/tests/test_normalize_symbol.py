import pytest

from services.api.daily_close import validate_symbol


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
