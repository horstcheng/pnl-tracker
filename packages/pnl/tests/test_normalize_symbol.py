import pytest

from services.api.daily_close import normalize_symbol, clear_symbol_normalizations


class TestNormalizeSymbol:
    def setup_method(self):
        clear_symbol_normalizations()

    def test_numeric_1_digit_pads_to_4(self):
        assert normalize_symbol(5) == "0005"
        assert normalize_symbol("5") == "0005"

    def test_numeric_2_digits_pads_to_4(self):
        assert normalize_symbol(50) == "0050"
        assert normalize_symbol("50") == "0050"

    def test_numeric_3_digits_pads_to_4(self):
        assert normalize_symbol(713) == "0713"
        assert normalize_symbol("713") == "0713"

    def test_numeric_4_digits_stays_4(self):
        assert normalize_symbol(9805) == "9805"
        assert normalize_symbol("9805") == "9805"
        assert normalize_symbol(2330) == "2330"

    def test_numeric_5_digits_stays_5(self):
        assert normalize_symbol(12345) == "12345"
        assert normalize_symbol("00965") == "00965"

    def test_numeric_6_digits_stays_6(self):
        assert normalize_symbol(123456) == "123456"
        assert normalize_symbol("009805") == "009805"

    def test_alphanumeric_uppercased_no_padding(self):
        assert normalize_symbol("00687B") == "00687B"
        assert normalize_symbol("00687b") == "00687B"
        assert normalize_symbol("AAPL") == "AAPL"
        assert normalize_symbol("aapl") == "AAPL"

    def test_strips_whitespace(self):
        assert normalize_symbol("  50  ") == "0050"
        assert normalize_symbol(" AAPL ") == "AAPL"

    def test_string_numeric_with_leading_zeros_preserved(self):
        # When passed as string with leading zeros, they're preserved
        assert normalize_symbol("0050") == "0050"
        assert normalize_symbol("00713") == "00713"
