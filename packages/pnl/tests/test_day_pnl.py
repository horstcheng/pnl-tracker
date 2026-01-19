# packages/pnl/tests/test_day_pnl.py
"""Unit tests for Day P&L calculation."""
from decimal import Decimal

from services.api.daily_close import compute_day_pnl


def test_day_pnl_basic_calculation():
    """
    Test basic Day P&L calculation.

    Day P&L = (today_close - prev_close) * quantity
    Example: (100 - 90) * 10 = 100
    """
    symbol_positions = {
        "AAPL": (Decimal("10"), "USD"),
    }
    prices = {"AAPL": Decimal("100")}
    prev_prices = {"AAPL": Decimal("90")}
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # (100 - 90) * 10 = 100 USD -> 100 * 32 = 3200 TWD
    assert total_day_pnl == Decimal("3200")
    assert len(sorted_pnl) == 1
    assert sorted_pnl[0][0] == "AAPL"
    assert sorted_pnl[0][1] == Decimal("3200")


def test_day_pnl_negative():
    """
    Test Day P&L when price goes down.

    Day P&L = (today_close - prev_close) * quantity
    Example: (85 - 100) * 20 = -300
    """
    symbol_positions = {
        "TSLA": (Decimal("20"), "USD"),
    }
    prices = {"TSLA": Decimal("85")}
    prev_prices = {"TSLA": Decimal("100")}
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # (85 - 100) * 20 = -300 USD -> -300 * 32 = -9600 TWD
    assert total_day_pnl == Decimal("-9600")
    assert sorted_pnl[0][1] == Decimal("-9600")


def test_day_pnl_twd_symbol():
    """
    Test Day P&L for TWD symbols (no currency conversion).

    Day P&L = (today_close - prev_close) * quantity
    """
    symbol_positions = {
        "2330": (Decimal("1000"), "TWD"),
    }
    prices = {"2330": Decimal("650")}
    prev_prices = {"2330": Decimal("640")}
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # (650 - 640) * 1000 = 10000 TWD (no conversion needed)
    assert total_day_pnl == Decimal("10000")


def test_day_pnl_multiple_symbols():
    """
    Test Day P&L with multiple symbols, mixed currencies.
    """
    symbol_positions = {
        "AAPL": (Decimal("10"), "USD"),
        "2330": (Decimal("100"), "TWD"),
    }
    prices = {
        "AAPL": Decimal("150"),
        "2330": Decimal("600"),
    }
    prev_prices = {
        "AAPL": Decimal("140"),
        "2330": Decimal("590"),
    }
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # AAPL: (150 - 140) * 10 = 100 USD -> 3200 TWD
    # 2330: (600 - 590) * 100 = 1000 TWD
    # Total: 3200 + 1000 = 4200 TWD
    assert total_day_pnl == Decimal("4200")
    assert len(sorted_pnl) == 2


def test_day_pnl_missing_prev_close():
    """
    Test that symbols without prev_close are skipped (Day P&L = 0).
    """
    symbol_positions = {
        "AAPL": (Decimal("10"), "USD"),
        "NEWSTOCK": (Decimal("50"), "USD"),  # no prev_close
    }
    prices = {
        "AAPL": Decimal("150"),
        "NEWSTOCK": Decimal("100"),
    }
    prev_prices = {
        "AAPL": Decimal("140"),
        # NEWSTOCK missing
    }
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # Only AAPL counted: (150 - 140) * 10 = 100 USD -> 3200 TWD
    assert total_day_pnl == Decimal("3200")
    assert len(sorted_pnl) == 1
    assert sorted_pnl[0][0] == "AAPL"


def test_day_pnl_zero_quantity():
    """
    Test that symbols with zero quantity have no Day P&L.
    """
    symbol_positions = {
        "AAPL": (Decimal("0"), "USD"),
    }
    prices = {"AAPL": Decimal("150")}
    prev_prices = {"AAPL": Decimal("140")}
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    assert total_day_pnl == Decimal("0")
    assert len(sorted_pnl) == 0


def test_day_pnl_sorting_by_absolute_value():
    """
    Test that sorted_pnl is sorted by absolute value descending.
    """
    symbol_positions = {
        "A": (Decimal("10"), "TWD"),
        "B": (Decimal("10"), "TWD"),
        "C": (Decimal("10"), "TWD"),
    }
    prices = {
        "A": Decimal("110"),  # (110-100)*10 = +100
        "B": Decimal("80"),   # (80-100)*10 = -200
        "C": Decimal("155"),  # (155-150)*10 = +50
    }
    prev_prices = {
        "A": Decimal("100"),
        "B": Decimal("100"),
        "C": Decimal("150"),
    }
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # Sorted by absolute value: B (-200), A (100), C (50)
    assert sorted_pnl[0][0] == "B"
    assert sorted_pnl[0][1] == Decimal("-200")
    assert sorted_pnl[1][0] == "A"
    assert sorted_pnl[1][1] == Decimal("100")
    assert sorted_pnl[2][0] == "C"
    assert sorted_pnl[2][1] == Decimal("50")


def test_day_pnl_does_not_affect_total_pnl():
    """
    Verify that Total P&L calculation remains unchanged.

    This test imports the existing compute_all_pnl function and ensures
    that the Total P&L logic is not affected by Day P&L changes.
    """
    from packages.pnl.engine import Transaction, TxType, compute_position_weighted_avg_cost

    # Same test case from test_pnl_twd.py
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("100"), Decimal("600"), Decimal("100")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("650"))

    # Total P&L logic unchanged: avg_cost = 601, unrealized = (650-601)*100 = 4900
    assert res.quantity == Decimal("100")
    assert res.avg_cost == Decimal("601")
    assert res.realized_pnl == Decimal("0")
    assert res.unrealized_pnl == Decimal("4900")


def test_day_pnl_lenient_mode_with_prev_close():
    """
    Test lenient mode: when prev_close exists, day_pnl is computed normally.

    This verifies the happy path where both today_close and prev_close are available.
    """
    symbol_positions = {
        "MSFT": (Decimal("50"), "USD"),
        "2317": (Decimal("200"), "TWD"),
    }
    prices = {
        "MSFT": Decimal("400"),
        "2317": Decimal("120"),
    }
    prev_prices = {
        "MSFT": Decimal("380"),
        "2317": Decimal("115"),
    }
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # MSFT: (400 - 380) * 50 = 1000 USD -> 32000 TWD
    # 2317: (120 - 115) * 200 = 1000 TWD
    # Total: 32000 + 1000 = 33000 TWD
    assert total_day_pnl == Decimal("33000")
    assert len(sorted_pnl) == 2
    # Both symbols should be in the result
    symbols_in_result = {s[0] for s in sorted_pnl}
    assert "MSFT" in symbols_in_result
    assert "2317" in symbols_in_result


def test_day_pnl_lenient_mode_missing_prev_close_returns_zero():
    """
    Test lenient mode: when only one valid close exists (prev_close missing),
    day_pnl for that symbol is 0 and symbol should be in missing_prev_close list.

    This test verifies:
    1) compute_day_pnl returns day_pnl=0 for symbols without prev_close
    2) The symbol is correctly excluded from the result list
    3) Other symbols with prev_close are still computed correctly
    """
    symbol_positions = {
        "AAPL": (Decimal("10"), "USD"),
        "NEWIPO": (Decimal("100"), "USD"),  # Only one close value exists
    }
    prices = {
        "AAPL": Decimal("180"),
        "NEWIPO": Decimal("50"),
    }
    prev_prices = {
        "AAPL": Decimal("175"),
        # NEWIPO missing - simulates only one valid close value
    }
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # Only AAPL is counted: (180 - 175) * 10 = 50 USD -> 1600 TWD
    # NEWIPO has day_pnl=0 (excluded from today's move)
    assert total_day_pnl == Decimal("1600")
    assert len(sorted_pnl) == 1
    assert sorted_pnl[0][0] == "AAPL"

    # Verify NEWIPO is NOT in the result (its day_pnl is effectively 0)
    symbols_in_result = {s[0] for s in sorted_pnl}
    assert "NEWIPO" not in symbols_in_result


def test_format_slack_message_includes_missing_prev_close_note():
    """
    Test that format_slack_message includes the lenient mode note when
    missing_prev_close list is non-empty.
    """
    from services.api.daily_close import format_slack_message

    message = format_slack_message(
        today="2026-01-19",
        usd_twd=Decimal("32.5"),
        total_pnl=Decimal("100000"),
        user_pnl={"user1": Decimal("100000")},
        top_symbols=[("AAPL", Decimal("50000")), ("TSLA", Decimal("30000"))],
        symbols_count=10,
        missing_symbols=[],
        lookup_attempts={},
        total_day_pnl=Decimal("5000"),
        top_day_symbols=[("AAPL", Decimal("3000"))],
        missing_prev_close=["NEWIPO", "IPO2"],
    )

    # Verify the warning line exists
    assert "Day P&L warnings: missing previous close for" in message
    assert "IPO2" in message
    assert "NEWIPO" in message

    # Verify the note about lenient mode is included
    assert "Day P&L note: missing prev_close symbols are treated as 0 (excluded from today's move)." in message


def test_format_slack_message_no_note_when_all_prev_close_present():
    """
    Test that format_slack_message does NOT include the lenient mode note
    when all symbols have prev_close (missing_prev_close is empty).
    """
    from services.api.daily_close import format_slack_message

    message = format_slack_message(
        today="2026-01-19",
        usd_twd=Decimal("32.5"),
        total_pnl=Decimal("100000"),
        user_pnl={"user1": Decimal("100000")},
        top_symbols=[("AAPL", Decimal("50000"))],
        symbols_count=5,
        missing_symbols=[],
        lookup_attempts={},
        total_day_pnl=Decimal("5000"),
        top_day_symbols=[("AAPL", Decimal("3000"))],
        missing_prev_close=[],  # Empty - all symbols have prev_close
    )

    # Verify the warning and note are NOT present
    assert "Day P&L warnings:" not in message
    assert "Day P&L note:" not in message


def test_day_pnl_fallback_fast_info_provides_prev_close(monkeypatch):
    """
    Test fallback when history returns only 1 row but fast_info.previousClose exists.

    Simulates 00983A.TW scenario where yfinance history has only 1 data point
    but fast_info provides previousClose.
    """
    import pandas as pd
    from unittest.mock import MagicMock
    from services.api.daily_close import try_fetch_price_with_prev

    # Create mock ticker with 1-row history and fast_info.previousClose
    mock_ticker = MagicMock()

    # History with only 1 row
    mock_hist = pd.DataFrame({
        "Close": [12.50],
    }, index=pd.to_datetime(["2026-01-17"]))
    mock_ticker.history.return_value = mock_hist

    # fast_info with previousClose
    mock_ticker.fast_info = {"previousClose": 12.30}

    # Patch yf.Ticker to return our mock
    def mock_yf_ticker(symbol):
        return mock_ticker

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", mock_yf_ticker)

    success, today_close, prev_close, debug_info = try_fetch_price_with_prev("00983A.TW", "00983A")

    assert success is True
    assert today_close == Decimal("12.50")
    assert prev_close == Decimal("12.30")  # From fallback
    assert debug_info["fallback_used"] is True
    assert debug_info["fallback_prev_close"] == "12.30"
    assert debug_info["rows"] == 1


def test_day_pnl_fallback_both_missing_stays_warning(monkeypatch):
    """
    Test when history returns 1 row AND fast_info.previousClose is None.

    prev_close should be None, symbol stays in missing_prev_close warnings,
    and day pnl = 0 (lenient mode).
    """
    import pandas as pd
    from unittest.mock import MagicMock
    from services.api.daily_close import try_fetch_price_with_prev

    # Create mock ticker with 1-row history and no fast_info.previousClose
    mock_ticker = MagicMock()

    # History with only 1 row
    mock_hist = pd.DataFrame({
        "Close": [12.50],
    }, index=pd.to_datetime(["2026-01-17"]))
    mock_ticker.history.return_value = mock_hist

    # fast_info without previousClose
    mock_ticker.fast_info = {}

    # Patch yf.Ticker to return our mock
    def mock_yf_ticker(symbol):
        return mock_ticker

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", mock_yf_ticker)

    success, today_close, prev_close, debug_info = try_fetch_price_with_prev("00983A.TW", "00983A")

    assert success is True
    assert today_close == Decimal("12.50")
    assert prev_close is None  # Both history and fallback failed
    assert debug_info["fallback_used"] is False
    assert debug_info["fallback_prev_close"] is None
    assert debug_info["rows"] == 1


def test_day_pnl_with_fallback_computes_correctly():
    """
    Test that Day P&L is computed correctly when prev_close comes from fallback.

    When prev_close is available (from fallback), the symbol should NOT be in
    missing_prev_close and Day P&L should be calculated normally.
    """
    symbol_positions = {
        "00983A": (Decimal("100"), "TWD"),
    }
    prices = {
        "00983A": Decimal("12.50"),
    }
    prev_prices = {
        "00983A": Decimal("12.30"),  # Came from fallback
    }
    usd_twd = Decimal("32")

    total_day_pnl, sorted_pnl = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)

    # (12.50 - 12.30) * 100 = 20 TWD
    assert total_day_pnl == Decimal("20")
    assert len(sorted_pnl) == 1
    assert sorted_pnl[0][0] == "00983A"
    assert sorted_pnl[0][1] == Decimal("20")
