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
