from decimal import Decimal
from packages.pnl.engine import Transaction, TxType, compute_position_weighted_avg_cost

def test_dividend_increases_realized_pnl():
    """
    Test that DIVIDEND transaction increases realized P&L.
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("10"), Decimal("100"), Decimal("0")),
        # Dividend: 10 shares * 5.0 = 50.0
        Transaction("2026-01-05", TxType.DIVIDEND, Decimal("10"), Decimal("5"), Decimal("0")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("110"))

    # Avg cost: 100
    # Unrealized: (110 - 100) * 10 = 100
    # Realized: 50 (from dividend)
    assert res.quantity == Decimal("10")
    assert res.avg_cost == Decimal("100")
    assert res.unrealized_pnl == Decimal("100")
    assert res.realized_pnl == Decimal("50")

def test_dividend_with_fee():
    """
    Test that DIVIDEND with fee is calculated correctly.
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("10"), Decimal("100"), Decimal("0")),
        # Dividend: 50.0, Fee: 5.0 -> Net 45.0
        Transaction("2026-01-05", TxType.DIVIDEND, Decimal("10"), Decimal("5"), Decimal("5")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("110"))

    assert res.realized_pnl == Decimal("45")

def test_fee_decreases_realized_pnl():
    """
    Test that FEE transaction decreases realized P&L.
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("10"), Decimal("100"), Decimal("0")),
        # Fee: 1 * 20 = 20
        Transaction("2026-01-05", TxType.FEE, Decimal("1"), Decimal("20"), Decimal("0")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("110"))

    # Realized: -20
    assert res.realized_pnl == Decimal("-20")

def test_fee_with_transaction_fee():
    """
    Test that FEE with 'fee' attribute is calculated correctly.
    Total deduction = quantity * price + fee
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("10"), Decimal("100"), Decimal("0")),
        # Fee transaction: 10 + 2 = 12
        Transaction("2026-01-05", TxType.FEE, Decimal("1"), Decimal("10"), Decimal("2")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("110"))

    assert res.realized_pnl == Decimal("-12")

def test_dividend_and_fee_mixed_with_trading():
    """
    Test mixing BUY, SELL, DIVIDEND, FEE.
    """
    txs = [
        Transaction("2026-01-01", TxType.BUY, Decimal("10"), Decimal("100"), Decimal("0")),
        # Dividend: +50
        Transaction("2026-01-02", TxType.DIVIDEND, Decimal("10"), Decimal("5"), Decimal("0")),
        # Fee: -10
        Transaction("2026-01-03", TxType.FEE, Decimal("1"), Decimal("10"), Decimal("0")),
        # Sell 5 @ 120. Cost basis 100.
        # Realized from sell: (120 - 100) * 5 = 100.
        # Total Realized: 50 - 10 + 100 = 140.
        Transaction("2026-01-04", TxType.SELL, Decimal("5"), Decimal("120"), Decimal("0")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("130"))

    assert res.quantity == Decimal("5")
    assert res.avg_cost == Decimal("100")
    # Unrealized: (130 - 100) * 5 = 150
    assert res.unrealized_pnl == Decimal("150")
    assert res.realized_pnl == Decimal("140")
