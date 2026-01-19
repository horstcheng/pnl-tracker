from decimal import Decimal
from packages.pnl.engine import Transaction, TxType, compute_position_weighted_avg_cost


def test_tw_stock_simple_unrealized_pnl():
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("100"), Decimal("600"), Decimal("100")),
    ]
    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("650"))

    assert res.quantity == Decimal("100")
    assert res.avg_cost == Decimal("601")
    assert res.realized_pnl == Decimal("0")
    assert res.unrealized_pnl == Decimal("4900")