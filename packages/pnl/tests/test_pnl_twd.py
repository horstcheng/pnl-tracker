# packages/pnl/tests/test_pnl_twd.py
from decimal import Decimal
from packages.pnl.engine import Transaction, TxType, compute_position_weighted_avg_cost


def test_tw_stock_simple_unrealized_pnl():
    """
    測試案例 B：2330
    BUY  100 @600 fee 100
    close_price = 650
    avg_cost = (600*100 + 100)/100 = 601
    unrealized = (650-601)*100 = 4900
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY, Decimal("100"), Decimal("600"), Decimal("100")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("650"))

    assert res.quantity == Decimal("100")
    assert res.avg_cost == Decimal("601")
    assert res.realized_pnl == Decimal("0")
    assert res.unrealized_pnl == Decimal("4900")