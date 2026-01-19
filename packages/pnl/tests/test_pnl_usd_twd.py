# packages/pnl/tests/test_pnl_usd_twd.py
from decimal import Decimal
from packages.pnl.engine import Transaction, TxType, compute_position_weighted_avg_cost


def test_us_stock_weighted_avg_cost_and_pnl():
    """
    測試案例 A：AAPL
    USD/TWD = 32.0
    BUY  10 @150 fee 5
    BUY  10 @170 fee 5
    SELL  5 @180 fee 5
    close_price = 175
    期望：
      avg_cost = 160.5
      qty = 15
      realized = 92.5
      unrealized = 217.5
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY,  Decimal("10"), Decimal("150"), Decimal("5")),
        Transaction("2026-01-10", TxType.BUY,  Decimal("10"), Decimal("170"), Decimal("5")),
        Transaction("2026-01-20", TxType.SELL, Decimal("5"),  Decimal("180"), Decimal("5")),
    ]

    res = compute_position_weighted_avg_cost(txs, close_price=Decimal("175"))

    assert res.quantity == Decimal("15")
    assert res.avg_cost == Decimal("160.5")
    assert res.realized_pnl == Decimal("92.5")
    assert res.unrealized_pnl == Decimal("217.5")


def test_usd_to_twd_conversion_for_overview():
    """
    總覽折算測試（不放在引擎也行，但至少要驗證折算數字）
    unrealized_twd = 217.5 * 32 = 6960
    realized_twd   = 92.5  * 32 = 2960
    value_twd      = 15 * 175 * 32 = 84000
    """
    usd_twd = Decimal("32.0")
    unrealized_usd = Decimal("217.5")
    realized_usd = Decimal("92.5")

    unrealized_twd = unrealized_usd * usd_twd
    realized_twd = realized_usd * usd_twd

    qty = Decimal("15")
    close_price = Decimal("175")
    value_twd = qty * close_price * usd_twd

    assert unrealized_twd == Decimal("6960.0")
    assert realized_twd == Decimal("2960.0")
    assert value_twd == Decimal("84000.0")