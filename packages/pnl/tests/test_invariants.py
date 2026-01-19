# packages/pnl/tests/test_invariants.py
from decimal import Decimal
import pytest
from packages.pnl.engine import Transaction, TxType, compute_position_weighted_avg_cost


def test_invariant_no_negative_quantity():
    txs = [
        Transaction("2026-01-02", TxType.BUY,  Decimal("1"), Decimal("100"), Decimal("0")),
        Transaction("2026-01-03", TxType.SELL, Decimal("2"), Decimal("110"), Decimal("0")),
    ]

    # 你可以選擇：賣超就 raise（我建議）
    with pytest.raises(Exception):
        compute_position_weighted_avg_cost(txs, close_price=Decimal("110"))


def test_invariant_avg_cost_unchanged_after_sell():
    """
    加權平均法：賣出不會改變 avg_cost（只改 qty & realized）
    """
    txs = [
        Transaction("2026-01-02", TxType.BUY,  Decimal("10"), Decimal("100"), Decimal("0")),
        Transaction("2026-01-03", TxType.BUY,  Decimal("10"), Decimal("200"), Decimal("0")),
    ]

    res_before_sell = compute_position_weighted_avg_cost(txs, close_price=Decimal("150"))
    avg_before = res_before_sell.avg_cost

    txs2 = txs + [
        Transaction("2026-01-04", TxType.SELL, Decimal("5"), Decimal("180"), Decimal("0")),
    ]
    res_after_sell = compute_position_weighted_avg_cost(txs2, close_price=Decimal("150"))

    assert res_after_sell.avg_cost == avg_before