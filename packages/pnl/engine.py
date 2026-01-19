from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import List


class TxType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    FEE = "FEE"


@dataclass(frozen=True)
class Transaction:
    trade_date: str
    tx_type: TxType
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal("0")


@dataclass(frozen=True)
class PositionResult:
    quantity: Decimal
    avg_cost: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


def compute_position_weighted_avg_cost(
    transactions: List[Transaction],
    close_price: Decimal,
) -> PositionResult:
    quantity = Decimal("0")
    total_cost = Decimal("0")
    realized_pnl = Decimal("0")

    sorted_txs = sorted(transactions, key=lambda tx: tx.trade_date)

    for tx in sorted_txs:
        if tx.tx_type == TxType.BUY:
            total_cost += tx.price * tx.quantity + tx.fee
            quantity += tx.quantity
        elif tx.tx_type == TxType.SELL:
            if tx.quantity > quantity:
                raise ValueError("Sell quantity exceeds current holding")
            avg_cost = total_cost / quantity
            realized_pnl += (tx.price - avg_cost) * tx.quantity - tx.fee
            total_cost -= avg_cost * tx.quantity
            quantity -= tx.quantity

    if quantity == Decimal("0"):
        avg_cost = Decimal("0")
        unrealized_pnl = Decimal("0")
    else:
        avg_cost = total_cost / quantity
        unrealized_pnl = (close_price - avg_cost) * quantity

    return PositionResult(
        quantity=quantity,
        avg_cost=avg_cost,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
    )