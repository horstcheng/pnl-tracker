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
    raise NotImplementedError("尚未實作")