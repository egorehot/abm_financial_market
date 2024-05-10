from typing import NamedTuple
from enum import Enum


class MarketAction(Enum):
    BUY = 2
    BUY_LIMIT = 1
    ABSTAIN = 0
    SELL_LIMIT = -1
    SELL = -2


class Transaction(NamedTuple):
    buyer_id: int
    seller_id: int
    price: float
    quantity: int
