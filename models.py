from enum import Enum


class MarketAction(Enum):
    BUY = 2
    BUY_LIMIT = 1
    ABSTAIN = 0
    SELL_LIMIT = -1
    SELL = -2
