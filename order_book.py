from bisect import insort_right
from collections import deque
from dataclasses import dataclass, field
from numbers import Real
from operator import attrgetter
import time

from models import MarketAction


@dataclass
class Order:
    agent_id: int | str
    type: MarketAction
    price: Real
    quantity: int
    ts: float = field(default_factory=time.time)


class OrderBook:
    def __init__(self):
        self.__bid: list[Order | None] = []
        self.__ask: list[Order | None] = []
        self.__market_orders: deque[Order | None] = deque([])

    def __len__(self):
        return len(self.__bid) + len(self.__ask)

    @property
    def bid(self):
        return self.__bid

    @property
    def ask(self):
        return self.__ask

    @property
    def market_orders(self):
        return self.__market_orders

    def get_best_ask(self) -> Order | None:
        if len(self.ask) == 0: return
        return self.ask[0]

    def get_best_bid(self) -> Order | None:
        if len(self.bid) == 0: return
        return self.bid[0]

    def get_mu_spread(self) -> Real | None:
        if not all([self.get_best_ask().price, self.get_best_bid().price]): return
        return 0.5 * (self.get_best_ask().price + self.get_best_bid().price)

    def _add_ask(self, order: Order):
        insort_right(self.__ask, order, key=attrgetter('price'))

    def _add_bid(self, order: Order):
        insort_right(self.__bid, order, key=lambda x: -x.price)

    def _add_market(self, order: Order):
        self.__market_orders.append(order)

    def place_order(self, agent_id: int | str, action: MarketAction, price: Real, quantity: int):
        order = Order(agent_id=agent_id, type=action, price=price, quantity=quantity)
        match action:
            case MarketAction.BUY | MarketAction.SELL:
                self._add_market(order)
            case MarketAction.BUY_LIMIT:
                self._add_bid(order)
            case MarketAction.SELL_LIMIT:
                self._add_ask(order)
            case MarketAction.ABSTAIN:
                pass
            case _:
                raise ValueError(f"Invalid action '{action}'. "
                                 f"Expected MarketAction one from `{', '.join(MarketAction.__members__.keys())}`.")

    @staticmethod
    def __make_transaction(order: Order, matched: Order) -> dict:
        trade_qty = min(order.quantity, matched.quantity)
        if abs(order.type.value) == 1:
            trade_price = order.price if order.ts < matched.ts else matched.price
        else:
            trade_price = matched.price
        transaction = {
            'buyer_id': order.agent_id if order.type.value > 0 else matched.agent_id,
            'seller_id': matched.agent_id if matched.type.value < 0 else order.agent_id,
            'price': trade_price,
            'quantity': trade_qty
        }
        order.quantity -= trade_qty
        matched.quantity -= trade_qty
        return transaction

    def __execute_market_order(self, order: Order) -> list[dict | None]:
        assert abs(order.type.value) == 2  # BUY or SELL
        transactions = []

        opposite = self.__ask if order.type == MarketAction.BUY else self.__bid
        idx = 0
        while order.quantity > 0 and idx < len(opposite):
            best_match = opposite[idx]
            transactions.append(self.__make_transaction(order, best_match))
            idx += 1

        return transactions


    def execute_orders(self) -> list[dict | None]:
        transactions = []
        while self.__market_orders:
            market_order = self.__market_orders.popleft()
            transactions.extend(self.__execute_market_order(market_order))

        bid_idx, ask_idx = 0, 0
        while bid_idx < len(self.__bid) and ask_idx < len(self.__ask):
            bid = self.__bid[bid_idx]
            ask = self.__ask[ask_idx]
            if bid.price < ask.price: break

            transactions.append(self.__make_transaction(bid, ask))
            if bid.quantity == 0:
                bid_idx += 1
            if ask.quantity == 0:
                ask_idx += 1

        if transactions:
            self.__bid = [b for b in self.__bid if b.quantity > 0]
            self.__ask = [a for a in self.__ask if a.quantity > 0]

        return transactions
