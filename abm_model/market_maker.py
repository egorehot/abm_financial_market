from mesa import Agent, Model

from abm_model.market_agent import MarketAgent
from utils.models import MarketAction
from utils.order_book import OrderBook


class MarketMaker(MarketAgent):
    def __init__(
            self,
            unique_id,
            model: Model,
            cash: float = 10**6,
            assets_quantity: int = 1000,
            initial_price: float = 100,
            spread: float = 0.1,
    ):
        cls = type(self)
        super().__init__(unique_id=unique_id, model=model, cash=cash, asset_quantity=assets_quantity)
        self._spread = float(spread)
        self._market_price = initial_price

    def fill_order_book(self):
        order_book: OrderBook = self.model.order_book

        if len(order_book.bid) == 0:
            order_price = self._market_price * (1 - self._spread / 2)
            order_qty = max(self.asset_quantity * 0.3, 20)
            order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, order_price, order_qty)

        if len(order_book.ask) == 0:
            order_price = self._market_price * (1 + self._spread / 2)
            order_qty = max(self.asset_quantity * 0.3, 20)
            order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, order_price, order_qty)

    def step(self):
        order_book: OrderBook = self.model.order_book
        self.model.prices.append(order_book.get_mu_spread())
