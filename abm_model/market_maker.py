from mesa import Model

from config import get_logger
from abm_model.market_agent import MarketAgent
from abm_model.utils import round_to_tick
from utils.models import MarketAction
from utils.order_book import OrderBook

logger = get_logger(__name__)


class MarketMaker(MarketAgent):
    inventory_max_coef: float = 1.5
    inventory_min_coef: float = 0.5

    def __init__(
            self,
            unique_id,
            model: Model,
            cash: float = 200000,
            assets_quantity: int = 2000,
            base_spread: float = 0.02,
            max_spread: float = 0.05,
            inventory_max_coef: float | None = None,
            inventory_min_coef: float | None = None,
    ):
        cls = type(self)
        super().__init__(unique_id=unique_id, model=model, cash=cash, assets_quantity=assets_quantity)
        self._base_spread = float(base_spread)
        if max_spread >= base_spread:
            self._max_spread = float(max_spread)
        elif max_spread < base_spread:
            raise ValueError(f"max_spread {max_spread} must be greater than base_spread {base_spread}.")
        else:
            self._max_spread = max(max_spread, self._base_spread + 0.05)
        self._inventory_max = int(assets_quantity * (inventory_max_coef if inventory_max_coef else cls.inventory_max_coef))
        self._inventory_min = int(assets_quantity * (inventory_min_coef if inventory_min_coef else cls.inventory_min_coef))

    @property
    def news_price_coeff(self) -> float:
        if not self.model.news_event_occurred:
            return 1.
        elif self.model._news_event_value > 0:
            return 1.005
        elif self.model._news_event_value < 0:
            return 0.995
        else:
            raise ValueError(f"Unexpected condition of `news_price_coeff`.")

    @property
    def news_spread_coeff(self) -> float:
        if not self.model.news_event_occurred:
            return 1.
        elif self.model._news_event_value > 0:
            return 0.95
        elif self.model._news_event_value < 0:
            return 1.05
        else:
            raise ValueError(f"Unexpected condition of `news_spread_coeff`.")

    @property
    def spread(self) -> float:
        if self.assets_quantity >= self._inventory_max:
            return self._max_spread * self.news_spread_coeff
        elif self.assets_quantity <= self._inventory_min:
            return self._base_spread * 0.95 * self.news_spread_coeff
        else:
            return self._base_spread * self.news_spread_coeff

    @property
    def buy_amount(self) -> float:
        if self.assets_quantity >= self._inventory_max:
            return self.cash * 0.3
        elif self.assets_quantity <= self._inventory_min:
            return self.cash * 0.8
        else:
            return self.cash * 0.5

    @property
    def sell_quantity(self) -> int:
        if self.assets_quantity >= self._inventory_max:
            return int(self.assets_quantity - (self._inventory_min + self._inventory_max) // 2)
        elif self.assets_quantity <= self._inventory_min:
            return max(int(self.assets_quantity // 3), 0)
        else:
            return int(self.assets_quantity // 2)

    def step(self):
        order_book: OrderBook = self.model.order_book
        order_book.cancel_limit_orders(self.unique_id)
        current_price = self.model.prices[-1] if not order_book.get_central_price() else order_book.get_central_price()

        best_bid = order_book.get_best_bid()
        if (self.wealth * 0.1 > self.cash or self.assets_quantity > self._inventory_max) and best_bid:
            market_qty = max((self.wealth * 0.1 - self.cash) // best_bid.price, best_bid.quantity)
            order_book.place_order(self.unique_id, MarketAction.SELL, best_bid.price, market_qty)

        best_ask = order_book.get_best_ask()
        if self._inventory_min * 0.1 > self.assets_quantity and best_ask:
            market_qty = max(self._inventory_min * 0.1 - self.assets_quantity, best_ask.quantity)
            order_book.place_order(self.unique_id, MarketAction.BUY, best_ask.price, market_qty)

        buy_price = round_to_tick(current_price * (1 - self.spread / 2) * self.news_price_coeff, self.model.tick_size)
        buy_qty = self.buy_amount // buy_price if self.buy_amount > buy_price else self.cash // buy_price
        order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, buy_price, int(buy_qty))

        if self.sell_quantity > 0:
            sell_price = round_to_tick(current_price * (1 + self.spread / 2) * self.news_price_coeff,
                                       self.model.tick_size)
            order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, sell_price, self.sell_quantity)
