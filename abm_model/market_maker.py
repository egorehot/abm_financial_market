from mesa import Model

from abm_model.market_agent import MarketAgent
from utils.models import MarketAction
from utils.order_book import OrderBook


class MarketMaker(MarketAgent):
    inventory_max_coef: float = 1.5
    inventory_min_coef: float = 0.3

    def __init__(
            self,
            unique_id,
            model: Model,
            cash: float = 10**5,
            assets_quantity: int = 1000,
            base_spread: float = 0.05,
            max_spread: float = 0.1,
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

    def fill_order_book(self):
        order_book: OrderBook = self.model.order_book

        if len(order_book.bid) == 0 or not order_book._mm_bid:
            spread = self._max_spread if self.assets_quantity > self._inventory_max else self._base_spread
            order_price = round(self.model.prices[-1] * (1 - spread / 2), self.model.tick_size)
            if self.cash * 0.7 > order_price:
                order_qty = self.cash * 0.7 // order_price
            else:
                order_qty = self.cash // order_price
            order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, order_price, int(order_qty))

        if len(order_book.ask) == 0 or not order_book._mm_ask:
            spread = self._base_spread # if self.assets_quantity < self._inventory_min else self._max_spread
            order_price = round(self.model.prices[-1] * (1 + spread / 2), self.model.tick_size)
            if self.assets_quantity > self._inventory_min:
                order_qty = self.assets_quantity - self._inventory_min
            else:
                order_qty = self.assets_quantity
            order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, order_price, int(order_qty))

    def step(self):
        order_book: OrderBook = self.model.order_book
        order_book.cancel_limit_orders(self.unique_id)
        current_price = self.model.prices[-1] if not order_book.get_central_price() else order_book.get_central_price()
        if self.assets_quantity > self._inventory_max:
            order_price = round(current_price * (1 + self._max_spread / 2), self.model.tick_size)
            order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, order_price,
                                   self.assets_quantity - self._inventory_max + 1)
        elif self.assets_quantity < self._inventory_min:
            order_price = round(current_price * (1 - self._base_spread / 2), self.model.tick_size)
            order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, order_price,
                                   self._inventory_min - self.assets_quantity + 1)
        else:
            best_ask = order_book.get_best_ask()
            best_bid = order_book.get_best_bid()
            if best_ask and current_price * (1 + self._base_spread / 2) >= best_ask.price:
                order_book.place_order(self.unique_id, MarketAction.BUY, best_ask.price, best_ask.quantity)
            if best_bid and current_price * (1 - self._base_spread / 2) <= best_bid.price:
                order_book.place_order(self.unique_id, MarketAction.SELL, best_bid.price, best_bid.quantity)
        self.fill_order_book()
        self.model.prices.append(round(order_book.get_central_price(), self.model.tick_size))
