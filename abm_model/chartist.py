import math

from mesa import Model
import numpy as np

from abm_model.market_agent import MarketAgent
from utils.models import MarketAction
from utils.order_book import OrderBook


class ChartistAgent(MarketAgent):
    """
    _distr: [mu, sigma] lognormal
    _range: [min, max] uniform
    """
    cash_distr: list[float] = [0.9, 0.3]
    cash_scale: float = 1000.
    optimistic_ratio: float = 0.51
    take_profit_range: list[float] = [0.01, 0.1]
    revaluation_freq: float = 0.5
    majority_importance: float = -1.
    price_trend_importance: float = -2.5
    order_amount_range: list[float] = [0.03, 0.17]
    lambda_limit: float = 3.

    def __init__(self, unique_id: int, model: Model, cash: float | None = None, assets_quantity: int | None = None):
        cls = type(self)
        cash = np.random.lognormal(*cls.cash_distr) * cls.cash_scale if not cash else cash
        super().__init__(unique_id=unique_id, model=model, cash=cash, assets_quantity=assets_quantity)
        self.__is_optimistic = np.random.choice([True, False], p=[cls.optimistic_ratio, 1 - cls.optimistic_ratio])
        self.take_profit = np.random.uniform(*cls.take_profit_range)
        self.__avg_opened_price = 0
        self.__order_amount_perc = np.random.uniform(*cls.order_amount_range)

        self.model._optimistic_chartists_number += int(self.__is_optimistic)

    @property
    def is_optimistic(self):
        return self.__is_optimistic

    @is_optimistic.setter
    def is_optimistic(self, value: bool):
        if self.__is_optimistic != value:
            self.model._optimistic_chartists_number += 1 if not self.__is_optimistic else -1
            side = 'bid' if self.__is_optimistic else 'ask'
            self.model.order_book.cancel_limit_orders(self.unique_id, side)
        self.__is_optimistic = value

    def __majority_opinion(self):
        cls = type(self)
        chartists_number = len(self.model.get_agents_of_type(cls))
        optimists_number = self.model._optimistic_chartists_number
        pessimists_number = chartists_number - optimists_number
        majority = (optimists_number - pessimists_number) / chartists_number
        price_trend = ((self.model.prices[-1] - self.model.prices[-2]) / self.model.prices[-2]) if len(self.model.prices) > 1 else 0.001
        return cls.majority_importance * majority + cls.price_trend_importance * price_trend / cls.revaluation_freq

    def __evaluate_opinion(self):
        cls = type(self)
        chartists_number = len(self.model.get_agents_of_type(cls))
        change_proba = (cls.revaluation_freq * chartists_number /
                        len(self.model.agents) * math.e**(self.__majority_opinion() * (1 if self.is_optimistic else -1)))
        change_proba = min(change_proba, 1.)
        self.is_optimistic = np.random.choice([self.is_optimistic, not self.is_optimistic], p=[1 - change_proba, change_proba])

    def _calc_order_quantity(self, intention: MarketAction, price: float | None = None) -> int:
        current_price = price if price else self.model.order_book.get_central_price()
        if current_price <= 0: return 0
        if intention.value > 0:
            order_qty = min(self.wealth * self.__order_amount_perc, self.cash) // current_price
        elif intention.value < 0:
            order_qty = self.wealth * self.__order_amount_perc // current_price
        else:
            raise ValueError(f'Wrong `MarketAction`. Got {intention}')
        return max(int(order_qty), 0)

    def _calc_limit_price(self, order_book: OrderBook | None = None) -> float:
        """
        f(x,lambda_limit,mu_spread) = lambda_limit * e**(-abs(lambda_limit * (x - mu_spread)))
        mu_spread = 1/2 * (best_ask + best_bid)
        """
        cls = type(self)
        order_book: OrderBook = self.model.order_book if not order_book else order_book
        return round(np.random.laplace(order_book.get_central_price(), 1 / cls.lambda_limit), self.model.tick_size)

    def step(self) -> None:
        order_book: OrderBook = self.model.order_book
        self.__evaluate_opinion()
        best_bid_price = order_book.get_best_bid().price
        best_ask_price = order_book.get_best_ask().price
        if self.assets_quantity < 0 and self.is_optimistic:
            order_book.place_order(self.unique_id, MarketAction.BUY, best_ask_price, self.assets_quantity)
            self.__avg_opened_price = 0
            return
        elif self.assets_quantity < 0:
            pass # TODO check best_bid - take_profit

        if self.assets_quantity > 0 and not self.is_optimistic:
            order_book.place_order(self.unique_id, MarketAction.SELL, best_bid_price, self.assets_quantity)
            self.__avg_opened_price = 0
            return
        elif self.assets_quantity > 0:
            pass # TODO check best_ask + take_profit

        if self.bankrupt: return
        order_price = self._calc_limit_price()
        if self.is_optimistic:
            order_quantity = self._calc_order_quantity(MarketAction.BUY_LIMIT, order_price)
            if order_quantity > 0:
                order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, order_price, order_quantity)
        else:
            order_quantity = self._calc_order_quantity(MarketAction.SELL_LIMIT, order_price)
            if order_quantity > 0:
                order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, order_price, order_quantity)
