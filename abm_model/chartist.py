import math
from collections import defaultdict

from mesa import Model
import numpy as np

from abm_model.market_agent import MarketAgent
from utils.models import MarketAction
from utils.order_book import OrderBook


class ChartistAgent(MarketAgent):
    cash_range: list[float] = [10**4, 10**5]
    optimistic_ratio: float = 0.5
    take_profit_range: list[float] = [0.01, 0.1]
    revaluation_freq: float = 1
    majority_importance: float = 0.5
    price_trend_importance: float = 0.5
    order_amount_range: list[float] = [0.01, 0.2]

    def __init__(self, unique_id: int, model: Model, asset_quantity: int = 0):
        cls = type(self)
        cash = np.random.uniform(*cls.cash_range)
        super().__init__(unique_id=unique_id, model=model, cash=cash, asset_quantity=asset_quantity)
        self.__is_optimistic = np.random.choice([True, False], p=[cls.optimistic_ratio, 1 - cls.optimistic_ratio])
        self.take_profit = np.random.uniform(*cls.take_profit_range)
        self.__opened_pos = defaultdict(int)
        self.__order_amount_perc = np.random.uniform(*cls.order_amount_range)

        self.model._optimistic_chartists_number += int(self.__is_optimistic)

    @property
    def is_optimistic(self):
        return self.__is_optimistic

    @is_optimistic.setter
    def is_optimistic(self, value: bool):
        if self.__is_optimistic != value:
            self.model._optimistic_chartists_number += 1 if not self.__is_optimistic else -1
        self.__is_optimistic = value

    def update_opened_pos(self, action: MarketAction, price: float, quantity: int):
        quantity = quantity if action.value > 0 else -quantity
        self.__opened_pos[price] += quantity
        if self.__opened_pos[price] == 0:
            self.__opened_pos.pop(price)

    def __majority_opinion(self):
        cls = type(self)
        chartists_number = len(self.model.agents.get(cls.__name__))
        optimists_number = self.model._optimistic_chartists_number
        pessimists_number = chartists_number - optimists_number
        majority = (optimists_number - pessimists_number) / chartists_number
        price_trend = (self.model.prices[-1] - self.model.prices[-2]) / self.model.prices[-2]
        return cls.majority_importance * majority + cls.price_trend_importance * price_trend

    def __evaluate_opinion(self):
        cls = type(self)
        chartists_number = len(self.model.agents.get(cls.__name__))
        change_proba = (cls.revaluation_freq * chartists_number /
                        len(self.model.agents) * math.e**(self.__majority_opinion() * 1 if self.is_optimistic else -1))
        self.is_optimistic = np.random.choice([self.is_optimistic, not self.is_optimistic], p=[1 - change_proba, change_proba])

    def _calc_order_quantity(self, price: float) -> int:
        order_amount = min(self.wealth * self.__order_amount_perc, self._cash)
        return int(order_amount // price)

    def step(self) -> None:
        order_book: OrderBook = self.model.order_book
        self.__evaluate_opinion()
        best_bid_price = order_book.get_best_bid().price
        best_ask_price = order_book.get_best_ask().price
        for price, quantity in self.__opened_pos.items():
            if quantity > 0:
                if not self.is_optimistic:
                    order_book.place_order(self.unique_id, MarketAction.SELL, price, quantity)
                    self.__opened_pos.pop(price)
                elif best_bid_price >= (1 + self.take_profit) * price:
                    order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, best_bid_price, quantity)
            elif quantity < 0:
                if self.is_optimistic:
                    order_book.place_order(self.unique_id, MarketAction.BUY, price, quantity)
                    self.__opened_pos.pop(price)
                elif best_ask_price <= (1 - self.take_profit) * price:
                    order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, best_ask_price, quantity)
            elif quantity == 0:
                raise ValueError(f"Opened position with zero quantity. Agent: {self.unique_id}, price: {price}.")

        if self.is_optimistic:
            order_quantity = self._calc_order_quantity(best_ask_price)
            order_book.place_order(self.unique_id, MarketAction.BUY, 0, order_quantity)
        else:
            order_quantity = self._calc_order_quantity(best_bid_price)
            order_book.place_order(self.unique_id, MarketAction.SELL, 0, order_quantity)


