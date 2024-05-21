import math

from mesa import Model
import numpy as np

import config
from abm_model.market_agent import MarketAgent
from utils.models import MarketAction
from utils.order_book import OrderBook

logger = config.get_logger(__name__)


logger.debug(f'Seed: {config.RANDOM_SEED}')
RNG = np.random.default_rng(config.RANDOM_SEED)


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
    lambda_limit: float = 3.5

    def __init__(self, unique_id: int, model: Model, cash: float | None = None, assets_quantity: int | None = None):
        cls = type(self)
        cash = RNG.lognormal(*cls.cash_distr) * cls.cash_scale if not cash else cash
        super().__init__(unique_id=unique_id, model=model, cash=cash, assets_quantity=assets_quantity)
        self.__is_optimistic = RNG.choice([True, False], p=[cls.optimistic_ratio, 1 - cls.optimistic_ratio])
        self.take_profit = RNG.uniform(*cls.take_profit_range)
        self.__avg_opened_price = 0
        self.__order_amount_perc = RNG.uniform(*cls.order_amount_range)

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
            logger.debug(f"Step: {self.model.schedule.steps}. Agent: {self.unique_id}. Changing opinion.")
        self.__is_optimistic = value

    def __majority_opinion(self):
        cls = type(self)
        chartists_number = len(self.model.get_agents_of_type(cls))
        optimists_number = self.model._optimistic_chartists_number
        pessimists_number = chartists_number - optimists_number
        majority = (optimists_number - pessimists_number) / chartists_number
        price_trend = ((self.model.prices[-1] - self.model.prices[-2]) / self.model.prices[-2]) if len(self.model.prices) > 1 else 0.001
        logger.debug(f"Step: {self.model.schedule.steps}. Agent: {self.unique_id}. "
                     f"Majority: {round(majority, 3)}. Price trend: {round(price_trend, 3)}.")
        return cls.majority_importance * majority + cls.price_trend_importance * price_trend / cls.revaluation_freq

    def __evaluate_opinion(self):
        cls = type(self)
        chartists_number = len(self.model.get_agents_of_type(cls))
        change_proba = (cls.revaluation_freq * chartists_number /
                        len(self.model.agents) * math.e**(self.__majority_opinion() * (1 if self.is_optimistic else -1)))
        change_proba = min(change_proba, 1.)
        logger.debug(f"Step: {self.model.schedule.steps}. Agent: {self.unique_id}. "
                     f"Optimist: {self.is_optimistic}. Change proba: {round(change_proba, 4)}.")
        self.is_optimistic = RNG.choice([self.is_optimistic, not self.is_optimistic], p=[1 - change_proba, change_proba])

    def _calc_order_quantity(self, price: float | None = None) -> int:
        current_price = price if price else self.model.order_book.get_central_price()
        if current_price <= 0: return 0
        if self.is_optimistic:
            order_qty = min(self.wealth * self.__order_amount_perc, self.cash) // current_price
            if order_qty == 0 and self.cash >= current_price:
                order_qty = 1
        else:
            free_cash = (self.wealth - self._cash_reserved) * 0.95
            order_qty = min(self.wealth * self.__order_amount_perc, free_cash) // current_price
            if order_qty == 0 and free_cash >= current_price:
                order_qty = 1
        return max(int(order_qty), 0)

    def _calc_limit_price(self, order_book: OrderBook | None = None) -> float:
        """
        f(x,lambda_limit,mu_spread) = lambda_limit * e**(-abs(lambda_limit * (x - mu_spread)))
        mu_spread = 1/2 * (best_ask + best_bid)
        """
        cls = type(self)
        order_book: OrderBook = self.model.order_book if not order_book else order_book
        price = order_book.get_central_price() if order_book.get_central_price() else self.model.prices[-1]
        return round(RNG.laplace(price, 1 / cls.lambda_limit), self.model.tick_size)

    def update_open_pos_price(self, action: str, price: float, quantity: int):
        match action:
            case 'buy':
                if self.assets_quantity + quantity == 0:
                    self.__avg_opened_price = 0
                elif self.assets_quantity >= 0:
                    self.__avg_opened_price = ((self.__avg_opened_price * self.assets_quantity + price * quantity) /
                                               (self.assets_quantity + quantity))
                else:
                    # self.__avg_opened_price = (
                    #         (self.__avg_opened_price * abs(self.assets_quantity) - price * quantity) /
                    #         (abs(self.assets_quantity) - quantity))
                    pass
            case 'sell':
                if abs(self.assets_quantity) - quantity == 0:
                    self.__avg_opened_price = 0
                elif self.assets_quantity <= 0:
                    self.__avg_opened_price = ((self.__avg_opened_price * abs(self.assets_quantity) + price * quantity) /
                                               (abs(self.assets_quantity) + quantity))
                else:
                    # self.__avg_opened_price = (
                    #             (self.__avg_opened_price * self.assets_quantity - price * quantity) /
                    #             (self.assets_quantity - quantity))
                    pass
            case _:
                raise ValueError(f'Wrong `action`. Expected "buy" or "sell", got {str(action)}.')

    def step(self):
        order_book: OrderBook = self.model.order_book
        self.__evaluate_opinion()

        best_ask = order_book.get_best_ask()
        if self.assets_quantity < 0 and self.is_optimistic and best_ask:
            order_book.place_order(self.unique_id, MarketAction.BUY, best_ask.price, abs(self.assets_quantity))
            return
        elif best_ask and self.assets_quantity < 0 and self.__avg_opened_price * (1 - self.take_profit) >= best_ask.price:
            order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, best_ask.price, abs(self.assets_quantity))

        best_bid = order_book.get_best_bid()
        if self.assets_quantity > 0 and not self.is_optimistic and best_bid:
            order_book.place_order(self.unique_id, MarketAction.SELL, best_bid.price, self.assets_quantity)
            return
        elif best_bid and self.assets_quantity > 0 and self.__avg_opened_price * (1 + self.take_profit) <= best_bid.price:
            order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, best_bid.price, self.assets_quantity)

        if self.bankrupt: return
        order_price = self._calc_limit_price()
        order_quantity = self._calc_order_quantity(order_price)
        if order_quantity > 0 and self.is_optimistic:
            order_book.place_order(self.unique_id, MarketAction.BUY_LIMIT, order_price, order_quantity)
        elif order_quantity > 0 and not self.is_optimistic:
            order_book.place_order(self.unique_id, MarketAction.SELL_LIMIT, order_price, order_quantity)
        else:
            pass
