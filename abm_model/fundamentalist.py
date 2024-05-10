import math

from mesa import Model
import numpy as np

from abm_model.market_agent import MarketAgent
from utils.order_book import MarketAction, OrderBook


class FundamentalistAgent(MarketAgent):
    cash_range: list[float] = [10**4, 10**5]
    eps_variance: int = 1
    lambda_limit: int = 3
    chi_market_range: list[float] = [0.01, 0.1]
    chi_opinion_range: list[float] = [0.01, 0.1]
    fundamental_price_range: list[float] = [95, 105]

    def __init__(self, unique_id: int, model: Model, asset_quantity: int = 0):
        cls = type(self)
        cash = np.random.uniform(*cls.cash_range)
        super().__init__(unique_id=unique_id, model=model, cash=cash, asset_quantity=asset_quantity)
        self._fundamental_price: float = np.random.uniform(*cls.fundamental_price_range)
        self._chi_market: float = np.random.uniform(*cls.chi_market_range)
        self._chi_opinion: float = np.random.uniform(*cls.chi_opinion_range)

    def _calc_fundamental_price(self, order_book: OrderBook) -> float:
        """
        ln(p_t) - ln(p_{t-1}) = eps, where eps ~ N(0, eps_variance)
        :return:
        """
        cls = type(self)
        eps_fundamental = np.random.normal(0, cls.eps_variance)
        self._fundamental_price = self._fundamental_price * math.e**eps_fundamental
        self._fundamental_price = self.__adjust_fundamental_price(self._fundamental_price, order_book)
        return self._fundamental_price

    def __adjust_fundamental_price(self, price: float, order_book: OrderBook) -> float:
        """
        Herding behavior. If estimated fundamental price differs "too large" then adjust the price.
        :param price:
        :param order_book:
        :return:
        """
        if self._chi_opinion < abs(1 - price / order_book.get_mu_spread()):
            if price >= order_book.get_mu_spread():
                price = order_book.get_mu_spread() * (1 + self._chi_opinion)
            else:
                price = order_book.get_mu_spread() * (1 - self._chi_opinion)
        return float(price)

    def _calc_limit_price(self, order_book: OrderBook) -> float:
        """
        f(x,lambda_limit,mu_spread) = lambda_limit * e**(-abs(lambda_limit*(x-mu_spread)))
        mu_spread = 1/2 * (best_ask + best_bid)
        :param order_book:
        :return:
        """
        cls = type(self)
        return np.random.laplace(order_book.get_mu_spread(), 1 / cls.lambda_limit)

    def _calc_order_quantity(self, price: float) -> int:
        if self._cash < price: return 0
        return max(int(0.5 * self._cash // price), 1) # TODO

    def _intention(self, order_book: OrderBook) -> MarketAction:
        fundamental_price = self._calc_fundamental_price(order_book)
        best_ask = order_book.get_best_ask().price
        best_bid = order_book.get_best_bid().price
        if fundamental_price > best_ask * (1 + self._chi_market):
            return MarketAction.BUY
        elif best_ask < fundamental_price <= best_ask * (1 + self._chi_market):
            return MarketAction.BUY_LIMIT
        elif fundamental_price < best_bid * (1 - self._chi_market):
            return MarketAction.SELL
        elif best_bid * (1 - self._chi_market) <= fundamental_price < best_bid:
            return MarketAction.SELL_LIMIT
        else:
            return MarketAction.ABSTAIN

    def step(self):
        order_book: OrderBook = self.model.order_book
        intention = self._intention(order_book)
        match intention:
            case MarketAction.BUY | MarketAction.SELL:
                order_quantity = self._calc_order_quantity(self._fundamental_price)
                order_book.place_order(agent_id=self.unique_id, action=intention, price=0, quantity=order_quantity)
            case MarketAction.BUY_LIMIT | MarketAction.SELL_LIMIT:
                limit_price = self._calc_limit_price(order_book)
                order_quantity = self._calc_order_quantity(limit_price)
                order_book.place_order(agent_id=self.unique_id, action=intention,
                                       price=limit_price, quantity=order_quantity)
            case _:
                return


