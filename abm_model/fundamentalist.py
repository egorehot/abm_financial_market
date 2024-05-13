import logging
import math

from mesa import Model
import numpy as np

from abm_model.market_agent import MarketAgent
from utils.order_book import MarketAction, OrderBook


class FundamentalistAgent(MarketAgent):
    """
    _distr: [mu, sigma] lognormal
    _range: [min, max] uniform
    """
    cash_distr: list[float] = [1., 0.4]
    cash_scale: float = 1000.
    eps_variance: float = 0.05
    lambda_limit: float = 3.
    chi_market_range: list[float] = [0.01, 0.1]
    chi_opinion_range: list[float] = [0.05, 0.15]
    fundamental_price_spread: float = 0.05
    risk_range: list[float] = [0.6, 4]

    def __init__(self, unique_id: int, model: Model, cash: float | None = None, assets_quantity: int | None = None):
        cls = type(self)
        cash = np.random.lognormal(*cls.cash_distr) * cls.cash_scale if not cash else cash
        super().__init__(unique_id=unique_id, model=model, cash=cash, assets_quantity=assets_quantity)
        self._fundamental_price = np.random.uniform(
            low=self.model.prices[-1] * (1 - cls.fundamental_price_spread),
            high=self.model.prices[-1] * (1 + cls.fundamental_price_spread),
        )
        self._chi_market = np.random.uniform(*cls.chi_market_range)
        self._chi_opinion = np.random.uniform(*cls.chi_opinion_range)
        self._risk_aversion = np.random.uniform(*cls.risk_range)

    def _calc_fundamental_price(self) -> float:
        """
        ln(p_t) - ln(p_{t-1}) = eps, where eps ~ N(0, eps_variance)
        """
        cls = type(self)
        eps_fundamental = np.random.normal(0, cls.eps_variance)
        self._fundamental_price = self._fundamental_price * math.e**eps_fundamental
        self._fundamental_price = self.__adjust_fundamental_price(self._fundamental_price)
        return self._fundamental_price

    def __adjust_fundamental_price(self, price: float) -> float:
        """
        Herding behavior. If estimated fundamental price differs "too large" then adjust the price.
        """
        order_book: OrderBook = self.model.order_book
        if self._chi_opinion < abs(1 - price / order_book.get_central_price()):
            if price >= order_book.get_central_price():
                price = order_book.get_central_price() * (1 + self._chi_opinion)
            else:
                price = order_book.get_central_price() * (1 - self._chi_opinion)
        return float(price)

    def _calc_limit_price(self, order_book: OrderBook | None = None) -> float:
        """
        f(x,lambda_limit,mu_spread) = lambda_limit * e**(-abs(lambda_limit * (x - mu_spread)))
        mu_spread = 1/2 * (best_ask + best_bid)
        """
        cls = type(self)
        order_book: OrderBook = self.model.order_book if not order_book else order_book
        return round(np.random.laplace(order_book.get_central_price(), 1 / cls.lambda_limit), self.model.tick_size)

    def _calc_order_quantity(self, intention: MarketAction, price: float | None = None) -> int:
        """
        Calcualtes order quantity based on exponential utility function.
        U(W) = -exp^-{a * W}
        W = E(returns) * Q - risk_aversion * volatility * Q^2
        dU/dQ = 0
        Q = returns / (2 * risk_aversion * volatility)
        """
        current_price = price if price else self.model.order_book.get_central_price()
        if current_price <= 0: return 0

        expected_return = (self._fundamental_price - current_price) / current_price
        log_volatility = self.model.log_returns.var(ddof=1) if len(self.model.log_returns) > 2 else 0.007
        optimal_quantity = abs(expected_return / (2 * self._risk_aversion * log_volatility))
        if intention.value > 0:
            return int(min(optimal_quantity, self.cash // current_price))
        elif intention.value < 0:
            return int(min(optimal_quantity, self.assets_quantity + self.cash // current_price))
        return 0

    def _intention(self) -> MarketAction:
        fundamental_price = self._calc_fundamental_price()

        order_book: OrderBook = self.model.order_book
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
        intention = self._intention()
        match intention:  # TODO отменять противоположные ордеры
            case MarketAction.BUY | MarketAction.SELL:
                order_quantity = self._calc_order_quantity(intention)
                if order_quantity > 0:
                    order_book.place_order(agent_id=self.unique_id, action=intention, price=0, quantity=order_quantity)
            case MarketAction.BUY_LIMIT | MarketAction.SELL_LIMIT:
                limit_price = self._calc_limit_price()
                order_quantity = self._calc_order_quantity(intention, limit_price)
                if order_quantity > 0:
                    order_book.place_order(agent_id=self.unique_id, action=intention,
                                           price=limit_price, quantity=order_quantity)
            case _:
                return
