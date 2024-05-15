from mesa import Model
import numpy as np

import config
from abm_model.market_agent import MarketAgent
from utils.order_book import MarketAction, OrderBook

logger = config.get_logger(__name__, 10)

logger.debug(f'Seed: {config.RANDOM_SEED}')
RNG = np.random.default_rng(config.RANDOM_SEED)


class FundamentalistAgent(MarketAgent):
    """
    _distr: [mu, sigma] lognormal
    _range: [min, max] uniform
    """
    cash_distr: list[float] = [1., 0.4]
    cash_scale: float = 1000.
    lambda_limit: float = 3.5
    chi_market_range: list[float] = [0.01, 0.1]
    chi_opinion_range: list[float] = [0.03, 0.1]
    fundamental_price_spread: float = 0.03
    fundamental_price_variance: float = 0.2
    order_amount_range: list[float] = [0.05, 0.10]

    def __init__(self, unique_id: int, model: Model, cash: float | None = None, assets_quantity: int | None = None):
        cls = type(self)
        cash = RNG.lognormal(*cls.cash_distr) * cls.cash_scale if not cash else cash
        super().__init__(unique_id=unique_id, model=model, cash=cash, assets_quantity=assets_quantity)
        self._fundamental_price = RNG.uniform(
            low=self.model.prices[-1] * (1 - cls.fundamental_price_spread),
            high=self.model.prices[-1] * (1 + cls.fundamental_price_spread),
        )
        self._chi_market = RNG.uniform(*cls.chi_market_range)
        self._chi_opinion = RNG.uniform(*cls.chi_opinion_range)
        self.__order_amount_perc = RNG.uniform(*cls.order_amount_range)

    def _calc_fundamental_price(self) -> float:
        """
        ln(p_t) - ln(p_{t-1}) = eps, where eps ~ N(0, eps_variance)
        """
        if self.model.news_event_occurred:
            cls = type(self)
            self._fundamental_price += RNG.normal(self.model._news_event_value, cls.fundamental_price_variance)
            self._fundamental_price = self.__adjust_fundamental_price(self._fundamental_price)
            logger.debug(f'Agent: {self.unique_id}. New fundamental price: {round(self._fundamental_price, 3)}.')
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
        return round(RNG.laplace(order_book.get_central_price(), 1 / cls.lambda_limit), self.model.tick_size)

    def _calc_order_quantity(self, intention: MarketAction, price: float | None = None) -> int:
        current_price = price if price else self.model.order_book.get_central_price()
        if current_price <= 0: return 0
        if intention.value > 0:
            order_qty = min(self.wealth * self.__order_amount_perc, self.cash) // current_price
            if order_qty == 0 and self.cash >= current_price:
                order_qty = 1
            order_qty += abs(self.assets_quantity) if self.assets_quantity < 0 else 0
        elif intention.value < 0:
            free_cash = (self.wealth - self._cash_reserved) * 0.95
            order_qty = min(self.wealth * self.__order_amount_perc, free_cash) // current_price
            if order_qty == 0 and free_cash >= current_price:
                order_qty = 1
            order_qty += self.assets_quantity if self.assets_quantity > 0 else 0
        else:
            raise ValueError(f'Wrong `MarketAction`. Got {intention}')
        return max(int(order_qty), 0)

    def _intention(self) -> MarketAction:
        fundamental_price = self._fundamental_price
        order_book: OrderBook = self.model.order_book
        best_ask = order_book.get_best_ask()
        best_bid = order_book.get_best_bid()
        if fundamental_price > best_ask.price * (1 + self._chi_market):
            return MarketAction.BUY
        elif best_ask.price < fundamental_price <= best_ask.price * (1 + self._chi_market):
            return MarketAction.BUY_LIMIT
        elif fundamental_price < best_bid.price * (1 - self._chi_market):
            return MarketAction.SELL
        elif best_bid.price * (1 - self._chi_market) <= fundamental_price < best_bid.price:
            return MarketAction.SELL_LIMIT
        else:
            return MarketAction.ABSTAIN

    def step(self):
        if self.bankrupt: return

        fundamental_price = self._calc_fundamental_price()
        order_book: OrderBook = self.model.order_book

        cancel_side = 'ask' if fundamental_price > order_book.get_central_price() else 'bid'
        order_book.cancel_limit_orders(self.unique_id, cancel_side)

        intention = self._intention()
        match intention:
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
                pass
        return
