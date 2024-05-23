import numpy as np
from mesa import Agent, Model

import config
from abm_model.utils import round_to_tick
from utils.order_book import OrderBook


logger = config.get_logger(__name__)

logger.debug(f'Seed: {config.RANDOM_SEED}')


class MarketAgent(Agent):
    RNG = np.random.default_rng(config.RANDOM_SEED)
    lambda_limit: float = 3.5

    def __init__(self, unique_id: int, model: Model, cash: float, assets_quantity: int):
        super().__init__(unique_id=unique_id, model=model)
        self._cash = float(cash)
        self._assets_quantity = int(assets_quantity) if assets_quantity else 0
        self._cash_reserved = 0

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value: float):
        self._cash = float(value)

    @property
    def assets_quantity(self):
        return self._assets_quantity

    @assets_quantity.setter
    def assets_quantity(self, value):
        if value < 0 and self._assets_quantity <= 0:
            short = self._assets_quantity - value
            self._cash_reserved += self.model.prices[-1] * short
        elif value < 0 and self._assets_quantity > 0:
            short = abs(value)
            self._cash_reserved += self.model.prices[-1] * short
        else:
            self._cash_reserved = 0
        self._assets_quantity = value

    @property
    def wealth(self):
        return self._cash + self.model.prices[-1] * self._assets_quantity

    @property
    def bankrupt(self):
        return self.wealth <= 0

    def _calc_limit_price(self, order_book: OrderBook | None = None) -> float:
        """
        f(x,lambda_limit,mu_spread) = lambda_limit * e**(-abs(lambda_limit * (x - mu_spread)))
        mu_spread = 1/2 * (best_ask + best_bid)
        """
        cls = type(self)
        order_book: OrderBook = self.model.order_book if not order_book else order_book
        price = order_book.get_central_price() if order_book.get_central_price() else self.model.prices[-1]
        return round_to_tick(cls.RNG.laplace(price, 1 / cls.lambda_limit), self.model.tick_size)
