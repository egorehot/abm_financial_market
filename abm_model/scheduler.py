from mesa import Model
from mesa.time import BaseScheduler
import numpy as np

import config
from abm_model.chartist import ChartistAgent
from abm_model.fundamentalist import FundamentalistAgent
from abm_model.market_maker import MarketMaker
from abm_model.news import NewsAgent
from utils.models import Transaction
from utils.order_book import OrderBook

logger = config.get_logger(__name__)


class MarketScheduler(BaseScheduler):
    news_lambda: float = 0.5

    def __init__(self, model: Model, seed: int | None = None):
        super().__init__(model)
        self._news_lambda = type(self).news_lambda
        self.RNG = np.random.default_rng(seed if seed else config.RANDOM_SEED)
        self.news_event_step = round(self.RNG.exponential(1 / self._news_lambda))

    def __complete_transaction(self, transaction: Transaction):
        buyer = self.model.agents.select(filter_func=lambda a: a.unique_id == transaction.buyer_id)[0]
        seller = self.model.agents.select(filter_func=lambda a: a.unique_id == transaction.seller_id)[0]

        if isinstance(buyer, ChartistAgent):
            buyer.update_open_pos_price('buy', transaction.price, transaction.quantity)
        buyer.cash -= transaction.price * transaction.quantity
        buyer.assets_quantity += transaction.quantity

        if isinstance(seller, ChartistAgent):
            seller.update_open_pos_price('sell', transaction.price, transaction.quantity)
        seller.cash += transaction.price * transaction.quantity
        seller.assets_quantity -= transaction.quantity

        self.model.traded_qty += transaction.quantity

    def __execute_order_book(self):
        order_book: OrderBook = self.model.order_book
        transactions = order_book.execute_orders()
        ttl_amount = ttl_qty = 0
        for transaction in transactions:
            self.__complete_transaction(transaction)
            ttl_amount += transaction.quantity * transaction.price
            ttl_qty += transaction.quantity
            self.model.completed_transactions += 1
        return (ttl_amount / ttl_qty) if transactions else 0

    def __mm_step(self):
        for mm in self.model.get_agents_of_type(MarketMaker).shuffle():
            mm.step()

    def __generate_news_event(self):
        self.news_event_step = self.steps + np.ceil(self.RNG.exponential(1 / self._news_lambda))
        for news_agent in self.model.get_agents_of_type(NewsAgent):
            news_agent.step()

    def step(self):
        logger.debug(f'Step #{self.steps} starts.')
        self.model.completed_transactions = 0
        self.model.traded_qty = 0

        self.model.news_event_occurred = False
        if self.steps == self.news_event_step:
            self.__generate_news_event()

        traders = []
        for agent_type in [ChartistAgent, FundamentalistAgent]:
            traders.extend(self.model.get_agents_of_type(agent_type))

        self.RNG.shuffle(traders)
        for trader in traders:
            if not all([self.model.order_book.get_best_ask(), self.model.order_book.get_best_bid()]):
                self.__mm_step()
            trader.step()
            self.__execute_order_book()

        self.__mm_step()
        avg_price = self.__execute_order_book()
        market_price = self.model.order_book.get_central_price() if self.model.order_book.get_central_price() else avg_price
        if market_price <= 0:
            logger.error(f"Wrong `market_price` {market_price}. Step: {self.steps}\n"
                         f"Market maker: {self.model.get_agents_of_type(MarketMaker)[0]}\n"
                         f"Order book: {self.model.order_book}")
            raise ValueError(f"Wrong `market_price` {market_price}. Step: {self.steps}.")
        self.model.prices.append(round(market_price, 4))

        logger.debug(f'Step #{self.steps} finished.')
        self.steps += 1
