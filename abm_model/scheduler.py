import random
import logging

from mesa import Model
from mesa.time import BaseScheduler
import numpy as np

from abm_model.chartist import ChartistAgent
from abm_model.fundamentalist import FundamentalistAgent
from abm_model.market_maker import MarketMaker
from abm_model.news import NewsAgent
from utils.models import Transaction
from utils.order_book import OrderBook

RNG = np.random.default_rng()


class MarketScheduler(BaseScheduler):
    news_lambda: float = 0.5

    def __init__(self, model: Model):
        super().__init__(model)
        self._news_lambda = type(self).news_lambda
        self.news_event_step = round(np.random.exponential(1 / self._news_lambda))

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
        for transaction in transactions:
            self.__complete_transaction(transaction)
            self.model.completed_transactions += 1

    def __mm_fill_order_book(self):
        for mm in self.model.get_agents_of_type(MarketMaker).shuffle():
            mm.fill_order_book()

    def __generate_news_event(self):
        self.news_event_step = self.steps + np.ceil(np.random.exponential(1 / self._news_lambda))
        for news_agent in self.model.get_agents_of_type(NewsAgent):
            news_agent.step()

    def step(self):
        self.model.completed_transactions = 0
        self.model.traded_qty = 0

        self.model.news_event_occurred = False
        if self.steps == self.news_event_step:
            print(f'Step {self.steps}', end=' ')
            self.__generate_news_event()

        traders = []
        for agent_type in [ChartistAgent, FundamentalistAgent]:
            traders.extend(self.model.get_agents_of_type(agent_type))

        random.shuffle(traders)
        for trader in traders:
            self.__mm_fill_order_book()
            trader.step()
            self.__execute_order_book()

        for mm in self.model.get_agents_of_type(MarketMaker).shuffle():
            mm.step()
            self.__execute_order_book()

        self.steps += 1
