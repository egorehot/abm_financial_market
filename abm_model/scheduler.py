import random
import logging

from mesa import Model
from mesa.time import BaseScheduler

from abm_model.market_maker import MarketMaker
from abm_model.chartist import ChartistAgent
from abm_model.fundamentalist import FundamentalistAgent
from utils.models import Transaction, MarketAction
from utils.order_book import OrderBook


class MarketScheduler(BaseScheduler):
    def __init__(self, model: Model):
        super().__init__(model)

    def __complete_transaction(self, transaction: Transaction):
        buyer = self.model.agents.select(filter_func=lambda a: a.unique_id == transaction.buyer_id)[0]
        seller = self.model.agents.select(filter_func=lambda a: a.unique_id == transaction.seller_id)[0]

        buyer.cash -= transaction.price * transaction.quantity
        buyer.assets_quantity += transaction.quantity

        seller.cash += transaction.price * transaction.quantity
        seller.assets_quantity -= transaction.quantity

        self.model.traded_qty += transaction.quantity

    def __mm_fill_order_book(self):
        for mm in self.model.get_agents_of_type(MarketMaker).shuffle():
            mm.fill_order_book()

    def step(self):
        self.model.completed_transactions = 0
        self.model.traded_qty = 0

        traders = []
        for agent_type in [ChartistAgent, FundamentalistAgent]:
            traders.extend(self.model.get_agents_of_type(agent_type))

        order_book: OrderBook = self.model.order_book

        random.shuffle(traders)
        for trader in traders:
            self.__mm_fill_order_book()
            trader.step()
            transactions = order_book.execute_orders()
            for transaction in transactions:
                self.__complete_transaction(transaction)
                self.model.completed_transactions += 1

        for mm in self.model.get_agents_of_type(MarketMaker).shuffle():
            mm.step()
            transactions = order_book.execute_orders()
            for transaction in transactions:
                self.__complete_transaction(transaction)
                self.model.completed_transactions += 1

        self.steps += 1
