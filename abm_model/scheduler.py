import random
import logging

from mesa import Model
from mesa.time import BaseScheduler

from abm_model.market_maker import MarketMaker
from abm_model.chartist import ChartistAgent
from abm_model.fundamentalist import FundamentalistAgent
from utils.models import Transaction
from utils.order_book import OrderBook


class MarketScheduler(BaseScheduler):
    def __init__(self, model: Model):
        super().__init__(model)

    def __complete_transaction(self, transaction: Transaction):
        buyer = self.model.agents.select(filter_func=lambda a: a.unique_id == transaction.buyer_id)[0]
        seller = self.model.agents.select(filter_func=lambda a: a.unique_id == transaction.seller_id)[0]

        buyer.cash -= transaction.price * transaction.quantity
        buyer.asset_quantity += transaction.quantity

        seller.cash += transaction.price * transaction.quantity
        seller.asset_quantity -= transaction.quantity

    def step(self):
        completed_transactions = 0
        traded_qty = 0
        for mm in self.model.get_agents_of_type(MarketMaker):
            mm.fill_order_book()

        traders = []
        for agent_type in [ChartistAgent, FundamentalistAgent]:
            traders.extend(self.model.get_agents_of_type(agent_type).shuffle())

        order_book: OrderBook = self.model.order_book
        random.shuffle(traders)
        for trader in traders:
            trader.step()
            transactions = order_book.execute_orders()
            for transaction in transactions:
                self.__complete_transaction(transaction)
                completed_transactions += 1

        for mm in self.model.get_agents_of_type(MarketMaker):
            mm.step()

        self.model.completed_transactions += completed_transactions
        self.model.traded_qty += traded_qty
        self.steps += 1

