import pytest
from order_book import OrderBook, Order
from models import MarketAction


@pytest.fixture
def empty_order_book():
    return OrderBook()


@pytest.fixture
def filled_order_book():
    ob = OrderBook()
    ob.place_order('agent1', MarketAction.BUY_LIMIT, 100.0, 10)
    ob.place_order('agent2', MarketAction.BUY_LIMIT, 101.0, 15)
    ob.place_order('agent3', MarketAction.SELL_LIMIT, 103.0, 5)
    ob.place_order('agent4', MarketAction.SELL_LIMIT, 104.0, 8)
    return ob


def test_initialization(empty_order_book):
    assert len(empty_order_book.bid) == 0
    assert len(empty_order_book.ask) == 0
    assert len(empty_order_book.market_orders) == 0


def test_place_order(empty_order_book):
    empty_order_book.place_order('agent1', MarketAction.BUY_LIMIT, 100.0, 10)
    assert len(empty_order_book.bid) == 1
    assert empty_order_book.bid[0].price == 100.0

    empty_order_book.place_order('agent2', MarketAction.SELL_LIMIT, 101.0, 5)
    assert len(empty_order_book.ask) == 1
    assert empty_order_book.ask[0].price == 101.0

    empty_order_book.place_order('agent3', MarketAction.BUY, 99.0, 5)
    assert len(empty_order_book.market_orders) == 1
    assert empty_order_book.market_orders[0].price == 99.0

    empty_order_book.place_order('agent4', MarketAction.SELL, 102.0, 5)
    assert len(empty_order_book.market_orders) == 2
    assert empty_order_book.market_orders[1].price == 102.0


def test_get_best_bid_and_ask(filled_order_book):
    assert filled_order_book.get_best_bid().price == 101.0
    assert filled_order_book.get_best_ask().price == 103.0


def test_get_mu_spread(filled_order_book):
    assert filled_order_book.get_mu_spread() == 102.0


def test_invalid_order(empty_order_book):
    with pytest.raises(ValueError):
        empty_order_book.place_order('agent1', 'INVALID_ACTION', 100.0, 10)


# EXECUTION
def test_buy_market_orders():
    ob = OrderBook()
    ob.place_order('agent1', MarketAction.BUY, 0, 10)
    ob.place_order('agent2', MarketAction.SELL_LIMIT, 101.0, 5)
    ob.place_order('agent3', MarketAction.SELL_LIMIT, 101.5, 5)
    ob.place_order('agent4', MarketAction.SELL_LIMIT, 102.5, 1)

    transaction = ob.execute_orders()
    assert len(transaction) == 2
    assert transaction[0] == {'buyer_id': 'agent1', 'seller_id': 'agent2', 'price': 101.0, 'quantity': 5}
    assert transaction[1] == {'buyer_id': 'agent1', 'seller_id': 'agent3', 'price': 101.5, 'quantity': 5}
    assert len(ob.market_orders) == 0
    assert len(ob.ask) == 1
    assert ob.ask[0] == Order('agent4', MarketAction.SELL_LIMIT, 102.5, 1)


def test_sell_market_orders():
    ob = OrderBook()
    ob.place_order('agent1', MarketAction.SELL, 0, 10)
    ob.place_order('agent2', MarketAction.BUY_LIMIT, 102.0, 5)
    ob.place_order('agent3', MarketAction.BUY_LIMIT, 101.5, 5)
    ob.place_order('agent4', MarketAction.BUY_LIMIT, 101.0, 1)

    transaction = ob.execute_orders()
    assert len(transaction) == 2
    assert transaction[0] == {'buyer_id': 'agent2', 'seller_id': 'agent1', 'price': 102.0, 'quantity': 5}
    assert transaction[1] == {'buyer_id': 'agent3', 'seller_id': 'agent1', 'price': 101.5, 'quantity': 5}
    assert len(ob.market_orders) == 0
    assert len(ob.bid) == 1
    assert ob.bid[0] == Order('agent4', MarketAction.BUY_LIMIT, 101.0, 1)


def test_non_matching_orders(filled_order_book):
    transactions = filled_order_book.execute_orders()
    assert len(transactions) == 0
    assert len(filled_order_book) == 4


def test_partial_buy_market_execution():
    ob = OrderBook()
    ob.place_order('agent1', MarketAction.BUY, 0.0, 12)
    ob.place_order('agent2', MarketAction.SELL_LIMIT, 101.0, 5)
    ob.place_order('agent3', MarketAction.SELL_LIMIT, 101.5, 5)
    ob.place_order('agent4', MarketAction.SELL_LIMIT, 102.5, 1)

    transaction = ob.execute_orders()
    assert len(transaction) == 3
    assert transaction[0] == {'buyer_id': 'agent1', 'seller_id': 'agent2', 'price': 101.0, 'quantity': 5}
    assert transaction[1] == {'buyer_id': 'agent1', 'seller_id': 'agent3', 'price': 101.5, 'quantity': 5}
    assert transaction[2] == {'buyer_id': 'agent1', 'seller_id': 'agent4', 'price': 102.5, 'quantity': 1}
    assert len(ob.market_orders) == 0
    assert len(ob.ask) == 0


def test_partial_buy_limit_execution():
    ob = OrderBook()
    ob.place_order('agent2', MarketAction.SELL_LIMIT, 101.0, 5)
    ob.place_order('agent1', MarketAction.BUY_LIMIT, 102.0, 12)
    ob.place_order('agent3', MarketAction.SELL_LIMIT, 101.5, 5)
    ob.place_order('agent4', MarketAction.SELL_LIMIT, 102.5, 1)

    transaction = ob.execute_orders()
    assert len(transaction) == 2
    assert transaction[0] == {'buyer_id': 'agent1', 'seller_id': 'agent2', 'price': 101.0, 'quantity': 5}
    assert transaction[1] == {'buyer_id': 'agent1', 'seller_id': 'agent3', 'price': 102.0, 'quantity': 5}
    assert len(ob.bid) == 1
    assert ob.bid[0].price == 102.0
    assert ob.bid[0].quantity == 2
    assert len(ob.ask) == 1
    assert ob.ask[0].price == 102.5
    assert ob.ask[0].quantity == 1


def test_partial_sell_market_execution():
    ob = OrderBook()
    ob.place_order('agent1', MarketAction.SELL, 0, 12)
    ob.place_order('agent2', MarketAction.BUY_LIMIT, 102.0, 5)
    ob.place_order('agent3', MarketAction.BUY_LIMIT, 101.5, 5)
    ob.place_order('agent4', MarketAction.BUY_LIMIT, 101.0, 1)

    transaction = ob.execute_orders()
    assert len(transaction) == 3
    assert transaction[0] == {'buyer_id': 'agent2', 'seller_id': 'agent1', 'price': 102.0, 'quantity': 5}
    assert transaction[1] == {'buyer_id': 'agent3', 'seller_id': 'agent1', 'price': 101.5, 'quantity': 5}
    assert transaction[2] == {'buyer_id': 'agent4', 'seller_id': 'agent1', 'price': 101.0, 'quantity': 1}
    assert len(ob.market_orders) == 0
    assert len(ob.bid) == 0


def test_partial_sell_limit_execution():
    ob = OrderBook()
    ob.place_order('agent2', MarketAction.BUY_LIMIT, 102.0, 5)
    ob.place_order('agent1', MarketAction.SELL_LIMIT, 101.5, 12)
    ob.place_order('agent3', MarketAction.BUY_LIMIT, 101.5, 5)
    ob.place_order('agent4', MarketAction.BUY_LIMIT, 101.0, 1)

    transaction = ob.execute_orders()
    assert len(transaction) == 2
    assert transaction[0] == {'buyer_id': 'agent2', 'seller_id': 'agent1', 'price': 102.0, 'quantity': 5}
    assert transaction[1] == {'buyer_id': 'agent3', 'seller_id': 'agent1', 'price': 101.5, 'quantity': 5}
    assert len(ob.ask) == 1
    assert ob.ask[0].price == 101.5
    assert ob.ask[0].quantity == 2
    assert len(ob.bid) == 1
    assert ob.bid[0].price == 101.0
    assert ob.bid[0].quantity == 1
