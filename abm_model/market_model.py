from functools import partial

from mesa import Model, DataCollector, Agent

import config
from abm_model.chartist import ChartistAgent
from abm_model.fundamentalist import FundamentalistAgent
from abm_model.market_agent import MarketAgent
from abm_model.market_maker import MarketMaker
from abm_model.news import NewsAgent
from abm_model.scheduler import MarketScheduler
from utils.order_book import OrderBook

logger = config.get_logger(__name__)


def _agents_factory(model: Model, agent_type: type[Agent], agents_number: int, class_config: dict | None = None):
    class_config = class_config or {}
    for attr, value in class_config.items():
        setattr(agent_type, attr, value)
    for _ in range(agents_number):
        match agent_type.__name__:
            case 'FundamentalistAgent':
                agent = FundamentalistAgent(unique_id=model.next_id(), model=model)
            case 'ChartistAgent':
                agent = ChartistAgent(unique_id=model.next_id(), model=model)
            case 'MarketMaker':
                agent = MarketMaker(unique_id=model.next_id(), model=model)
            case 'NewsAgent':
                agent = NewsAgent(unique_id=model.next_id(), model=model)
            case _:
                raise ValueError(f'Unknown agent_type. Got {str(agent_type)}')

        model.schedule.add(agent)


def get_type_attr_ttl(model: Model, agent_type: MarketAgent, attr: str):
    return sum(model.get_agents_of_type(agent_type).get(attr))


class MarketModel(Model):
    def __init__(
            self,
            *,
            fundamentalists_number: int,
            chartists_number: int,
            steps_number: int,
            initial_market_price: float = 100.,
            tick_size: int = 2,
            fundamentalists_config: dict | None = None,
            chartists_config: dict | None = None,
    ):
        logger.info('Initializing model.')
        super().__init__()
        self.running = True
        if steps_number <= 0:
            raise ValueError(f"`steps_number` must be >0. Got {str(steps_number)}")
        self.__steps_number = int(steps_number)
        self.schedule = MarketScheduler(self)
        self.order_book = OrderBook()

        self.prices = [initial_market_price]
        self._optimistic_chartists_number = 0
        self.completed_transactions = 0
        self.traded_qty = 0
        self.tick_size = tick_size
        self._news_event_value: float = 0.
        self.news_event_occurred = False

        self.datacollector = DataCollector(
            model_reporters={
                'Price': lambda model: model.prices[-1],
                'Transactions': 'completed_transactions',
                'Volume': 'traded_qty',
                'MM total wealth': partial(get_type_attr_ttl, agent_type=MarketMaker, attr='wealth'),
                'MM total cash': partial(get_type_attr_ttl, agent_type=MarketMaker, attr='cash'),
                'MM total assets': partial(get_type_attr_ttl, agent_type=MarketMaker, attr='assets_quantity'),
                'News occurred': 'news_event_occurred',
                'Fundamentalists total wealth': partial(get_type_attr_ttl, agent_type=FundamentalistAgent, attr='wealth'),
                'Fundamentalists total cash': partial(get_type_attr_ttl, agent_type=FundamentalistAgent, attr='cash'),
                'Fundamentalists total assets': partial(get_type_attr_ttl, agent_type=FundamentalistAgent, attr='assets_quantity'),
                'Optimists': '_optimistic_chartists_number',
                'Chartists total wealth': partial(get_type_attr_ttl, agent_type=ChartistAgent, attr='wealth'),
                'Chartists total cash': partial(get_type_attr_ttl, agent_type=ChartistAgent, attr='cash'),
                'Chartists total assets': partial(get_type_attr_ttl, agent_type=ChartistAgent, attr='assets_quantity'),
            },
            agent_reporters={
                'Type': lambda a: type(a).__name__,
                'Wealth': 'wealth',
                'Is bankrupt': 'bankrupt',
                'Fundamental prices': '_fundamental_price',
            }
        )

        _agents_factory(self, NewsAgent, 1)
        _agents_factory(self, MarketMaker, 1)
        _agents_factory(self, FundamentalistAgent, fundamentalists_number, fundamentalists_config)
        _agents_factory(self, ChartistAgent, chartists_number, chartists_config)

        logger.info('Model initialized.')
        log_agents = {t.__name__: len(l) for t, l in self.agents_.items()}
        logger.debug(f'Agents: {log_agents}')

    # @property
    # def log_returns(self) -> np.array:
    #     if len(self.prices) < 2: return np.array([0])
    #     return np.log(np.array(self.prices)[1:] / np.array(self.prices)[:-1])

    @property
    def news_event_value(self):
        return self._news_event_value

    @news_event_value.setter
    def news_event_value(self, value: float):
        self.news_event_occurred = True
        self._news_event_value = float(value)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.steps == self.__steps_number:
            self.running = False

    def run_model(self) -> None:
        while self.running:
            self.step()
        self.datacollector.collect(self)
