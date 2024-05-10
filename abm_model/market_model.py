import logging

from mesa import Model, DataCollector

from abm_model.chartist import ChartistAgent
from abm_model.fundamentalist import FundamentalistAgent
from abm_model.market_agent import MarketAgent
from abm_model.market_maker import MarketMaker
from abm_model.scheduler import MarketScheduler
from utils.order_book import OrderBook


logger = logging.getLogger('MARKET_MODEL')
logger.setLevel(logging.DEBUG)


def _agents_factory(model: Model, agent_type: type[MarketAgent], agents_number: int, class_config: dict | None = None):
    class_config = class_config or {}
    for attr, value in class_config.items():
        setattr(agent_type, attr, value)
    for _ in range(agents_number):
        match agent_type.__name__:
            case 'FundamentalistAgent':
                agent = FundamentalistAgent(
                    unique_id=model.next_id(),
                    model=model,
                )
            case 'ChartistAgent':
                agent = ChartistAgent(
                    unique_id=model.next_id(),
                    model=model,
                )
            case 'MarketMaker':
                agent = MarketMaker(
                    unique_id=model.next_id(),
                    model=model,
                )
            case _:
                raise ValueError(f'Unknown agent_type. Got {str(agent_type)}')

        model.schedule.add(agent)


class MarketModel(Model):
    def __init__(
            self,
            fundamentalists_number: int,
            chartists_number: int,
            steps_number: int,
            initial_market_price: float,
            fundamentalists_config: dict | None = None,
            chartists_config: dict | None = None,
    ):
        logging.info('Initializing model.')
        super().__init__()
        self.running = True
        self.schedule = MarketScheduler(self)
        self.order_book: OrderBook = OrderBook()
        self.prices = [initial_market_price]
        self.datacollector = DataCollector(
            model_reporters={'Transactions': 'completed_transactions', 'Traded quantity': 'traded_qty'},
            agent_reporters={'Fundamental prices': '_fundamental_price'}
        )
        self.completed_transactions = 0
        self.traded_qty = 0
        self._optimistic_chartists_number = 0
        self.__steps_number = steps_number

        _agents_factory(self, MarketMaker, 1)
        _agents_factory(self, FundamentalistAgent, fundamentalists_number, fundamentalists_config)
        _agents_factory(self, ChartistAgent, chartists_number, chartists_config)

        logger.info('Model initialized.')
        log_agents = {t.__name__: len(l) for t, l in self.agents_.items()}
        logger.debug(f'Agents: {log_agents}')

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.steps == self.__steps_number:
            self.running = False
        logger.debug(f'Step {self.schedule.steps} finished.')

    def run_model(self) -> None:
        while self.running:
            self.step()
        # logger.debug(f'{self.datacollector.get_agent_vars_dataframe().groupby("Step").mean()}')
        logger.debug(f'{self.datacollector.get_model_vars_dataframe()}')
