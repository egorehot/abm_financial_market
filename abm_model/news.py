from mesa import Agent, Model
import numpy as np

import config

logger = config.get_logger(__name__)

logger.debug(f'Seed: {config.RANDOM_SEED}')
RNG = np.random.default_rng(config.RANDOM_SEED)


class NewsAgent(Agent):
    mean = 0
    variance = 0.15

    def __init__(self, unique_id: int, model: Model, mean: float | None = None, variance: float | None = None):
        cls = type(self)
        super().__init__(unique_id, model)
        self._mean = mean or cls.mean
        self._variance = variance or cls.variance

    def step(self):
        self.model.news_event_value = RNG.normal(self._mean, self._variance)
        logger.debug(f'Step {self.model.schedule.steps + 1}. News event {round(self.model.news_event_value, 4)}')
