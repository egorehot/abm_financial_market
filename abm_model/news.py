from mesa import Agent, Model
import numpy as np


class NewsAgent(Agent):
    mean = 0
    variance = 1

    def __init__(self, unique_id: int, model: Model, mean: float | None = None, variance: float | None = None):
        cls = type(self)
        super().__init__(unique_id, model)
        self._mean = mean or cls.mean
        self._variance = variance or cls.variance

    def step(self):
        self.model.news_event_value = np.random.normal(self._mean, self._variance)
        print(f'News event {round(self.model.news_event_value, 4)}')
