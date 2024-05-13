from mesa import Agent, Model


class MarketAgent(Agent):
    def __init__(self, unique_id: int, model: Model, cash: float, assets_quantity: int):
        super().__init__(unique_id=unique_id, model=model)
        self._cash = float(cash)
        self._assets_quantity = int(assets_quantity) if assets_quantity else 0

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
        self._assets_quantity = value

    @property
    def wealth(self):
        return self._cash + self.model.prices[-1] * self._assets_quantity

    @property
    def bankrupt(self):
        return self.wealth <= 0
