from mesa import Agent, Model


class MarketAgent(Agent):
    def __init__(self, unique_id: int, model: Model, cash: float, asset_quantity: int = 0):
        super().__init__(unique_id=unique_id, model=model)
        self._cash = float(cash)
        self._asset_quantity = int(asset_quantity)

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value: float):
        self._cash = float(value)

    @property
    def asset_quantity(self):
        return self._asset_quantity

    @asset_quantity.setter
    def asset_quantity(self, value):
        self._asset_quantity = value

    @property
    def wealth(self):
        return self._cash + self.model.prices[-1] * self._asset_quantity

