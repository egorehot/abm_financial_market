from abm_model.market_model import MarketModel


def run():
    model = MarketModel(1, 0, 30, 100)
    model.run_model()


if __name__ == '__main__':
    run()
