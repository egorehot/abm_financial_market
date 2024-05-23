from abm_model.market_model import MarketModel


def run():
    model = MarketModel(
        fundamentalists_number=0,
        chartists_number=1,
        steps_number=30,
        initial_market_price=100.,
    )
    model.run_model()
    print(model.datacollector.get_model_vars_dataframe())


if __name__ == '__main__':
    run()
