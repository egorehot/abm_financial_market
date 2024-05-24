def calc_spread(model_df, absolute: bool = False):
    return round(
        ((model_df['Best ask price'] - model_df['Best bid price']) / (1 if absolute else model_df['Price'])).mean(),
        6)


def calc_price_std(price_series, absolute: bool = False):
    return round(price_series.std() / (1 if absolute else price_series.mean()), 6)


def calc_series_metrics(series, metric: str = 'avg'):
    match metric:
        case 'avg':
            return round(series.mean(), 6)
        case 'sum':
            return round(series.sum(), 6)
        case _:
            raise ValueError(f"Unexpected `metric` {metric}. Expected 'avg' or 'sum'")


def calculate_metrics(model_df, absolute: bool = False):
    return {
        'Spread': calc_spread(model_df, absolute=absolute),
        'Price std': calc_price_std(model_df['Price'], absolute=absolute),
        'Transactions mean': calc_series_metrics(model_df['Transactions'], metric='avg'),
        'Transactions sum': int(calc_series_metrics(model_df['Transactions'], metric='sum')),
        'Volume mean': calc_series_metrics(model_df['Volume'], metric='avg'),
        'Volume sum': int(calc_series_metrics(model_df['Volume'], metric='sum')),
    }
