def round_to_tick(price: float, tick_size: float):
    return round(round(price / tick_size) * tick_size, 4)
