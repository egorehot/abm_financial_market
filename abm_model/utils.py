def round_to_tick(price: float, tick_size: float):
    return max(round(round(price / tick_size) * tick_size, 4), 1e-10)
