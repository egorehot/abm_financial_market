import logging

RANDOM_SEED: int | None = 42

LOGGING_LEVEL = logging.INFO
LOG_FORMAT = '%(name)s:%(levelname)s:%(message)s'


def get_logger(name, level=None):
    logger = logging.getLogger(name)
    logger.setLevel(level if level else LOGGING_LEVEL)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)
    ch.setLevel(level if level else LOGGING_LEVEL)
    logger.addHandler(ch)
    return logger
