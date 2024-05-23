import os
import datetime
import functools
import hashlib
import json
import multiprocessing
from time import time

import config
from abm_model.market_model import MarketModel

logger = config.get_logger(__name__)


def _calc_hash(params):
    params_str = json.dumps(params, sort_keys=True)
    hex_digest = hashlib.md5(params_str.encode()).hexdigest()
    return int(hex_digest, base=16) % 1000


def run_experiment(params: dict, folder_name: str | None = None):
    if not folder_name:
        dir_path = './experiments_data/'
        dttm = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M')
        folder_name = os.path.join(dir_path, dttm)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    params_hash = _calc_hash(params)
    model = MarketModel(**params)
    model.run_model()

    with open(os.path.join(folder_name, f'{params_hash}_params.json'), 'w') as p_file:
        json.dump(params, p_file)

    with open(os.path.join(folder_name, f'{params_hash}_model_data.csv'), 'w') as m_file:
        model.datacollector.get_model_vars_dataframe().to_csv(m_file, index_label='Step')

    with open(os.path.join(folder_name, f'{params_hash}_agents_data.csv'), 'w') as a_file:
        model.datacollector.get_agent_vars_dataframe().to_csv(a_file, index_label='Step')


def run_experiments(params_list: list[dict], experiment_name: str | None = None):
    dir_path = './experiments_data/'
    dttm = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M')
    folder_name = os.path.join(dir_path, dttm + '_' + experiment_name) if experiment_name else os.path.join(dir_path, dttm)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    processes = multiprocessing.cpu_count() // 2 + 1
    pool = multiprocessing.Pool(processes=processes)
    func = functools.partial(run_experiment, folder_name=folder_name)

    logger.info(f"Starting {len(params_list)} experiments. {processes} processes in parallel.")
    start = time()
    pool.map(func, params_list)
    logger.info(f"Experiments finished. Time spent: {round(time() - start)} seconds")


if __name__ == '__main__':
    params = [
        {
            'fundamentalists_number': 1,
            'chartists_number': 0,
            'steps_number': 10,
        },
        {
            'fundamentalists_number': 0,
            'chartists_number': 1,
            'steps_number': 10,
        },
        {
            'fundamentalists_number': 2,
            'chartists_number': 1,
            'steps_number': 10,
        }
    ]

    run_experiments(params, 'test')
