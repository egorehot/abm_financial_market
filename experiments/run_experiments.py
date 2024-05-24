import os
import datetime
import functools
import hashlib
import json
import multiprocessing
from time import time

import config
from abm_model.market_model import MarketModel
from abm_model.market_agent import MarketAgent
from experiments.mertics import calculate_metrics

logger = config.get_logger(__name__)


def _floor_minutes_to_5(dt):
    year, month, day, hour, minute = dt.year, dt.month, dt.day, dt.hour, dt.minute
    floored_minute = (minute // 5) * 5
    return datetime.datetime(year, month, day, hour, floored_minute, 0, 0)


def _calc_hash(params):
    params_str = json.dumps(params, sort_keys=True)
    hex_digest = hashlib.md5(params_str.encode()).hexdigest()
    return int(hex_digest, base=16) % 1000


def _change_seed(new_seed):
    config.RANDOM_SEED = new_seed
    MarketAgent.update_rng(new_seed)


def run_experiment(params: dict, folder_name: str | None = None):
    if not folder_name:
        dir_path = 'experiments_data'
        dttm = _floor_minutes_to_5(datetime.datetime.utcnow()).strftime('%Y%m%dT%H%M')
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

    return int(params_hash), calculate_metrics(model.datacollector.get_model_vars_dataframe())


def run_experiments(params_list: list[dict], experiment_name: str | None = None):
    dir_path = 'experiments_data'
    dttm = _floor_minutes_to_5(datetime.datetime.utcnow()).strftime('%Y%m%dT%H%M')
    folder_name = os.path.join(dir_path, dttm + '_' + experiment_name) if experiment_name else os.path.join(dir_path, dttm)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # processes = min(multiprocessing.cpu_count() // 2, len(params_list))
    # pool = multiprocessing.Pool(processes=processes)
    # func = functools.partial(run_experiment, folder_name=folder_name)

    seed = config.SEEDS.index(config.RANDOM_SEED)
    logger.info(f"Starting {len(params_list)} experiments. Seed: {seed}")
    start = time()
    # results = pool.map(func, params_list)
    results = []
    for param in params_list:
        run_experiment(param, folder_name=folder_name)
    logger.info(f"Experiments finished. Time spent: {round(time() - start)} seconds")

    for result in results:
        result[1].update(Seed=seed)

    data = []
    if os.path.exists(os.path.join(folder_name, 'metrics.json')):
        with open(os.path.join(folder_name, 'metrics.json'), 'r') as r_file:
            data = json.load(r_file)
    data.append(dict(results))
    with open(os.path.join(folder_name, 'metrics.json'), 'w') as w_file:
        json.dump(data, w_file)


if __name__ == '__main__':
    import copy
    # from time import sleep
    # while True:
    #     now = datetime.datetime.now()
    #     print(f'Time: {now.strftime("%H:%M:%S")}.', end=' ')
    #     if now.minute % 5 == 0:
    #         print('Starting experiments!')
    #         break
    #     print('Sleeping...')
    #     sleep(20)

    tick_sizes = [0.01, 0.05, 0.1, 0.2, 0.5]
    param = {
        'fundamentalists_number': 100,
        'chartists_number': 100,
        'steps_number': 252,
    }
    params = []
    for tick_size in tick_sizes:
        new_param = copy.deepcopy(param)
        new_param.update(tick_size=tick_size)
        params.append(new_param)

    print(params)
    raise ValueError()

    ttl_experiments = len(params) * len(config.SEEDS)
    print(f"Total number of experiments: {ttl_experiments}")
    tick = time()
    done_exp = 0
    for seed in config.SEEDS:
        _change_seed(seed)
        run_experiments(params, f'seed{config.SEEDS.index(seed)}')
        done_exp += len(params)
        print(f"Done {done_exp}/{ttl_experiments}.")
    print(f'Total time spent: {round(time() - tick)} seconds.')
