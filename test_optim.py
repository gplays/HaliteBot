import json
import math
import os
import zstd
from multiprocessing import Pool
from os import path

import numpy as np
from scipy import optimize as opt

SCORE_WEIGHTS = {"winner": 0.5,
                 "time_efficiency": 0.125,
                 "damage_per_turn": 0.125,
                 "ratio_prod": 0.125,
                 "ratio_damage": 0.125, }
STORE_PATH = ".data"


def func(x):
    print(x[0])
    return 2 * math.log(x[0] ** 2) ** 2 + 3 * x[0] + 1


def clean_repository():
    onlyfiles = [f for f in os.listdir(".") if f.endswith(".log")]
    for file in onlyfiles:
        os.remove(file)


def async_score(args):
    pool = Pool()
    iterables = (args) * 10
    pool.starmap(launch_gym, iterables)
    pool.close()
    pool.join()

    score = handle_replays(args)

    return score


def update_session_id():
    with open("last_session", "r") as f:
        session_id = int(next(f)) + 1
    with open("last_session", "w") as f:
        f.write(session_id)
    return "session" + str(session_id)


def get_experiment_id():
    with open("experiment_id", "r") as f:
        experiment_id = int(next(f))
    return "experiment" + str(experiment_id)


def update_experiment_id():
    with open("experiment_id", "r") as f:
        experiment_id = int(next(f)) + 1
    with open("experiment_id", "w") as f:
        f.write(experiment_id)
    return "experiment" + str(experiment_id)


def launch_gym(kwargs):
    # create log_folder with parameters

    # launch training

    # log results
    clean_repository()
    pass


def score_stats(kpis, weights):
    """

    :param kpis:
    :type kpis: dict
    :param weights:
    :type weights: dict
    :return:
    :rtype:
    """
    score = 0
    for kpi in kpis:
        score += kpis[kpi] * weights[kpi]

    score /= sum(weights.values())

    return score


def read_stats(replay_data):
    """

    :param replay_data:
    :type replay_data: dict
    :return:
    :rtype:
    """
    stats = replay_data['stats']
    turn_alive = stats[0]["last_frame_alive"]

    kpi = {
        "winner": stats[0]["rank"] == 1,
        "time_efficiency": (2000 - stats[0][
            "average_frame_response_time"]) / 2000,
        "damage_per_turn": stats[0]["damage_dealt"] / turn_alive,
        "ratio_prod": (stats[0]["total_ship_count"] /
                       stats[0]["total_ship_count"] +
                       stats[1]["total_ship_count"]),
        "ratio_damage": (stats[0]["damage_dealt"] /
                         stats[0]["damage_dealt"] +
                         stats[1]["damage_dealt"])}

    return kpi


def decode_replay(replay_file, rewrite=False):
    """

    :param replay_file:
    :type replay_file: str
    :param rewrite:
    :type rewrite: bool
    :return:
    :rtype:
    """
    with open(replay_file, "rb") as rpfile:
        decoded_data = zstd.decompress(rpfile.read()).decode('utf-8')
        replay_data = json.loads(decoded_data.strip())
    if rewrite:
        with open(replay_file, "w") as rpfile:
            json.dump(replay_data, rpfile)

    return replay_data


def handle_replays():
    onlyfiles = [f for f in os.listdir(".") if f.endswith(".hlt")]
    session_store_path = path.join(STORE_PATH,
                                   get_experiment_id(),
                                   update_session_id())
    os.mkdir(session_store_path)
    all_stats = []
    score = 0
    i = 1
    for file in onlyfiles:
        replay_data = decode_replay(file)
        stats = read_stats(replay_data)
        all_stats.append(stats)
        score += score_stats(stats, SCORE_WEIGHTS)
        # Moving replays out of the way
        os.rename(file,
                  os.path.join(session_store_path, "replay{}.hlt".format(i)))
        i += 1

    with open(path.join(session_store_path, "stats"), "w") as f:
        json.dump(all_stats, f)

    with open(path.join(session_store_path, "params"), "w") as f:
        json.dump(all_stats, f)

    score /= len(onlyfiles)

    return score


def map_parameters(args):
    """

    :param args:
    :type args:
    :return:
    :rtype: dict
    """
    params = {}
    if args:
        iter_args = iter(args)
    with open("parameter_mapping.json") as f:
        params = json.load(f)
        if args:
            params = {k: next(iter_args) for k in params}
    return params



if __name__ == "__main__":
    os.mkdir(STORE_PATH)
    experiment_store_path = path.join(STORE_PATH, update_experiment_id())
    os.mkdir(experiment_store_path)

    a = opt.fmin_l_bfgs_b(func, np.array([15, 1]),
                          bounds=[(None, None), (None, None)],
                          epsilon=0.001,
                          approx_grad=True)
