import json
import math
import os
import random
from multiprocessing import Pool, cpu_count
from os import path

import numpy as np
from hlt_client.compare_bots import _play_game
from scipy import optimize as opt

from utils.logging_id import get_id, update_id, reset_id
from utils.scoring import get_score

OPPONENT_COMMAND = "python3 simpleBot.py"
STORE_PATH = ".data"
MAP_HEIGHTS = [160, 180, 200, 256]
CORES = cpu_count()


def func(x):
    print(x[0])
    return 2 * math.log(x[0] ** 2) ** 2 + 3 * x[0] + 1


def clean_repository():
    """
    Remove log files
    """
    onlyfiles = [f for f in os.listdir(".") if f.endswith(".log")]
    for file in onlyfiles:
        os.remove(file)


def async_score(args, num_games=CORES):
    """
    Evaluating bot
    :param args: the values of the learnable parameters passed to the bot
    :type args: ndarray
    :return:
    :rtype:
    """
    iterables = [args] * num_games
    print("New session, playing {} games".format(num_games))
    pool = Pool()
    pool.map(launch_game, iterables)
    pool.close()
    pool.join()
    print("All games of the session have been played")
    session_store_path = path.join(STORE_PATH,
                                   get_id("experiment"),
                                   update_id("session"))
    os.mkdir(session_store_path)

    score, log_perfs = get_score(session_store_path)
    print("Score: {:.4}".format(score))

    with open(path.join(session_store_path,
                        "stats-{:.4}.json".format(score)), "w") as f:
        json.dump(log_perfs, f)

    with open(path.join(session_store_path, "params.json"), "w") as f:
        json.dump(map_parameters(list(args)), f)

    clean_repository()

    # function often try to minimize score ^^
    return -score


def launch_game(args):
    kwargs = map_parameters(list(args))
    map_height = random.sample(MAP_HEIGHTS, 1)[0]
    map_width = int(3 * map_height / 2)

    bot_command = "python MyBot.py "
    bot_command += " ".join(["--{} {}".format(k, v) for k, v in kwargs.items()])

    _play_game("./halite", map_width, map_height,
               [bot_command, OPPONENT_COMMAND])
    print(".")


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
    experiment_store_path = path.join(STORE_PATH, update_id("experiment"))
    os.mkdir(experiment_store_path)

    reset_id("session")

    args = np.array([v[0] for v in map_parameters([]).values()])
    bounds = [(v[1], v[2]) for v in map_parameters([]).values()]
    async_score(args)
    a = opt.fmin_l_bfgs_b(async_score, args,
                          bounds=bounds,
                          epsilon=0.001,
                          approx_grad=True)
