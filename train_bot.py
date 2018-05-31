import json
import os
import random
import subprocess
from multiprocessing import cpu_count
from time import time

import numpy as np
from hlt_client.compare_bots import _play_game
from scipy import optimize as opt

from utils.logging_id import (update_id, log_performances,
                              update_parameter_mapping)
from utils.param_handling import map_parameters
from utils.scoring import get_score
from multiprocessing import Pool

OPPONENT_COMMAND = "self"
STORE_PATH = ".data"
MAP_HEIGHTS = [160, 180, 200, 256]
CORES = int(cpu_count() / 2)


# CORES = 3

def clean_repository():
    """
    Remove log files
    """
    onlyfiles = [f for f in os.listdir(".") if f.endswith(".hlt")]
    # onlyfiles = [f for f in os.listdir(".") if
    #              f.endswith(".log") or f.endswith(".hlt")]
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
    if max(args) > 1 or min(args) < 0:
        print("error!!!")
        return 0
    print('no error')
    start = time()
    session_id = update_id("session")

    command = get_play_command(args)
    print("Session {}, playing {} games".format(session_id, num_games))
    pool = Pool()
    pool.map(play_game, [command] * num_games)
    pool.close()
    pool.join()

    print("All games of the session have been played")

    score, log_perfs = get_score()
    print("Score: {:.4}".format(score))
    print("Session run in {:.4}s".format(time() - start))

    log_performances(score, log_perfs, map_parameters(list(args)))

    # clean_repository()

    # function often try to minimize score ^^
    return score


def get_play_command(args):
    kwargs = map_parameters(list(args))
    map_height = random.sample(MAP_HEIGHTS, 1)[0]
    map_width = int(3 * map_height / 2)

    bot_command = "python3 MyBot.py "
    bot_command += " ".join(["--{} {}".format(k, v) for k, v in kwargs.items()])

    if OPPONENT_COMMAND == "self":
        opp_kwargs = map_parameters(map_parameters(()).values())
        opp_command = "python3 MyBot.py "
        opp_command += " ".join(["--{} {}".format(k, v)
                                 for k, v in opp_kwargs.items()])
    else:
        opp_command = OPPONENT_COMMAND

    binary = "./halite"
    game_run_command = '\"{}\" -d "{} {}" -t'.format(binary, map_width,
                                                     map_height)
    game_run_command += " \"{}\"".format(bot_command)
    game_run_command += " \"{}\"".format(opp_command)

    return game_run_command


def play_game(command):
    subprocess.run(command, shell=True)


def launch_game(args):
    kwargs = map_parameters(list(args))
    map_height = random.sample(MAP_HEIGHTS, 1)[0]
    map_width = int(3 * map_height / 2)

    bot_command = "python3 MyBot.py "
    bot_command += " ".join(
            ["--{} {}".format(k, v) for k, v in kwargs.items()])

    if OPPONENT_COMMAND == "self":
        opp_kwargs = map_parameters(map_parameters(()).values())
        opp_command = "python3 MyBot.py "
        opp_command += " ".join(["--{} {}".format(k, v)
                                 for k, v in opp_kwargs.items()])
    else:
        opp_command = OPPONENT_COMMAND

    _play_game("./halite", map_width, map_height,
               [bot_command, opp_command])

    print(".")


if __name__ == "__main__":
    while True:
        update_id("experiment")

        args = np.array([v for v in map_parameters([]).values()])
        bounds = [(0, 1) for _ in args]
        try:
            a = opt.minimize(async_score, args,
                             method='L-BFGS-B',
                             jac='2-point',
                             bounds=bounds,
                             options={"eps": 0.05,
                                      "maxls": 5,
                                      "maxfun": 50})
            update_parameter_mapping()
        except InterruptedError:
            update_parameter_mapping()
        finally:
            clean_repository()
            # pass
