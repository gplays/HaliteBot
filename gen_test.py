import argparse
import heapq
import json
from subprocess import PIPE, run
from itertools import combinations
import random
from multiprocessing.pool import Pool
from os import path
import os

from deap import creator, base, tools, algorithms

from train_bot import async_score, play_game, MAP_HEIGHTS
from utils.logging_id import update_id, data_path, write_tournament_perfs
from utils.param_handling import map_parameters
from utils.scoring import get_score_tournament

NGEN = 40
MU = 6
CXPB = 0.5
MUTPB = 0.5
N_PARAMS = 16
T1_POOL = 3
T1_WIN = 2
T2_POOL = 6
T2_WIN = 3
BOT = ["MyBot.py"]
BOSSES = ["holypegasus", "DaanPosthuma"]


def gen_opt():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("population_guess", initPopulation, list,
                     creator.Individual)

    # toolbox.register("evaluate", async_score)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
    toolbox.decorate("mate", checkBounds(0, 1))
    toolbox.decorate("mutate", checkBounds(0, 1))
    # toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population_guess()

    update_id("experiment")
    for gen in range(NGEN):
        update_id("session")
        print("New Generation: {}/{}".format(gen, NGEN))
        offspring1 = algorithms.varAnd(random.sample(population, k=MU), toolbox,
                                       cxpb=CXPB, mutpb=MUTPB)
        offspring2 = algorithms.varAnd(population, toolbox,
                                       cxpb=CXPB, mutpb=MUTPB)
        population += offspring1 + offspring2
        # 1st Turn
        best = tournament(population, T1_POOL, T1_WIN)
        # 2nd Turn
        population = tournament(best, T2_POOL, T2_WIN,boss=True)


def initPopulation(pcls, ind_init):
    my_path = "./.data/params"
    param_files = [f for f in os.listdir(my_path) if f.startswith("params")]
    param_files = sorted(param_files, reverse=True)[:MU]
    population = []
    for file in param_files:
        with open(path.join(my_path, file)) as f:
            population.append(list(map_parameters(json.load(f)).values()))
    return pcls(ind_init(c) for c in population)


def tournament(pop, size, n_best, boss=False):
    # Init path info
    my_path = path.join(data_path("session"), "tournament_{}".format(size))
    os.mkdir(my_path)
    pool_path = path.join(my_path, "pool_")

    # Init useful data
    l_pop = len(pop)
    idx = random.sample(list(range(l_pop)), l_pop)
    n_pool = int(l_pop / size)

    # Init Variables
    commands = []
    pool_list = [-1] * l_pop
    if boss:
        pool_list = [-1] * (l_pop + n_pool)

    print("New Tournament: {} pools of {}".format(n_pool, size))

    for k in range(0, l_pop, size):
        pool_id = int(k / size)

        for i in range(size):
            pool_list[idx[k + i]] = pool_id
        os.mkdir(pool_path + str(pool_id))
        for i, j in combinations(list(range(size)), r=2):
            x, y = k + i, k + j
            commands.append(play_command(command_player(pop[idx[x]], idx[x]),
                                         command_player(pop[idx[y]], idx[y])))
        if boss:
            id_boss = l_pop + pool_id
            pool_list[id_boss] = pool_id
            for i in range(size):
                x = k + i
                boss = BOSSES[pool_id % len(BOSSES)]
                commands.append(play_command(command_boss(boss, id_boss),
                                             command_player(pop[idx[x]],
                                                            idx[x])))
    # Run all games
    pool = Pool()
    pool.map(play_game2, commands)
    pool.close()
    pool.join()

    # Extrat information from games played

    scores, log_perf = get_score_tournament(pool_path, n_pool, pool_list,
                                            size+int(boss))
    write_tournament_perfs(pool_path, scores, log_perf, pop)

    winners = []
    # Extract winners for each pool
    for scores_pool in scores:
        sorted_score = sorted(scores_pool.items(),
                              key=lambda z: z[1], reverse=True)
        # Bosses names are longer than 3 characters, all others are just numbers
        # We assume we will have less than 1k individuals in the tournament
        winners.extend([pop[id] for id, score in sorted_score
                        if id < l_pop][:n_best])

    return winners


def play_game2(command):
    run(command, bufsize=0, stdin=PIPE, stdout=PIPE, shell=True)


def command_player(args, idx_player):
    kwargs = map_parameters(args)
    bot_command = "python3 " + BOT[0] + " "
    bot_command += " ".join(["--{} {}".format(k, v)
                             for k, v in kwargs.items()])
    bot_command += " --name {}".format(idx_player)

    return bot_command


def command_boss(boss, id_boss):
    boss_path = path.join("bosses", boss, "MyBot.py")
    return " ".join(["python3", boss_path, "--name", str(id_boss)])


def play_command(bot_command, opp_command):
    map_height = random.sample(MAP_HEIGHTS, 1)[0]
    map_width = int(3 * map_height / 2)

    binary = "./halite"
    game_run_command = '\"{}\" -d "{} {}" -t'.format(binary, map_width,
                                                     map_height)
    game_run_command += " \"{}\"".format(bot_command)
    game_run_command += " \"{}\"".format(opp_command)

    return game_run_command


def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring

        return wrapper

    return decorator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if args.fast:
        BOT[0] = "fastBot.py"

    print("Starting Genetic Optimization")
    gen_opt()
