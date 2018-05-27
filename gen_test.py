import heapq
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
N_PARAMS = 13
T1_POOL = 3
T1_WIN = 2
T2_POOL = 6
T2_WIN = 3


def gen_opt():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=N_PARAMS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", async_score)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.decorate("mate", checkBounds(0, 1))
    toolbox.decorate("mutate", checkBounds(0, 1))
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=MU)
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
        population = tournament(best, T2_POOL, T2_WIN)


def tournament(pop, size, n_best):
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
    pool_list = []

    print("New Tournament: {} pools of {}".format(n_pool, size))

    for k in range(0, l_pop, size):
        pool_id = int(k / size)

        pool_list.append([idx[k + i] for i in range(size)])
        os.mkdir(pool_path + str(pool_id))
        for i, j in combinations(list(range(size)), r=2):
            x, y = k + i, k + j
            commands.append(get_play_command_2_player(pop[idx[x]], pop[idx[y]],
                                                      idx[x], idx[y]))
    # Run all games
    pool = Pool()
    pool.map(play_game, commands)
    pool.close()
    pool.join()

    # Extrat information from games played
    scores, log_perf = get_score_tournament(pool_path, n_pool, pool_list, size)
    write_tournament_perfs(pool_path, scores, log_perf, pop)

    # Extract winners
    # List comprehension flatten list of dict and for each dict sort keys
    # according to value
    winners = [pop[tuple[0]] for scores_pool in scores
               for tuple in sorted(scores_pool.items(),
                                   key=lambda z: z[1],
                                   reverse=True)[:n_best]]

    return winners


def get_play_command_2_player(args1, args2, idx_player1, idx_player2):
    kwargs1 = map_parameters(args1)
    kwargs2 = map_parameters(args2)
    map_height = random.sample(MAP_HEIGHTS, 1)[0]
    map_width = int(3 * map_height / 2)

    bot_command = "python3 MyBot.py "
    bot_command += " ".join(["--{} {}".format(k, v)
                             for k, v in kwargs1.items()])
    bot_command += " --name {}".format(idx_player1)

    opp_command = "python3 MyBot.py "
    opp_command += " ".join(["--{} {}".format(k, v)
                             for k, v in kwargs2.items()])
    bot_command += " --name {}".format(idx_player2)

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
    print("Starting Genetic Optimization")
    gen_opt()
