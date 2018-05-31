import json
import os
import os.path as path
import shutil

from utils.param_handling import map_parameters

STORE_PATH = ".data"
PREV_LEVEL = {"gen": "session",
              "session": "experiment",
              "experiment": STORE_PATH,
              STORE_PATH: None}


def log_performances(score, log_perfs, params):
    session_path = data_path("session")
    best_path = path.join(data_path("experiment"), "best")

    if is_best_score_yet(score, best_path):
        print("Best Score yet!")
        write_perfs(best_path, score, log_perfs, params)
    write_perfs(session_path, score, log_perfs, params)

    update_win_rate(log_perfs[-1])


def update_win_rate(win_loss):
    experiment_path = path.join(data_path("experiment"), "win_loss")

    if path.exists(experiment_path):
        with open(experiment_path) as f:
            win_dict = json.load(f)
    else:
        win_dict = {"win": 0, "loss": 0, "ratio": 0}
    win_dict["win"] += win_loss["win"]
    win_dict["loss"] += win_loss["loss"]
    n_games = win_dict["win"] + win_dict["loss"]
    win_dict["ratio"] = win_dict["win"] / n_games

    with open(experiment_path, "w") as f:
        json.dump(win_dict, f)

    if n_games > 12 and win_dict["ratio"] > 0.65:
        pass
        # raise InterruptedError


def write_perfs(my_path, score, log_perfs, params):
    with open(path.join(my_path, "score"), "w") as f:
        f.write(str(score))
    with open(path.join(my_path, "stats-{:.4}.json".format(score)), "w") as f:
        json.dump(log_perfs, f)

    with open(path.join(my_path, "params-{:.4}.json".format(score)), "w") as f:
        json.dump(params, f)


def write_tournament_perfs(pool_path_prefix, scores, log_perfs, params):
    for pool, pool_scores in enumerate(scores):
        pool_path = pool_path_prefix + str(pool)

        for p, s in pool_scores.items():
            file_name = "params-{:.4}-{}.json".format(float(s), p)
            log_path = path.join(pool_path, file_name)
            with open(log_path, "w") as f:
                if int(p)<len(params):
                    json.dump(map_parameters(params[int(p)]), f)

        log_path = path.join(pool_path, "stats.json")
        with open(log_path, "w") as f:
            json.dump(log_perfs[pool], f)


def is_best_score_yet(score, best_path):
    if not path.exists(best_path):
        os.mkdir(best_path)
        best_score = 0
    else:
        with open(path.join(best_path, "score")) as f:
            best_score = float(f.read())

    return best_score < score


def data_path(level):
    if PREV_LEVEL[level] is None:
        return STORE_PATH
    else:
        root = data_path(PREV_LEVEL[level])
        return path.join(root, level + str(get_id("", root)))


def get_id(level, root=None):
    if root is None:
        root = data_path(PREV_LEVEL[level])
    id_path = path.join(root, "last_id")
    if not path.exists(id_path):
        with open(id_path, "w") as f:
            f.write("1")

    with open(id_path, "r") as f:
        my_id = int(next(f))

    return my_id


def update_id(level):
    root = data_path(PREV_LEVEL[level])
    id_path = path.join(root, "last_id")

    if not path.exists(id_path):
        with open(id_path, "w") as f:
            f.write("0")

    with open(id_path, "r") as f:
        my_id = str(int(next(f)) + 1)
    with open(id_path, "w") as f:
        f.write(my_id)

    new_dir = path.join(root, level + my_id)
    os.mkdir(new_dir)

    if level == "experiment":
        copy_game_files(new_dir)

    return my_id


def copy_game_files(out_path):
    hlt_files = [f for f in os.listdir("./hlt") if f.endswith(".py")]
    os.mkdir(path.join(out_path, "Bot"))
    hlt_path = path.join(out_path, "Bot", "hlt")
    os.mkdir(hlt_path)
    for f in hlt_files:
        shutil.copy("./hlt/" + f, hlt_path)
    shutil.copy("./MyBot.py", path.join(out_path, "Bot", "MyBot.py"))


def update_parameter_mapping():
    best_path = path.join(data_path("experiment"), "best")
    onlyfiles = [f for f in os.listdir(best_path) if f.startswith("params")]
    best_params_file = max(onlyfiles)
    with open(path.join(best_path, best_params_file)) as f:
        best_params = json.load(f)
    with open("parameter_mapping.json") as f:
        params = json.load(f)

    params = {k: [best_params[k], v[1], v[2]] for k, v in params.items()}

    with open("parameter_mapping.json", "w") as f:
        json.dump(params, f)


if not os.path.exists(STORE_PATH):
    os.mkdir(STORE_PATH)
