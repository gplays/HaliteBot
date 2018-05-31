import json
import math
import os
import zstd

from utils.logging_id import data_path

SCORE_WEIGHTS = {"winner": 0.5,
                 "time_efficiency": 0.1,
                 "ratio_prod": 0.2,
                 "ratio_damage": 0.2, }


def extract_perf_from_stats(replay_data):
    """

    :param replay_data:
    :type replay_data: dict
    :return:
    :rtype:
    """
    stats = replay_data['stats']
    turn_alive = stats["0"]["last_frame_alive"]
    response_time = stats["0"]["average_frame_response_time"]
    kpi = {
        "winner": stats["0"]["rank"] == 1,
        "time": response_time,
        "time_efficiency": ((0.05 + response_time) /
                            (0.1 + response_time +
                             stats["1"]["average_frame_response_time"])),
        "ratio_prod": ((0.05 + stats["0"]["total_ship_count"]) /
                       (.1 + stats["0"]["total_ship_count"] +
                        stats["1"]["total_ship_count"])),
        "ratio_damage": ((0.05 + stats["0"]["damage_dealt"]) /
                         (.1 + stats["0"]["damage_dealt"] +
                          stats["1"]["damage_dealt"]))}
    stats.update(kpi)
    return stats


def decode_replay(replay_file):
    """
    Decode replay file to python dict
    :param replay_file:
    :type replay_file: str
    :return:
    :rtype:
    """

    with open(replay_file, "rb") as rpfile:
        decoded_data = zstd.decompress(rpfile.read()).decode('utf-8')
        replay_data = json.loads(decoded_data.strip())
    return replay_data


def get_score():
    onlyfiles = [f for f in os.listdir(".") if f.endswith(".hlt")]

    log_perfs = []
    score = 0
    i = 1
    wins = 0
    for file in onlyfiles:
        try:
            replay_data = decode_replay(file)
            perfs = extract_perf_from_stats(replay_data)
            log_perfs.append(perfs)
            wins += perfs["winner"]
            for kpi in SCORE_WEIGHTS:
                score += perfs[kpi] * SCORE_WEIGHTS[kpi]

        except zstd.Error:
            pass
        os.rename(file,
                  os.path.join(data_path("session"), "replay{}.hlt".format(i)))
        i += 1

    log_perfs.append({"win": wins, "loss": i - 1 - wins})

    score /= len(onlyfiles)

    return score, log_perfs


def get_score_tournament(pool_path, n_pool, pool_list, pool_size):
    onlyfiles = [f for f in os.listdir(".") if f.endswith(".hlt")]

    log_perfs = [{} for _ in range(n_pool)]
    scores = [{} for _ in range(n_pool)]
    for player, pool in enumerate(pool_list):
        scores[pool][player] = 0

    my_pool_path = pool_path + "0"
    for file in onlyfiles:
        match_name = "ERR"
        try:
            replay_data = decode_replay(file)
            players = replay_data["player_names"]
            pool = pool_list[int(players[0])]
            match_name = "vs".join(players)
            perfs = extract_perf_from_stats(replay_data)
            log_perfs[pool][match_name] = perfs
            score = sum([perfs[kpi] * SCORE_WEIGHTS[kpi]
                         for kpi in SCORE_WEIGHTS])
            scores[pool][int(players[0])] += score / (pool_size - 1)
            scores[pool][int(players[1])] += (1 - score) / (pool_size - 1)
            my_pool_path = pool_path + str(pool)
        except zstd.Error:
            pass
        os.rename(file,
                  os.path.join(my_pool_path, "replay{}.hlt".format(match_name)))

    return scores, log_perfs
