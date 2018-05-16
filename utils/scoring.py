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
        "time_efficiency": math.exp(- response_time / 100),
        "ratio_prod": (stats["0"]["total_ship_count"] /
                       (1 + stats["0"]["total_ship_count"] +
                        stats["1"]["total_ship_count"])),
        "ratio_damage": (stats["0"]["damage_dealt"] /
                         (1 + stats["0"]["damage_dealt"] +
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
                # Moving replays out of the way
        except zstd.Error:
            pass
        os.rename(file,
                  os.path.join(data_path("session"), "replay{}.hlt".format(i)))
        i += 1

    log_perfs.append({"win": wins, "loss": i - 1 - wins})

    score /= len(onlyfiles) * sum(SCORE_WEIGHTS.values())

    return score, log_perfs
