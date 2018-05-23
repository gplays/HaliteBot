"""
Note: Please do not place print statements here as they are used to
communicate with the Halite engine. If you need
to log anything use the logging module.
"""
import argparse
import logging
import pickle
import sys
import traceback

import hlt
from hlt.Grid import GridSet
from hlt.constants import MAX_SPEED, DOCK_THRESHOLD, UNDOCK_THRESHOLD
from hlt.navigation_optim import navigate_optimize
from utils.param_handling import map_parameters


def run_game(game, kwargs):
    while True:

        game_map = game.update_map()

        command_queue = []

        game_map.augment_entities(kwargs)
        all_entities = game_map.all_entities

        gridSet = GridSet((16, 32, 64), all_entities)
        game_map.set_gridSet(gridSet)
        game_map.compute_planet_threat_attractivity()

        for ship in game_map.my_undocked_ships:
            nearest_planet = game_map.get_nearest_planet(ship)
            if (nearest_planet.threat_level < DOCK_THRESHOLD and
                    ship.can_dock(nearest_planet)):
                command_queue.append(ship.dock(nearest_planet))

        for ship in game_map.my_docked_ships:
            if (ship.planet is not None and
                    ship.planet.threat_level > UNDOCK_THRESHOLD):
                command_queue.append(ship.undock())

        game_map.assign_targets()

        mobile_ships = game_map.my_undocked_ships
        if mobile_ships:
            # max_speed = kwargs["thrust_ratio"] * MAX_SPEED
            args = [game_map, mobile_ships, MAX_SPEED]
            # dump_pre_navigation(*args)
            commands = navigate_optimize(*args)
            command_queue.extend(commands)

        game.send_command_queue(command_queue)
        # TURN END


def dump_pre_navigation(*args):
    with open("pickled", "wb") as f:
        pickle.dump(args, f)


if __name__ == "__main__":

    default = map_parameters(map_parameters([]).values())

    parser = argparse.ArgumentParser()
    for key, value in default.items():
        parser.add_argument("--" + key, type=float, default=value)


    def exception_handler(exception_type, exception, tb):
        logging.error(traceback.format_exception(exception_type, exception, tb))


    sys.excepthook = exception_handler

    run_game(hlt.Game("Optim"), vars(parser.parse_args()))
