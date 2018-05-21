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
from hlt.constants import MAX_SPEED
from hlt.navigation_optim import navigate_optimize
from utils.param_handling import map_parameters

THREAT_THRESHOLD = 1


def run_game(game, kwargs):
    while True:

        game_map = game.update_map()

        command_queue = []
        immobile_ships = []
        mobile_ships = []
        game_map.augment_entities(kwargs)
        for ship in game_map.get_me().all_ships():

            # Skip docked ships
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                immobile_ships.append(ship)
                continue

            # Check possibility of docking
            nearest_planet = game_map.get_nearest_planet(ship)
            if ship.can_dock(nearest_planet):
                logging.info("DOCKING")
                # Threat_level set at 0
                if nearest_planet.threat_level < THREAT_THRESHOLD:
                    command_queue.append(ship.dock(nearest_planet))
                    immobile_ships.append(ship)
                    continue

            mobile_ships.append(ship)

        immobile_entities = immobile_ships + game_map.all_planets()
        foes = game_map.all_ennemy_ships()

        if mobile_ships:
            borders = (game_map.width, game_map.height)
            args = [mobile_ships, foes, immobile_entities,
                    kwargs["thrust_ratio"] * MAX_SPEED, borders]
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
