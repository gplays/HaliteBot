"""
Note: Please do not place print statements here as they are used to
communicate with the Halite engine. If you need
to log anything use the logging module.
"""
import argparse
import logging
import sys
import traceback

import hlt
from hlt.constants import MAX_SPEED
from hlt.navigation_optim import navigate_optimize


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
                if nearest_planet.threat_level < kwargs["threat_threshold"]:
                    command_queue.append(ship.dock(nearest_planet))
                    immobile_ships.append(ship)
                    continue

            mobile_ships.append(ship)

        immobile_entities = immobile_ships + game_map.all_planets()
        foes = game_map.all_ennemy_ships()

        if mobile_ships:
            commands = navigate_optimize(mobile_ships, foes, immobile_entities,
                                         kwargs["thrust_ratio"] * MAX_SPEED)
            command_queue.extend(commands)

        # Send our set of commands to the Halite engine for this turn
        game.send_command_queue(command_queue)
        # TURN END


        # GAME END


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--collision_offset", type=float, default=1)
    parser.add_argument("--k_collision", type=float, default=1.5)
    parser.add_argument("--k_foe", type=float, default=60)
    parser.add_argument("--k_planet", type=float, default=60)
    parser.add_argument("--k_swarm", type=float, default=30)
    parser.add_argument("--threat_threshold", type=float, default=1)
    parser.add_argument("--w_collision", type=float, default=2)
    parser.add_argument("--w_swarm", type=float, default=-1000)
    parser.add_argument("--w_foe", type=float, default=-1000)
    parser.add_argument("--w_planet", type=float, default=-1000)
    parser.add_argument("--thrust_ratio", type=float, default=0.5)


    def exception_handler(exception_type, exception, tb):
        logging.error(traceback.format_exception(exception_type, exception, tb))


    sys.excepthook = exception_handler

    run_game(hlt.Game("Optim"), vars(parser.parse_args()))
