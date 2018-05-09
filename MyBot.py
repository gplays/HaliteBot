"""
Note: Please do not place print statements here as they are used to
communicate with the Halite engine. If you need
to log anything use the logging module.
"""
import argparse
import logging
import math

import hlt
import hlt.learnable_constants as const
from RVO import resolve_conflicts
from hlt.entity import Position


def run_game(game, **kwargs):
    conflict_check = kwargs.get("conflict_check", False)

    while True:

        game_map = game.update_map()

        command_queue = []
        immobile_ships = []
        if conflict_check:
            list_moves = []

        for ship in game_map.get_me().all_ships():

            # Skip docked ships
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                immobile_ships.append(ship)
                continue

            # Check possibility of docking
            nearest_planet = game_map.get_nearest_planet(ship)
            if ship.can_dock(nearest_planet):
                logging.info("DOCKING")
                threat = game_map.compute_threat_docking(nearest_planet)
                # Overriding threat assesment for the time being
                if True or nearest_planet.threat_level < const.THREAT_THRESHOLD:
                    command_queue.append(ship.dock(nearest_planet))
                    continue

            # Compute gradient
            grad_x, grad_y = (0, 0)
            for e_type, list_entity in game_map.all_entities_by_type.items():
                for entity in list_entity:
                    if entity == ship:
                        continue
                    simple_grad = ship.compute_grad(e_type, entity, **kwargs)
                    logging.info("grad: {}".format(simple_grad))
                    logging.info("dist: {}".format(
                        ship.calculate_distance_between(entity)))
                    grad_x += simple_grad[0]
                    grad_y += simple_grad[1]

            if conflict_check:
                list_moves.append((ship, (grad_x, grad_y)))
            else:
                speed = hlt.constants.MAX_SPEED * const.THRUST_RATIO
                norm_grad = math.sqrt(grad_x ** 2 + grad_y ** 2)
                grad_x *= speed / norm_grad
                grad_y *= speed / norm_grad
                navigate_command = ship.navigate(Position(ship.x + grad_x,
                                                          ship.y + grad_y),
                                                 game_map,
                                                 speed)
                if navigate_command is not None:
                    command_queue.append(navigate_command)

        if conflict_check:
            immobile_obstacles = immobile_ships + game_map.all_planets
            valid_commands = resolve_conflicts(list_moves,
                                               immobile_obstacles)
            command_queue.append(valid_commands)

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
    parser.add_argument("--w_collision", type=float, default=-2)
    parser.add_argument("--w_swarm", type=float, default=1)
    parser.add_argument("--w_foe", type=float, default=1)
    parser.add_argument("--w_planet", type=float, default=1)
    parser.add_argument("--thrust_ratio", type=float, default=0.5)

    run_game(hlt.Game("Settler"), **vars(parser.parse_args()))
