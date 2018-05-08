"""
Welcome to your first Halite-II bot!

This bot's name is Settler. It's purpose is simple (don't expect it to win
complex games :) ):
1. Initialize game
2. If a ship is not docked and there are unowned planets
2.a. Try to Dock in the planet if close enough
2.b If not, go towards the planet

Note: Please do not place print statements here as they are used to
communicate with the Halite engine. If you need
to log anything use the logging module.
"""
# Then let's import the logging module so we can print out information
import logging

# Let's start by importing the Halite Starter Kit so we can interface with
# the Halite engine
import hlt
from RVO import resolve_conflicts
import learnable_constants as const

CONFLICT_CHECK = False

# GAME START
# Here we define the bot's name as Settler and initialize the game, including
#  communication with the Halite engine.
game = hlt.Game("Settler")
# Then we print our start message to the logs
logging.info("Starting my Settler bot!")

while True:
    # TURN START
    # Update the map for the new turn and get the latest version
    game_map = game.update_map()

    # Here we define the set of commands to be sent to the Halite engine at
    # the end of the turn
    command_queue = []

    if CONFLICT_CHECK:
        list_moves = []

    for ship in game_map.get_me().all_ships():

        # Skip docked ships
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue

        # Check possibility of docking
        nearest_planet = game_map.get_nearest_planet(ship)
        if ship.can_dock(nearest_planet):
            threat = game_map.compute_threat(nearest_planet)
            if nearest_planet.threat_level < const.THREAT_THRESHOLD:
                command_queue.append(ship.dock(nearest_planet))
                continue

        # Compute gradient
        grad_x, grad_y = (0, 0)
        for e_type, entity in game_map.all_entities_by_type:
            if entity != ship:
                simple_grad = ship.compute_grad(e_type, entity)
                grad_x += simple_grad[0]
                grad_y += simple_grad[1]

        if CONFLICT_CHECK:
            list_moves.append((ship, (grad_x, grad_y)))
        else:
            navigate_command = ship.thrust_x_y(grad_x, grad_y)
            command_queue.append(navigate_command)

    if CONFLICT_CHECK:
        valid_commands = resolve_conflicts(list_moves,
                                           game_map.all_planets,
                                           avoid_planets=True,
                                           avoid_ships=True)

    # Send our set of commands to the Halite engine for this turn
    game.send_command_queue(command_queue)
    # TURN END


# GAME END


