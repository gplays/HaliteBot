import logging
from time import time

import numpy as np
from scipy.optimize import minimize
from hlt.constr_RVO import get_cone_vel_const
from hlt.objective_function import objective_function


def navigate_optimize(game_map, mobile_ships, max_speed, target_weight):
    """

    :param game_map:
    :type game_map: hlt.Map
    :param mobile_ships:
    :type mobile_ships: list
    :param max_speed:
    :type max_speed:
    :return:
    :rtype:
    """
    start = time()
    gridSet = game_map.gridSet
    borders = (game_map.width, game_map.height)
    foes = game_map.all_ennemy_ships()
    commands = []
    while mobile_ships:
        ref_ship = mobile_ships.pop()
        optim_set = gridSet.get_neighbours(ref_ship, dist=0)
        free_optim_set = [ref_ship] + [ship for ship in optim_set
                                       if ship in mobile_ships]
        neighbourhood = gridSet.get_neighbours(ref_ship, dist=1)
        neighbourhood = [entity for entity in neighbourhood
                         if entity not in free_optim_set]

        collision_set = [entity for entity in neighbourhood
                         if entity not in foes]
        nearest_planet = game_map.get_nearest_planet(ref_ship)
        if nearest_planet not in collision_set:
            collision_set.append(nearest_planet)
        centroids = gridSet.get_centroids(ref_ship)

        all_entities_collision = free_optim_set + collision_set
        all_entities = free_optim_set + neighbourhood + centroids
        n = len(free_optim_set)
        m1 = len(all_entities_collision)
        m2 = len(all_entities)

        const, bounds = get_cone_vel_const(n, m1, all_entities_collision,
                                           max_speed, borders, obj=True)
        fun_grad, hess = objective_function(n, m2, all_entities, target_weight)
        x0 = np.zeros(2 * n)

        res = minimize(fun_grad, x0, method='trust-constr',
                       jac=True, hess=hess,
                       constraints=const,
                       options={'xtol': 0.05,
                                "maxiter": 10},
                       bounds=bounds)

        logging.info("#Var:{:2} #obst:{:2} #E:{:2} constDev:{:.2f} "
                     "time1:{:.2f} time2:{:.2f}"
                     "".format(n, m1, m2, res.constr_violation,
                               res.execution_time, time() - start))
        commands.extend([e.thrust_x_y(x, y)
                         for e, x, y in zip(free_optim_set,
                                            *res.x.reshape((2, n)))])

        for ship in free_optim_set:
            if ship != ref_ship:
                mobile_ships.remove(ship)

    return commands
