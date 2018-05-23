import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds
from scipy.sparse import csr_matrix

from hlt.constants import FUDGE, RAD_FUDGE
from hlt.constr_RVO import check_initial_violation


def get_velocity_const(n, m, all_entities, max_speed, borders, obj=True):
    """


    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param all_entities: list of entities
    :type all_entities: [Entity]
    :param max_speed: Maximum speed allowed
    :param borders: tuple of map_width,map_length
    :type borders: (int,int)
    :type max_speed: float
    :return:
    :rtype:
    """
    preprocess = constraints_pre(n, m, all_entities, max_speed)
    unit_vec, sum_rad, all_pos, dist, list_index, prev_vel = preprocess

    cons = const_RVO(n, m, unit_vec, dist, sum_rad, max_speed ** 2, list_index,
                     prev_vel)
    jac = jac_RVO(n, m, unit_vec, sum_rad, list_index, prev_vel)
    hess = hess_RVO(n, m, unit_vec, list_index)

    inf = -np.hstack((all_pos[:n, 0], all_pos[:n, 1]))
    sup = np.hstack((borders[0] - all_pos[:n, 0],
                     borders[1] - all_pos[:n, 1]))
    bounds = Bounds(inf, sup, keep_feasible=True)
    # logging.info("Indexes {}".format(list_index))
    # logging.info("Previous velocity: {}".format(prev_vel))
    # logging.info("Initial const: {}".format(cons(np.zeros(2 * n))))
    check_initial_violation(n, cons, list_index, all_entities)

    if obj:
        const = [NonlinearConstraint(cons, -np.inf, -FUDGE, jac=jac,
                                     hess=hess)]
    else:
        const = [{'type': 'ineq',
                  'fun': cons,
                  'jac': jac}]
    return const, bounds


def constraints_pre(n, m, all_entities, max_speed):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param all_entities: list of entities
    :type all_entities: [Entity]
    :return:
    :rtype:
    """
    all_pos = np.array([[e.x, e.y] for e in all_entities])

    diff_pos = (np.broadcast_to(all_pos, (n, m, 2)) -
                np.broadcast_to(all_pos[0:n].reshape(n, 1, 2), (n, m, 2)))
    dist = np.sqrt(np.sum(diff_pos * diff_pos, axis=2))

    unit_vec = diff_pos / (dist.reshape((n, m, 1)) + 0.0001)

    all_radius = np.array([e.radius for e in all_entities])
    sum_rad = (np.broadcast_to(all_radius[0:n].reshape(n, 1), (n, m)) +
               np.broadcast_to(all_radius, (n, m)))

    # TODO check usefulness
    sum_rad = sum_rad + RAD_FUDGE

    prev_vel = np.zeros((m, 2))
    if m > n:
        prev_vel[n:] = np.array([[e.vel.x, e.vel.y] for e in all_entities[n:]])

    counter = iter(range(n * m))
    speed = np.hstack((np.ones((n, n)) * 2, np.ones((n, m - n)))) * max_speed
    reachable = dist - all_radius - speed - prev_vel
    list_index = [(next(counter), i, j) for i in range(n) for j in range(m) if
                  i < j and reachable[i, j] < 0]

    return unit_vec, sum_rad, all_pos, dist, list_index, prev_vel


def const_RVO(n, m, unit_vec, dist, sum_rad, speed_sq, list_index, prev_vel):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m:
    :param unit_vec: Matrix containing the vector between the center of any
    two entities with norm 1. Shape (n,m,2)
    :type unit_vec: ndarray
    :param sum_rad: Matrix of sum of radius shape (n,m)
    :type sum_rad: ndarray
    :param speed_sq: Max speed allowed squared
    :type speed_sq: float
    :return: the constraint function
    :rtype: Callable
    """
    U = unit_vec
    V = np.zeros(unit_vec.shape)
    V[:, :, 0] = -U[:, :, 1]
    V[:, :, 1] = U[:, :, 0]

    def const(x):
        vel = np.zeros((m, 2))
        vel[:n, 0] = x[:n]
        vel[:n, 1] = x[n:]
        vel = vel + prev_vel
        diff_vel = (np.broadcast_to(vel[0:n].reshape(n, 1, 2), (n, m, 2)) -
                    np.broadcast_to(vel, (n, m, 2)))

        Y = np.sum(diff_vel * V, axis=2)
        X = np.sum(diff_vel * U, axis=2)
        RVO = 2 * sum_rad * (X - dist + sum_rad) - Y ** 2

        speed_limit = np.sum(vel[:n] ** 2, axis=1) - speed_sq

        mobile = [RVO[i, j] for c, i, j in list_index]

        cons = np.hstack((np.array(mobile), speed_limit))
        return cons

    return const


def jac_RVO(n, m, unit_vec, sum_rad, list_index, prev_vel):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param unit_vec: Matrix containing the vector between the center of any
    two entities with norm 1. Shape (n,m,2)
    :type unit_vec: ndarray
    :param sum_rad: Matrix of sum of radius shape (n,m)
    :type sum_rad: ndarray
    :return: the Jacobian of the constraint function
    :rtype: Callable
    """
    U = unit_vec
    V = np.zeros(unit_vec.shape)
    V[:, :, 0] = -U[:, :, 1]
    V[:, :, 1] = U[:, :, 0]
    iterator_speed = [(i + len(list_index), i) for i in range(n)]

    def jac(x):

        vel = np.zeros((m, 2))
        vel[:n, 0] = x[:n]
        vel[:n, 1] = x[n:]
        vel = vel + prev_vel
        diff_vel = (
                np.broadcast_to(vel[0:n].reshape(n, 1, 2), (n, m, 2)) -
                np.broadcast_to(vel, (n, m, 2)))

        Y = np.sum(diff_vel * V, axis=2)

        grad = (2 * sum_rad.reshape((n, m, 1)) * U -
                2 * V * Y.reshape((n, m, 1)))
        row_col_data = []

        for c_ind, i, j in list_index:

            # True for all entities collision
            row_col_data.append((c_ind, i, grad[i, j, 0]))
            row_col_data.append((c_ind, n + i, grad[i, j, 1]))

            # For mobile ship-mobile ship collisions
            if j < n:
                row_col_data.append((c_ind, j, -grad[i, j, 0]))
                row_col_data.append((c_ind, n + j, -grad[i, j, 1]))

        for c_ind, i in iterator_speed:
            row_col_data.append((c_ind, i, 2 * x[i]))
            row_col_data.append((c_ind, n + i, 2 * x[n + i]))

        row, col, data = zip(*row_col_data)
        jacob = csr_matrix((data, (row, col))).toarray()
        return jacob

    return jac


def hess_RVO(n, m, unit_vec, list_index):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param unit_vec: Matrix containing the vector between the center of any
    two entities with norm 1. Shape (n,m,2)
    :type unit_vec: ndarray
    :return: the Hessian of the constraint function (takes two argument x and v,
    see scipy documentation)
    :rtype: Callable
    """
    U = unit_vec
    V = np.zeros(unit_vec.shape)
    V[:, :, 0] = -U[:, :, 1]
    V[:, :, 1] = U[:, :, 0]
    V_x2 = V[:, :, 0] ** 2
    V_y2 = V[:, :, 1] ** 2
    V_x_y = V[:, :, 0] * V[:, :, 1]

    iterator_speed = [(i + len(list_index), i) for i in range(n)]

    def hess(x, v):

        my_hess = np.zeros((2 * n, 2 * n))

        for c_ind, i, j in list_index:

            my_hess[i, i] += v[c_ind] * 2 * V_x2[i, j]
            my_hess[n + i, n + i] += v[c_ind] * 2 * V_y2[i, j]
            my_hess[i, n + i] += v[c_ind] * 2 * V_x_y[i, j]
            my_hess[n + i, i] += v[c_ind] * 2 * V_x_y[i, j]
            if j < n:
                my_hess[j, i] += -v[c_ind] * 2 * V_x2[i, j]
                my_hess[i, j] += -v[c_ind] * 2 * V_x2[i, j]
                my_hess[i, n + j] += -v[c_ind] * 2 * V_x_y[i, j]
                my_hess[j, n + i] += -v[c_ind] * 2 * V_x_y[i, j]
                my_hess[n + j, i] += -v[c_ind] * 2 * V_x_y[i, j]
                my_hess[n + i, j] += -v[c_ind] * 2 * V_x_y[i, j]

        for c_ind, i in iterator_speed:
            my_hess[i, i] += v[c_ind] * 2
            my_hess[n + i, n + i] += v[c_ind] * 2

        return my_hess

    return hess
