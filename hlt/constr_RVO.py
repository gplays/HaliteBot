import logging

import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds

from .constants import FUDGE, RAD_FUDGE



def get_cone_vel_const(n, m, all_entities, max_speed, borders, **kwargs):
    preprocess = constraints_pre_cone(n, m, all_entities, max_speed)
    all_pos, diff_pos, sum_rad, u1, u2, u3, d3, speed_sq, collision_idx, \
    prev_vel = preprocess

    cons = const_RVO_cone(diff_pos, sum_rad, u1, u2, u3, d3, speed_sq,
                          collision_idx, prev_vel)
    jac = jac_RVO_cone(diff_pos, sum_rad, u1, u2, u3, d3, collision_idx,
                       prev_vel)
    hess = hess_RVO_cone(n)

    const = [NonlinearConstraint(cons, -np.inf, -FUDGE, jac=jac,
                                 hess=hess)]
    inf = -np.hstack((all_pos[:n, 0], all_pos[:n, 1]))
    sup = np.hstack((borders[0] - all_pos[:n, 0],
                     borders[1] - all_pos[:n, 1]))
    bounds = Bounds(inf, sup, keep_feasible=True)

    check_initial_violation(n, cons, collision_idx, all_entities)

    return const, bounds


def constraints_pre_cone(n, m, all_entities, max_speed):
    all_pos = np.array([[e.x, e.y] for e in all_entities])

    diff_pos = (np.broadcast_to(all_pos, (n, m, 2)) -
                np.broadcast_to(all_pos[0:n].reshape(n, 1, 2),
                                (n, m, 2)))

    dist = np.triu(np.sqrt(np.sum(diff_pos * diff_pos, axis=2)), k=1)
    all_radius = np.array([e.radius for e in all_entities])
    sum_rad = np.triu((np.broadcast_to(all_radius[0:n].reshape(n, 1), (n, m)) +
                       np.broadcast_to(all_radius, (n, m))), k=1)
    sum_rad = sum_rad + RAD_FUDGE

    dist_safe = (dist + np.tril(np.ones(dist.shape)))
    a = sum_rad / dist_safe
    one_min_a2 = 1 - a ** 2
    one_min_a2 = np.max((np.zeros(one_min_a2.shape) + 1e-4, one_min_a2), axis=0)
    sqrt_one_min_a2 = np.sqrt(one_min_a2)

    main_component = diff_pos * one_min_a2.reshape((n, m, 1))
    sub_component = orth(diff_pos) * (a * sqrt_one_min_a2).reshape((n, m, 1))
    c1 = main_component + sub_component
    c2 = main_component - sub_component
    c3 = (c2 + c1) / 2
    d3 = np.sqrt(np.sum(c3 ** 2, axis=2))
    u3 = norm_triangle(c3)
    u2 = norm_triangle(orth(c2))
    u1 = norm_triangle(-orth(c1))

    prev_vel = np.zeros((m, 2))
    if m > n:
        prev_vel[n:] = np.array([[e.vel.x, e.vel.y] for e in all_entities[n:]])
    prev_vel_norm = np.sqrt(np.sum(prev_vel ** 2, axis=1))
    counter = iter(range(n * m))
    speed = np.hstack((np.ones((n, n)) * 2, np.ones((n, m - n)))) * max_speed
    reachable = dist - all_radius - speed - prev_vel_norm
    collision_idx = [(next(counter), i, j) for i in range(n) for j in range(m)
                     if
                     i < j and reachable[i, j] < 0]

    speed_sq = max_speed ** 2
    return (all_pos, diff_pos, sum_rad, u1, u2, u3, d3, speed_sq,
            collision_idx, prev_vel)


def const_RVO_cone(diff_pos, sum_rad, u1, u2, u3, d3, speed_sq,
                   collision_idx, prev_vel):
    """

    :return: the constraint function
    :rtype: Callable
    """

    n, m, _ = u1.shape

    def const(x):
        vel = np.zeros((m, 2))
        vel[:n, 0] = x[:n]
        vel[:n, 1] = x[n:]
        vel = vel + prev_vel
        diff_vel = (np.broadcast_to(vel[0:n].reshape(n, 1, 2), (n, m, 2)) -
                    np.broadcast_to(vel, (n, m, 2)))
        c1 = np.sum(u1 * diff_vel, axis=2)
        c2 = np.sum(u2 * diff_vel, axis=2)
        c3 = np.sum(u3 * diff_vel, axis=2) - d3

        c4 = sum_rad - np.sqrt(np.sum((diff_pos - diff_vel) ** 2, axis=2))

        active_cons = np.max((c4, np.min((c1, c2, c3), axis=0)), axis=0)

        rvo = [active_cons[i, j] for ind, i, j in collision_idx]
        speed_limit = np.sum(vel[:n] ** 2, axis=1) - speed_sq

        cons = np.hstack((np.array(rvo), speed_limit))
        return cons

    return const


def jac_RVO_cone(diff_pos, sum_rad, u1, u2, u3, d3, collision_idx, prev_vel):
    n, m, _ = u1.shape
    grads = [u1, u2, u3]

    def jac(x):
        vel = np.zeros((m, 2))
        vel[:n, 0] = x[:n]
        vel[:n, 1] = x[n:]
        vel = vel + prev_vel
        diff_vel = (np.broadcast_to(vel[0:n].reshape(n, 1, 2), (n, m, 2)) -
                    np.broadcast_to(vel, (n, m, 2)))
        c1 = np.sum(u1 * diff_vel, axis=2)
        c2 = np.sum(u2 * diff_vel, axis=2)
        c3 = np.sum(u3 * diff_vel, axis=2) - d3
        c4 = sum_rad - np.sqrt(np.sum((diff_pos - diff_vel) ** 2, axis=2))
        cons = [c1, c2, c3]
        min_ind = np.argmin(cons, axis=0)
        max_ind = np.argmax((c4, np.min(cons, axis=0)), axis=0)
        # Big trick: use the fact that if one of the initial constant from
        # the min is the max then argmin is 1 which is convenient to multiply
        #  with the argmin matrix to get detailed knowledge on the most
        # active condition
        active_cons = max_ind * (min_ind + 1)

        list_grad = [2 * (diff_pos - diff_vel)] + grads

        rvo = [active_cons[i, j] for ind, i, j in collision_idx]
        grad = np.zeros((len(collision_idx) + n, n, 2))
        for idx, i, j in collision_idx:
            grad[idx, i] = grad[idx, i] + list_grad[rvo[idx]][i, j]
            if j < n:
                grad[idx, j] = grad[idx, j] - list_grad[rvo[idx]][i, j]
        for i in range(n):
            grad[len(collision_idx) + i, i] = 2 * vel[i]

        grad = grad.reshape((len(collision_idx) + n, 2 * n))
        return grad

    return jac


def hess_RVO_cone(n):
    def hess(x, v):
        hessian = np.diag(np.hstack((2 * v[-n:], 2 * v[-n:])))
        return hessian

    return hess

def norm_triangle(vect):
    n, m, _ = vect.shape
    norm = (np.sqrt(np.sum(vect ** 2, axis=2)) +
            np.tril(np.ones((n, m)))).reshape((n, m, 1))
    return vect / norm


def orth(vect):
    """
    Compute orthogonal on the last axis
    :param vect:
    :type vect:
    :return:
    :rtype:
    """
    ort = np.zeros(vect.shape)
    if len(vect.shape) == 3:
        ort[:, :, 0] = -vect[:, :, 1]
        ort[:, :, 1] = vect[:, :, 0]
    elif len(vect.shape) == 2:
        ort[:, 0] = -vect[:, 1]
        ort[:, 1] = vect[:, 0]
    else:
        ort[0] = -vect[1]
        ort[1] = vect[0]
    return ort


def check_initial_violation(n, cons, collision_idx, all_entities):
    pos_const_ind = [i for i, c in enumerate(cons(np.zeros(2 * n)))
                     if c > FUDGE]
    for ind in pos_const_ind:
        _, i, j = collision_idx[ind]
        logging.info("Conflicting constraint from:\n"
                     "Entity1: {}\nEntity2: {}".format(all_entities[i],
                                                       all_entities[j]))
