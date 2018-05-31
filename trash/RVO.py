"""
author: MengGuo for initial version, modified by Guillaume Plays
"""

# TODO modify for use as navigation system

import math
from math import cos, sin, atan2, asin
from math import pi as PI
from math import sqrt

import numpy
import numpy as np
from numpy import linalg as la

from hlt.entity import Entity, Position


def velocity_constraint_parabola(entity1, entity2, max_speed):
    origin = Position.from_entity(entity1)
    b = Position.from_entity(entity2) - origin

    r = entity1.radius + entity2.radius

    norm_b = Position.normed(b)
    orth_b = Position.orth(norm_b)

    apex = b - norm_b * r

    # velocity difference as if the apex was the origin,
    # simplify operations later
    diff_vel = entity1.vel - entity2.vel - apex

    constraint = diff_vel.dot(norm_b) ** 2 - 2 * r * diff_vel.dot(orth_b) < 0

    return constraint


def check_velocity_constraint(entity1, entity2, max_speed):
    """

    :param entity1:
    :type entity1: Entity
    :param entity2:
    :type entity2: Entity
    :return:
    :rtype:
    """
    origin = Position.from_entity(entity1)
    b = Position.from_entity(entity2) - origin
    d = math.sqrt(b.x ** 2 + b.x ** 2)
    r = entity1.radius + entity2.radius

    if d - r < 2 * max_speed:
        p1, p2 = find_adhesion_cone_circle_centered(b, r)
        # diff_vel = entity1.vel - entity2.vel

        # orthogonal because tangent definition
        ray_normals = [b - p1, b - p2]

        np.array([[ray_normals[0].x, ray_normals[0].y],
                  [ray_normals[0].x, ray_normals[0].y]])

        def constraint(diff_vel):
            return (ray_normals[0] * diff_vel < 0 or
                    ray_normals[1] * diff_vel < 0 or
                    b * diff_vel < 1 - r / d)
    else:
        constraint = None
    return constraint


def find_adhesion_cone_circle_centered(b, r):
    """
    Compute the coordinates of the intersection between the circle of center
    b and radius r and its tangent passing by a. A is at the origin of the space

    :param b: Position of point B
    :type b: Position
    :param r: Sum of the radius of A and B
    :type r: float
    :return: The coordinates of the intersections
    :rtype: tuple(Position)
    """
    d2 = b.x ** 2 + b.x ** 2
    r2 = r ** 2
    norm_c_2 = d2 - r2
    norm_c = math.sqrt(norm_c_2)

    # NB can be solved with determinant computation
    # Solving using scalar product and vectorial product and using the
    # the geometrical projection to have explicit expression of sin and cos
    p1 = Position(*solve_x_y([b.x, -b.x],
                             [b.x, b.x],
                             [norm_c_2, norm_c * r]))

    p2 = Position(*solve_x_y([b.x, b.x],
                             [b.x, -b.x],
                             [norm_c_2, norm_c * r]))

    return (p1, p2)


def find_adhesion_cone_circle(a, b, r):
    """
    Compute the coordinates of the intersection between the circle of center
    b and radius r and its tangent passing by a

    :param a: Position of point A
    :type a: (float,float)
    :param b: Position of point B
    :type b: (float,float)
    :param r: Sum of the radius of A and B
    :type r: float
    :return: The coordinates of the intersections
    :rtype:
    """
    xA, yA = a
    xB, yB = b
    xB -= xA
    yB -= yA
    d2 = xB ** 2 + yB ** 2
    r2 = r ** 2
    norm_c_2 = d2 - r2
    norm_c = math.sqrt(norm_c_2)

    # NB can be solved with determinant computation
    x1C, y1C = solve_x_y([xB, -yB], [yB, xB], [norm_c_2, norm_c * r])

    x2C, y2C = solve_x_y([xB, yB], [yB, -xB], [norm_c_2, norm_c * r])

    adhesion_x_y = [(x + xA, y + yA) for x, y in [(x1C, y1C), (x2C, y2C)]]
    return adhesion_x_y


def solve_x_y(a, b, c):
    n = b[1] * a[0] - b[0] * a[1]
    y = (c[1] * a[0] - c[0] * a[1]) / n
    x = (c[1] * b[0] - c[0] * b[1]) / -n
    return x, y


def distance(pose1, pose2):
    """ compute Euclidean distance for 2D """
    return sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2) + 0.001


def RVO_update(X, V_des, V_current,
               list_obstacles=None,
               agent_radius=0.5,
               mode=None):
    """
    Compute best velocity given the desired velocity, current velocity
    and workspace model.
    Warning: O(N²) algorithm, should trim to conflicting sets.
    Using Reciprocal Velocity object model.
    :param X:
    :type X:
    :param V_des:
    :type V_des:
    :param V_current:
    :type V_current:
    :param mode: algorithm used from VO, RVO, HRVO. Assumed HRVO
    :type mode:
    :return :
    :type return:

    """
    # Taking some margin
    agent_radius = agent_radius * 1.01

    V_opt = list(V_current)
    for i in range(len(X)):
        vA = [V_current[i][0], V_current[i][1]]
        pA = [X[i][0], X[i][1]]
        RVO_BA_all = []
        for j in range(len(X)):
            if i != j:
                vB = [V_current[j][0], V_current[j][1]]
                pB = [X[j][0], X[j][1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])

                # Removing margin if too close, avoid bugs
                if 2 * agent_radius > dist_BA:
                    dist_BA = 2 * agent_radius

                theta_BAort = asin(2 * agent_radius / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                theta_ort_right = theta_BA - theta_BAort
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]

                if mode == "RVO":
                    transl_vB_vA = [pA[0] + 0.5 * (vB[0] + vA[0]),
                                    pA[1] + 0.5 * (vB[1] + vA[1])]
                elif mode == 'VO':
                    transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                else:  # HRVO
                    dist_dif = distance([0.5 * (vB[0] - vA[0]),
                                         0.5 * (vB[1] - vA[1])],
                                        [0, 0])
                    transl_vB_vA = [
                        pA[0] + vB[0] + cos(theta_ort_left) * dist_dif,
                        pA[1] + vB[1] + sin(theta_ort_left) * dist_dif]

                RVO_BA = [transl_vB_vA,
                          bound_left,
                          bound_right,
                          dist_BA,
                          2 * agent_radius]
                RVO_BA_all.append(RVO_BA)
        for hole in list_obstacles:
            # hole = [x, y, rad]
            vB = [0, 0]
            pB = hole[0:2]
            transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
            dist_BA = distance(pA, pB)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            # over-approximation of square to circular
            OVER_APPROX_C2S = 1.5
            rad = hole[2] * OVER_APPROX_C2S
            if (rad + agent_radius) > dist_BA:
                dist_BA = rad + agent_radius
            theta_BAort = asin((rad + agent_radius) / dist_BA)
            theta_ort_left = theta_BA + theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA - theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA,
                      rad + agent_radius]
            RVO_BA_all.append(RVO_BA)

        vA_post = intersect(pA, V_des[i], RVO_BA_all)
        V_opt[i] = vA_post[:]

    return V_opt


def intersect(pA, vA, RVO_BA_all):
    # print '----------------------------------------'
    # print 'Start intersection test'
    norm_v = distance(vA, [0, 0])
    suitable_V = []
    unsuitable_V = []
    for theta in numpy.arange(0, 2 * PI, 0.1):
        for rad in numpy.arange(0.02, norm_v + 0.02, norm_v / 5.0):
            new_v = [rad * cos(theta), rad * sin(theta)]
            suit = True
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    suit = False
                    break
            if suit:
                suitable_V.append(new_v)
            else:
                unsuitable_V.append(new_v)
    new_v = vA[:]
    suit = True
    for RVO_BA in RVO_BA_all:
        p_0 = RVO_BA[0]
        left = RVO_BA[1]
        right = RVO_BA[2]
        dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
        theta_dif = atan2(dif[1], dif[0])
        theta_right = atan2(right[1], right[0])
        theta_left = atan2(left[1], left[0])
        if in_between(theta_right, theta_dif, theta_left):
            suit = False
            break
    if suit:
        suitable_V.append(new_v)
    else:
        unsuitable_V.append(new_v)
    # ----------------------
    if suitable_V:
        # print 'Suitable found'
        vA_post = min(suitable_V, key=lambda v: distance(v, vA))
        new_v = vA_post[:]
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
    else:
        # print 'Suitable not found'
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            tc_V[tuple(unsuit_v)] = 0
            tc = []
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dist = RVO_BA[3]
                rad = RVO_BA[4]
                dif = [unsuit_v[0] + pA[0] - p_0[0],
                       unsuit_v[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    small_theta = abs(
                            theta_dif - 0.5 * (theta_left + theta_right))
                    if abs(dist * sin(small_theta)) >= rad:
                        rad = abs(dist * sin(small_theta))
                    big_theta = asin(abs(dist * sin(small_theta)) / rad)
                    dist_tg = abs(dist * cos(small_theta)) - abs(
                            rad * cos(big_theta))
                    if dist_tg < 0:
                        dist_tg = 0
                    tc_v = dist_tg / distance(dif, [0, 0])
                    tc.append(tc_v)
            tc_V[tuple(unsuit_v)] = min(tc) + 0.001
        WT = 0.2
        vA_post = min(unsuitable_V,
                      key=lambda v: ((WT / tc_V[tuple(v)]) + distance(v, vA)))
    return vA_post


def in_between(theta_right, theta_dif, theta_left):
    if abs(theta_right - theta_left) <= PI:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left < 0) and (theta_right > 0):
            theta_left += 2 * PI
            if theta_dif < 0:
                theta_dif += 2 * PI
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left > 0) and (theta_right < 0):
            theta_right += 2 * PI
            if theta_dif < 0:
                theta_dif += 2 * PI
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False


def compute_V_des(X, goal, V_max):
    V_des = []
    for i in range(len(X)):
        dif_x = [goal[i][k] - X[i][k] for k in range(2)]
        norm = distance(dif_x, [0, 0])
        norm_dif_x = [dif_x[k] * V_max[k] / norm for k in range(2)]
        V_des.append(norm_dif_x[:])
        if reach(X[i], goal[i], 0.1):
            V_des[i][0] = 0
            V_des[i][1] = 0
    return V_des


def reach(p1, p2, bound=0.5):
    if distance(p1, p2) < bound:
        return True
    else:
        return False
