import math
import unittest

import cv2
import numpy as np
import pickle
import hlt.navigation_optim as nav
from hlt.entity import Entity, Ship, Planet
from utils.param_handling import map_parameters


class TestNavigation(unittest.TestCase):
    """
    Test the add function from the mymath library
    """

    def setUp(self):
        e1 = Entity(100, 100, 5, 10, 1, 1)
        e2 = Entity(100, 150, 10, 10, 1, 2)
        e3 = Entity(20, 40, 5, 10, 1, 3)
        e4 = Entity(100, 50, 30, 10, 1, 4)
        self.entities = [e1, e2, e3, e4]
        self.max_speed = 50

    def _test_constraints_pre(self):
        m = 2
        n = 1
        all_entities = self.entities[:m]
        unit_vec, apex, rad, all_pos, dist = nav.constraints_pre(n, m,
                                                                 all_entities)
        for e in all_entities:
            print(e)
        print("unit_vec", unit_vec)
        print("apex", apex[0, 1])
        print("rad", rad)
        print("all_pos", all_pos)
        print("dist ", dist)

    def _test_RVO_constraints_1call(self):
        m = 2
        n = 1
        e1 = Entity(100, 100, 5, 10, 1, 1)
        e2 = Entity(100, 150, 10, 10, 1, 2)
        all_entities = [e1, e2]
        unit_vec, apex, rad, all_pos, dist = nav.constraints_pre(n, m,
                                                                 all_entities)

        cons = nav.const_RVO(n, m, unit_vec, dist, rad, self.max_speed ** 2)

        print("value before apex", cons([0, 30]))
        print("value at apex", cons([0, 35]))
        print("value after apex", cons([0, 40]))

    def _test_RVO_jac_1call(self):
        m = 2
        n = 1
        e1 = Entity(100, 100, 5, 10, 1, 1)
        e2 = Entity(100, 150, 10, 10, 1, 2)
        all_entities = [e1, e2]
        unit_vec, apex, rad, all_pos, dist = nav.constraints_pre(n, m,
                                                                 all_entities)

        jac = nav.jac_RVO(n, m, unit_vec, rad)

        print("grad value before apex", jac([0, 30])[0])
        print("grad value at apex", jac([0, 35])[0])
        print("grad value after apex", jac([0, 40])[0])
        print("#######")
        print("grad value before apex", jac([15, 35])[0])
        print("grad value at apex", jac([0, 35])[0])
        print("grad value after apex", jac([-5, 40])[0])

    def _test_obj_fun_1call(self):
        m = 2
        n = 1

        e1 = Ship(player_id=1, ship_id=1, x=100, y=100, hp=100, vel_x=0,
                  vel_y=0, docking_status=0, planet=None, progress=0,
                  cooldown=0)
        e2 = Planet(planet_id=1, x=100, y=150, hp=100, radius=15,
                    docking_spots=2, current=0,
                    remaining=5, owned=0, owner=0, docked_ships=0)
        params = map_parameters(map_parameters([]).values())

        e1.augment(1, params)
        e2.augment(1, params)
        all_entities = [e1, e2]
        fun_grad, hess = nav.objective_function(n, m, all_entities)

        print(fun_grad([0, 10])[0])

    def _test_compute_1RVO_constraint(self):
        m = 4
        n = 1
        all_entities = self.entities[:m]
        unit_vec, apex, rad, all_pos, dist = nav.constraints_pre(n, m,
                                                                 all_entities)

        cons = nav.const_RVO(n, m, unit_vec, dist, rad, self.max_speed ** 2)

        x = self.entities[0].x
        y = self.entities[0].y

        n_cons = len(cons([0, 0]))
        img = [[all([cons([i - x, j - y])[k] < 0 for k in range(n_cons)])
                for i in range(200)]
               for j in range(200)]

        img = np.array(img, dtype=np.uint8)
        img = img * 255
        img = np.dstack((img, img, img))

        draw_entities(img, unit_vec, all_entities)

    def _test_grad_cons(self):
        m = 2
        n = 1
        all_entities = self.entities[:m]
        unit_vec, apex, rad, all_pos, dist = nav.constraints_pre(n, m,
                                                                 all_entities)

        jac = nav.jac_RVO(n, m, unit_vec, rad)

        x = self.entities[0].x
        y = self.entities[0].y
        r = self.entities[0].radius

        grad = [[jac([i - x, j - y])[0] for i in range(200)]
                for j in range(200)]
        grad = np.array(grad)
        grad[:, :, 0] = fit_255(grad[:, :, 0])
        grad[:, :, 1] = fit_255(grad[:, :, 1])

        img = np.zeros((200, 200, 3))
        img[:, :, :2] = np.array(grad)
        img = np.uint8(img)

        draw_entities(img, unit_vec, all_entities)

    def test_obj_fun(self):
        m = 3
        n = 1

        e1 = Ship(player_id=1, ship_id=1, x=100, y=100, hp=100, vel_x=0,
                  vel_y=0, docking_status=0, planet=None, progress=0,
                  cooldown=0)
        e2 = Planet(planet_id=1, x=100, y=150, hp=100, radius=15,
                    docking_spots=2, current=0,
                    remaining=5, owned=0, owner=0, docked_ships=0)
        e3 = Planet(planet_id=2, x=30, y=75, hp=100, radius=10,
                    docking_spots=7, current=0,
                    remaining=5, owned=0, owner=0, docked_ships=0)
        all_entities = [e1, e2, e3]
        params = map_parameters(map_parameters([]).values())

        for e in all_entities:
            e.augment(1, params)

        fun_grad, hess = nav.objective_function(n, m, all_entities)

        unit_vec, apex, rad, all_pos, dist = nav.constraints_pre(n, m,
                                                                 all_entities)

        x = self.entities[0].x
        y = self.entities[0].y

        img = [[fun_grad([i - x, j - y])[0] for i in range(200)]
               for j in range(200)]

        img = fit_255(np.array(img))

        img = np.dstack((img, img, img))

        draw_entities(img, unit_vec, all_entities)

    def _test_optimize(self):
        with open("pickled", "rb") as f:
            args = pickle.load(f)

        nav.navigate_optimize(*args)


def draw_entities(img, unit_vec, all_entities):
    x = all_entities[0].x
    y = all_entities[0].y
    r = all_entities[0].radius

    unit_vec_x, unit_vec_y = (int(i * 10) for i in unit_vec[0, 1])

    # Draw unit vect
    cv2.line(img, (x, y), (x + unit_vec_x, y + unit_vec_y),
             (255, 0, 0), 1, cv2.LINE_AA)

    # Draw all entities
    for e in all_entities:
        cv2.circle(img, (e.x, e.y), math.ceil(e.radius), (0, 0, 255), 1,
                   cv2.LINE_AA)

    # Draw all entities apparent radius for entity1
    for e in all_entities[1:]:
        cv2.circle(img, (e.x, e.y), math.ceil(e.radius + r), (0, 255, 0), 1,
                   cv2.LINE_AA)

    display_image(img)


def fit_255(nparray):
    return np.uint8(255 * (nparray - nparray.min()) /
                    (nparray.max() - nparray.min()))

def fit_255_log(nparray):
    print("max ",nparray.max())
    print("min ",nparray.min())
    print("mean ",nparray.mean())
    nparray = nparray -nparray.min()+1e-3
    nparray = np.log(nparray)
    return np.uint8(255 * (nparray - nparray.min()) /
                    (nparray.max() - nparray.min()))


def display_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.resizeWindow('image', 800, 600)

    cv2.waitKey()


if __name__ == '__main__':
    unittest.main()
