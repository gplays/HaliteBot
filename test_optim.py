import math

import numpy as np
import scipy.optimize as opt


def f(x):
    print(x[0])
    return 2 * math.log(x[0] ** 2) ** 2 + 3 * x[0] + 1

def clean_repository():
    # clean repository of log and replays
    # keep one victory one defeat
    pass

def launch_gym():
    # create log_folder with parameters

    # launch training

    # log results
    clean_repository()
    pass


if __name__ == "__main__":
    a = opt.fmin_l_bfgs_b(f, np.array([15, 1]),
                          bounds=[(None, None), (None, None)],
                          epsilon=0.001,
                          approx_grad=True)
