import numpy as np

def simple_objective_function(n, m, all_entities):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param all_entities: List of all_entities
    :type all_entities: [Entity]
    :return:
    :rtype:
    """
    all_pos = np.array([[e.x, e.y] for e in all_entities])
    factors = np.array([entity.factors for entity in
                        all_entities])
    kernel_widths = np.array([entity.kernels for entity in all_entities])
    kernel_mult = 1 / kernel_widths ** 2

    fun_grad = sum_gaussians(n, m, all_pos, factors, kernel_mult)

    f, g = fun_grad(np.zeros(2 * n))

    def fun_grad_simple(x):
        fun = np.sum(x * g)
        grad = g
        return fun, grad

    def hess(x):
        return np.zeros((2 * n, 2 * n))

    return fun_grad_simple, hess


def objective_function(n, m, all_entities):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param all_entities: List of all_entities
    :type all_entities: [Entity]
    :return:
    :rtype:
    """
    all_pos = np.array([[e.x + e.vel.x, e.y + e.vel.y] for e in all_entities])
    factors = np.array([entity.factors for entity in all_entities])
    kernel_widths = np.array([entity.kernels for entity in all_entities])
    kernel_mult = 1 / kernel_widths ** 2

    targets = np.array([[e.target.x, e.target.y] for e in all_entities[:n]])

    grad_target = all_pos[:n] - targets
    norm = np.sum(grad_target ** 2, axis=1).reshape((n, 1))
    grad_target = grad_target / norm
    grad_target = grad_target.reshape((2 * n))

    fun_grad = sum_gaussians(n, m, all_pos, factors, kernel_mult, grad_target)

    hess = hess_gaussian(n, m, all_pos, factors, kernel_mult, )

    return fun_grad, hess


def hess_gaussian(n, m, all_pos, factors, kernels):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param all_pos: Matrix of position of shape(n,m,2)
    :type all_pos: ndarray
    :param factors: Array of factor (weight) for gaussian energy profile
    :type factors: ndarray
    :param kernels: Array of kernel multipliers for gaussian energy profile
    :type kernels: ndarray
    :return:
    :rtype:
    """

    def hess(x):
        vel = np.zeros((m, 2))
        vel[:n, 0] = x[:n]
        vel[:n, 1] = x[n:]
        next_pos = all_pos + vel
        diff_pos = (np.broadcast_to(next_pos[0:n].reshape(n, 1, 2), (n, m, 2)) -
                    np.broadcast_to(next_pos, (n, m, 2)))
        dist = np.sum(diff_pos**2, axis=2)
        my_hess = np.zeros((2 * n, 2 * n))
        for i in range(2):
            fun_mat = factors[:, i] * np.exp(-dist * kernels[:, i])
            fun_mat = fun_mat - np.hstack((np.diag(np.diag(fun_mat)),
                                           np.zeros((n, m - n))))

            kernel_sq = kernels[:, i] ** 2
            x_x = ((-4 * kernel_sq * diff_pos[:, :, 0] ** 2 + 2 * kernels[:, i])
                   * fun_mat)
            y_y = ((-4 * kernel_sq * diff_pos[:, :, 1] ** 2 + 2 * kernels[:, i])
                   * fun_mat)
            x_y = (-4 * kernel_sq * diff_pos[:, :, 0] * diff_pos[:, :, 1] +
                   2 * kernels[:, i]) * fun_mat
            y_x = x_y

            my_hess = my_hess + np.hstack((np.vstack((x_x[:, :n], y_x[:, :n])),
                                           np.vstack((x_y[:, :n], y_y[:, :n]))))

            # Computation of diagonal is different from the rest
            diag = np.diag(np.hstack((np.sum(x_x, axis=1),
                                      np.sum(y_y, axis=1))))
            my_hess = my_hess - diag

        return my_hess

    return hess


def sum_gaussians(n, m, all_pos, factors, kernels, grad_target):
    """

    :param n: Number of moving ships
    :type n: int
    :param m: Number of entities
    :type m: int
    :param all_pos: Matrix of position of shape(n,m,2)
    :type all_pos: ndarray
    :param factors: Array of factor (weight) for gaussian energy profile
    :type factors: ndarray
    :param kernels: Array of kernel multipliers for gaussian energy profile
    :type kernels: ndarray
    :return:
    :rtype:
    """

    def fun_grad(x):
        vel = np.zeros((m, 2))
        vel[:n, 0] = x[:n]
        vel[:n, 1] = x[n:]
        next_pos = all_pos + vel
        diff_pos = (np.broadcast_to(next_pos[0:n].reshape(n, 1, 2), (n, m, 2)) -
                    np.broadcast_to(next_pos, (n, m, 2)))
        dist = np.sum(diff_pos**2, axis=2)
        fun = 0
        grad = np.zeros(2 * n)
        for i in range(2):
            fun_mat = factors[:, i] * np.exp(-dist * kernels[:, i])

            fun_mat = fun_mat - np.hstack((np.diag(np.diag(fun_mat)),
                                           np.zeros((n, m - n))))

            grad_mat_x = -2 * kernels[:, i] * diff_pos[:, :, 0] * fun_mat
            grad_mat_y = -2 * kernels[:, i] * diff_pos[:, :, 1] * fun_mat
            grad = grad + np.hstack((np.sum(grad_mat_x, axis=1),
                                     np.sum(grad_mat_y, axis=1)))

            # The divide by 2 operation comes from the mutual energy wich has to
            # be accounted for in ship-ship interaction. Allow simpler gradient
            fun += np.sum(fun_mat[:, :n]) / 2 + np.sum(fun_mat[:, n:])
        fun += np.sum(x * grad_target)
        grad += grad_target
        return fun, grad

    return fun_grad
