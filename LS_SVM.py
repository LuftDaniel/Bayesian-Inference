import numpy as np
from numpy import linalg
from scipy import optimize as opt
import matplotlib.pyplot as plt


class Kernel:
    def __init__(self, kern, param=(None, None), hyperparam=(1, 1)):
        self.kernel = {
            'lin': self.linear,
            'pol': self.polynomial,
            'rad': self.radial,
            'sig': self.sigmoid
        }.get(kern, self.linear)
        self.param = param
        self.hyperparam = hyperparam
        self.omega = np.array([])
        self.eig = np.array([])

    def linear(self, u, v):
        return np.inner(u, v)

    def polynomial(self, u, v):
        if self.param[0] is None:
            theta = 1
        else:
            theta = self.param[0]
        if self.param[1] is None:
            grad = 3
        else:
            grad = self.param[1]
        return (np.multiply(theta, np.ones(np.size(v, axis=0))) + np.inner(u, v)) ** grad

    def radial(self, u, v):
        if self.param[0] is None:
            theta = 0.1
        else:
            theta = self.param[0]
        diff = np.transpose(u) - np.tile(np.expand_dims(v, 2), u.shape[0])
        return np.transpose(np.exp(- 1.0 / theta * np.sum(np.square(diff), axis=1)))

    def sigmoid(self, u, v):
        if self.param[0] is None:
            alpha = 1
        else:
            alpha = self.param[0]
        if self.param[1] is None:
            c = 1
        else:
            c = self.param[1]
        return np.tanh(np.multiply(alpha, np.inner(u, v)) + c)


def decision(a, x, y, v, kern):
    return np.dot(np.multiply(y, a[1:]), kern.kernel(x, v)) + a[0]


def gesamt(data_dict, kern):
    data = np.concatenate((data_dict[1], data_dict[-1]), axis=0)
    data_y = np.append(np.ones([np.size(data_dict[1], axis=0), 1]), -np.ones([np.size(data_dict[-1], axis=0), 1]))
    size_x = np.size(data_y)

    def centering(mat):
        M = np.eye(size_x) + np.multiply(1.0 / size_x, np.ones([size_x, size_x]))
        return np.dot(M, np.dot(mat, M))

    for i, k in enumerate(kern):
        k.omega = k.kernel(data, data)
        eig, v = linalg.eig(centering(k.omega))
        j = 0
        for e in eig:
            if e <= 1e-6:
                eig = np.delete(eig, j)
            else:
                j += 1
        k.eig = eig

    ker = kern
    while np.size(ker) > 1:
        alpha = solve(data_dict, ker, plot=True)
        solve2(alpha, ker)
        ker = solve3(ker)
    solve(data_dict, ker, plot=True)
    pass


def solve3(kern):
    min = np.inf
    min_i = 0
    for i, k in enumerate(kern):
        n_eff = np.size(k.eig)
        n = np.size(k.omega, axis=0)
        gamma_eff = 1 + np.sum(np.divide(np.multiply(k.hyperparam[1], k.eig),
                                         (np.add(k.hyperparam[0], np.multiply(k.hyperparam[1], k.eig)))))
        evi = np.sqrt((k.hyperparam[0] ** n_eff * k.hyperparam[1] ** (n - 1)) /
                      ((gamma_eff - 1) * (n - gamma_eff)) * np.prod(np.add(k.hyperparam[0],
                                                                           np.multiply(k.hyperparam[1], k.eig))))
        if evi < min:
            min = evi
            min_i = i
    if np.size(kern) > 1:
        ker = np.delete(kern, min_i)
    else:
        ker = kern
    return ker


def solve2(alpha, kern):
    cap = 1e-9
    sol = np.array([])
    for i, k in enumerate(kern):
        omega = k.omega
        eig = k.eig
        n_eff = np.size(eig)
        n = np.size(omega, axis=0)

        def fun(hyper):
            h1 = hyper[0] / 2.0 * np.dot(alpha[i, 1:], np.dot(omega, alpha[i, 1:]))
            h2 = hyper[1] / 2.0 * (k.hyperparam[0] / k.hyperparam[1]) ** 2 * np.dot(alpha[i, 1:], alpha[i, 1:])
            h3 = 1.0 / 2.0 * np.sum(np.log(np.add(hyper[0], np.multiply(hyper[1], eig))))
            h4 = - n_eff / 2.0 * np.log(hyper[0]) - (n - 1) / 2.0 * np.log(hyper[1])
            return np.sum([h1, h2, h3, h4])

        # res = opt.fmin_cg(fun, np.array([k.hyperparam[0], k.hyperparam[1]]), full_output=False)
        res = opt.fmin(fun, np.array([k.hyperparam[0], k.hyperparam[1]]), disp=False)
        res = np.array([np.max([res[0], cap]), np.max([res[1], cap])])
        sol = np.reshape(np.append(sol, res), (i + 1, 2))
        k.hyperparam = sol[i]
    pass


def solve(data_dict, kern, plot=False):
    data = np.concatenate((data_dict[1], data_dict[-1]), axis=0)
    data_y = np.append(np.ones([np.size(data_dict[1], axis=0), 1]), -np.ones([np.size(data_dict[-1], axis=0), 1]))
    size_x = np.size(data_y)

    sol = np.array([])
    for i, k in enumerate(kern):
        gamma = k.hyperparam[1] / k.hyperparam[0]
        h1 = np.append([0], data_y)
        omega_ = np.multiply(k.omega, np.outer(data_y, data_y)) \
                 + 1 / gamma * np.eye(size_x)
        h2 = np.concatenate(([data_y], omega_)).T

        A = np.append([h1], h2, axis=0)
        b = np.append([0], np.ones([1, size_x]))
        alpha = linalg.solve(A, b)

        sol = np.reshape(np.append(sol, alpha), (i + 1, size_x + 1))

        if plot:
            plt.figure(i)
            plt.scatter(data_dict[1][:, 0], data_dict[1][:, 1], color='b')
            plt.scatter(data_dict[-1][:, 0], data_dict[-1][:, 1], color='r')

            x = np.arange(np.min(data[:, 0]) - 10, np.max(data[:, 0]) + 10, 0.25)
            y = np.arange(np.min(data[:, 1]) - 10, np.max(data[:, 1]) + 10, 0.25)

            X, Y = np.meshgrid(x, y)

            Z = np.zeros([np.size(y), np.size(x)])
            for i, item in enumerate(x):
                for j, jtem in enumerate(y):
                    Z[j, i] = decision(alpha, data, data_y, [[item, jtem]], kern=k)

            plt.contour(X, Y, Z, [0])

    plt.show()
    return sol
