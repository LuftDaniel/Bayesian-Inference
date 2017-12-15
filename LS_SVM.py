import numpy as np
from numpy import linalg
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


def solve(data_dict, kern, plot=False):
    data = np.concatenate((data_dict[1], data_dict[-1]), axis=0)
    data_y = np.append(np.ones([np.size(data_dict[1], axis=0), 1]), -np.ones([np.size(data_dict[-1], axis=0), 1]))
    size_x = np.size(data_y)

    sol = np.array([])
    e = np.array([])
    for i, k in enumerate(kern):
        gamma = k.hyperparam[0] / k.hyperparam[1]
        h1 = np.append([0], data_y)
        omega = np.multiply(k.kernel(data, data), np.outer(data_y, data_y)) \
                + 1 / gamma * np.eye(size_x)
        h2 = np.concatenate(([data_y], omega)).T

        A = np.append([h1], h2, axis=0)
        b = np.append([0], np.ones([1, size_x]))
        alpha = linalg.solve(A, b)

        eig, vs = linalg.eig(omega)
        print(eig)

        sol = np.reshape(np.append(sol, alpha), (i + 1, size_x + 1))

        if plot:
            plt.figure(i)
            plt.scatter(data_dict[1][:, 0], data_dict[1][:, 1], color='b')
            plt.scatter(data_dict[-1][:, 0], data_dict[-1][:, 1], color='r')

            x = np.arange(np.min(data[:, 0]) - 4, np.max(data[:, 0]) + 4, 0.5)
            y = np.arange(np.min(data[:, 1]) - 4, np.max(data[:, 1]) + 4, 0.5)

            X, Y = np.meshgrid(x, y)

            Z = np.zeros([np.size(y), np.size(x)])
            for i, item in enumerate(x):
                for j, jtem in enumerate(y):
                    Z[j, i] = decision(alpha, data, data_y, [[item, jtem]], kern=k)

            plt.contour(X, Y, Z, [0])

    plt.show()
    return sol
