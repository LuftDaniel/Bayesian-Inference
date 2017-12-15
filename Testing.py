import numpy as np
import LS_SVM

if __name__ == '__main__':
    data_dict = {-1: np.array([[1, 2],
                               [1, 1],
                               [2, 1],
                               [3, 4]]),
                 1: np.array([[2, -3],
                              [0, -3],
                              [1, -1],
                              [1.5, -2],
                              [4, 2]])}

    k1 = LS_SVM.Kernel('lin', param=[None, None], hyperparam=[1, 3])
    k2 = LS_SVM.Kernel('pol', param=[1, 3], hyperparam=[4, 1])
    k3 = LS_SVM.Kernel('rad', param=[10, None], hyperparam=[1, 3])
    k4 = LS_SVM.Kernel('sig', param=[0.1, 1], hyperparam=[1, 1])

    alpha = LS_SVM.solve(data_dict, [k1, k2, k3, k4], plot=True)
