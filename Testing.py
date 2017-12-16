import warnings
import numpy as np
import LS_SVM
from sklearn import datasets
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

X, y = datasets.make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                    random_state=7, class_sep=1.5, flip_y=0.1)

if __name__ == '__main__':
    data_dict = {1: np.array([[1, 2],
                              [1, 1],
                              [3, 4],
                              [4, 2],
                              [2, 1]]),
                 -1: np.array([[2, -3],
                               [0, -3],
                               [1, -1],
                               [1.5, -2]])}
    '''
    data_dict = {1: np.array([[0, 0],
                              [1, 0]]),
                 -1: np.array([[0, 1],
                              [1, 2]])}
    '''

    k1 = LS_SVM.Kernel('lin', param=[None, None], hyperparam=[1, 1])
    k2 = LS_SVM.Kernel('pol', param=[1, 2], hyperparam=[1, 1])
    k3 = LS_SVM.Kernel('rad', param=[10, None], hyperparam=[1, 1])
    k4 = LS_SVM.Kernel('sig', param=[0.1, 10], hyperparam=[1, 3])

    M1 = np.array([])
    M2 = np.array([])
    j = 1
    k = 1
    for i, item in enumerate(X):
        if y[i] == 1:
            M1 = np.reshape(np.append(M1, item), (j, 2))
            j += 1
        else:
            M2 = np.reshape(np.append(M2, item), (k, 2))
            k += 1
    data_dict_new = {1: M1, -1: M2}
    LS_SVM.gesamt(data_dict_new, [k1, k2, k3, k4])
