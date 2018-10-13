import numpy as np
import pandas as pd


class KEstimator:
    """Estimates the best value for K using Pham-Dimov-Nguyen method"""
    def __init__(self, cluster_fn=None):
        self.K = 0
        self.cluster = cluster_fn
        self.s_k = dict()
        self.a_k = dict()

    def alpha_k(self, k, dim):
        if k == 2:
            ak = 1.0 - 3.0 / (4.0 * dim)
        else:
            ak1 = self.a_k[k - 1]
            ak = ak1 + (1.0 - ak1) / 6.0
        return ak

    def cluster_eval(self, k, dim):
        if k == 1 or self.s_k[k - 1] == 0.0:
            return 1.0

        self.a_k[k] = self.alpha_k(k, dim)
        if self.s_k[k - 1] != 0.0:
            return self.s_k[k] / (self.a_k[k] * self.s_k[k - 1])
        else:
            return 1.0

    def fit(self, X, max_k=50):
        self.s_k = dict()
        self.a_k = dict()
        f_k = np.ones(max_k)

        if isinstance(X, pd.DataFrame):
            dim = len(X.columns)
        elif isinstance(X, np.ndarray):
            dim = X.shape[1]
        else:
            print(type(X))

        for k in range(1, max_k + 1):
            i = k - 1
            self.s_k[k] = self.cluster(X, k)
            f_k[i] = self.cluster_eval(k, dim)
        k = np.argmin(f_k)

        if f_k[k] <= 0.85:
            self.K = np.argmin(f_k) + 1
        else:
            self.K = 1
        return self.K

    def fit_s_k(self, s_k, max_k=50, dim=2):
        """Fits the value of K using the s_k series"""
        self.a_k = dict()
        f_k = np.ones(max_k)

        for k in range(1, max_k + 1):
            i = k - 1
            self.s_k[k] = s_k[k]
            f_k[i] = self.cluster_eval(k, dim)
        k = np.argmin(f_k)

        if f_k[k] <= 0.85:
            self.K = np.argmin(f_k) + 1
        else:
            self.K = 1
        return self
