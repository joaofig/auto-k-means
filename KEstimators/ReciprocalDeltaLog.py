import math


class KEstimator:
    """Estimates the correct value for K using the reciprocal delta log rule"""
    def __init__(self, cluster_fn):
        self.K = 0
        self.cluster = cluster_fn
        self.s_k = dict()

    def fit(self, X, max_k=50):
        r_k = dict()
        max_val = float('-inf')

        for k in range(1, max_k + 1):
            self.s_k[k] = self.cluster(X, k)
            r_k[k] = 1.0 / self.s_k[k]

            if k > 1:
                d = (r_k[k] - r_k[k-1]) / math.log(k)
                if d > max_val:
                    max_val = d
                    self.K = k
        return self.K

    def fit_s_k(self, s_k, max_k=50):
        r_k = dict()
        max_val = float('-inf')

        for k in range(1, max_k + 1):
            r_k[k] = 1.0 / s_k[k]

            if k > 1:
                d = (r_k[k] - r_k[k-1]) / math.log(k)
                if d > max_val:
                    max_val = d
                    self.K = k
        self.s_k = s_k
        return self.K
