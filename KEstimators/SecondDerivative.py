from sklearn.cluster import MiniBatchKMeans


class KEstimator:

    def __init__(self):
        self.K = 0

    @staticmethod
    def calculate_s_k(X, k):
        km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
        return km.inertia_  # -km.score(df) #

    def fit(self, X):
        max_derivative = -1
        s_k = {1: self.calculate_s_k(X, 1), 2: self.calculate_s_k(X, 2)}
        for k in range(3, len(X) + 1):
            s_k[k] = self.calculate_s_k(X, k)

            d = s_k[k] - 2 * s_k[k-1] + s_k[k-2]

            if k == 3:
                max_derivative = d

            if d >= max_derivative:
                max_derivative = d
            else:
                self.K = k - 1
                break
