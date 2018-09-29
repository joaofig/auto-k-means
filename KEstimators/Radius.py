import math
from sklearn.cluster import MiniBatchKMeans

class KEstimator:

    def __init__(self):
        self.K = 0

    @staticmethod
    def calculate_s_k(X, k):
        km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
        return km.inertia_  # -km.score(df) #

    def fit(self, X, max_k=50, tolerance=3):
        min_distance = 0

        for k in range(1, len(X) + 1):
            sk = self.calculate_s_k(X, k)
            radius = math.sqrt(k * k + sk * sk)

            if k == 1:
                min_distance = radius

            if radius <= min_distance:
                min_distance = radius
                self.K = k
            elif k - self.K > tolerance:
                break