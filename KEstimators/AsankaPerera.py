import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

class KEstimator:
    """Estimates the K-Means K hyperparameter through geometrical analysis of the distortion curve"""

    def __init__(self):
        self.K = 0

    def calculate_s_k(self, X, k):
        km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
        return km.inertia_  # -km.score(df) #

    @staticmethod
    def distance_to_line(x0, y0, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx * dx + dy * dy)

    def fit(self, X, tolerance=1e-3):
        """Fits the value of K"""
        max_distance = -1
        s_k_list = list()
        sk0 = 0

        for k in range(1, len(X) + 1):
            sk1 = self.calculate_s_k(X, k)
            s_k_list.append(sk1)
            if k > 2 and abs(sk0 - sk1) < tolerance:
                break
            sk0 = sk1

        s_k = np.array(s_k_list)
        x0 = 1
        y0 = s_k[0]

        x1 = len(s_k)
        y1 = 0

        for k in range(1, len(s_k)):
            dist = self.distance_to_line(k, s_k[k-1], x0, y0, x1, y1)
            if dist > max_distance:
                max_distance = dist
            else:
                self.K = k - 1
                break
