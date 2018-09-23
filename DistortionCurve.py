class DistortionCurve:
    """Stores the distortion curve as a function of K, the number of clusters."""

    def __init__(self):
        self.distortions = dict()

    def add(self, k, distortion):
        self.distortions[k] = distortion

    def get(self, k):
        return self.distortions[k]

    def __getitem__(self, item):
        return self.distortions[item]

    def __setitem__(self, key, value):
        self.distortions[key] = value
