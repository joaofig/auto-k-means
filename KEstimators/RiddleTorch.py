import torch
import numpy as np


class KEstimator:

    def __init__(self):
        self.K = 0

    def calculate_determination(self, y, y_hat):
        y_np = y.numpy() if torch.is_tensor(y) else y
        y_bar = np.sum(y_np) / len(y_np)
        ssreg = np.sum((y_hat - y_bar) ** 2)
        sstot = np.sum((y_np - y_bar) ** 2)
        r2 = ssreg / sstot
        return r2

    def calculate_y_hat(self, x, y):
        xx, yy = x, y
        if torch.is_tensor(x):
            xx = x.numpy()
        if torch.is_tensor(y):
            yy = y.numpy()
        coefficients = np.polyfit(xx, yy, 2)
        polynomial = np.poly1d(coefficients)
        y_hat = polynomial(xx)
        return y_hat

    def calculate_r2(self, x, y):
        y_hat = self.calculate_y_hat(x, y)
        r2 = self.calculate_determination(y, y_hat)
        return r2

    def fit(self, s_k):
        if not torch.is_tensor(s_k):
            raise ValueError('s_k must be a torch tensor.')

        item_count = s_k.size()[0]
        k_k = torch.arange(0, item_count, dtype=torch.float32)
        r_k = torch.ones_like(s_k)
        r_k[1:] = 1.0 / s_k[1:]
        d_k = (r_k[2:] - r_k[1:-1]) / torch.log(k_k[2:])

        # First estimate for K
        k = torch.argmax(d_k) + 2

        x = torch.log(k_k[1:])
        y = torch.log(s_k[1:])

        # Calculate the r2 value for the whole curve
        r2 = self.calculate_r2(x, y)

        if 2 < k < item_count - 3:
            r2_1 = self.calculate_r2(x[:k], y[:k])
            r2_2 = self.calculate_r2(x[k+1:], y[k+1:])

            if (r2_1 + r2_2) / 2.0 > r2:
                self.K = k
            else:
                self.K = 1
        else:
            self.K = k
        return self
