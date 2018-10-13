import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

from sklearn.datasets.samples_generator import make_blobs

from KEstimators import PhamDimovNguyen
from KEstimators import Riddle
from KEstimators import RiddleTorch

#
#
# def load_data_1024():
#     df = pd.read_csv('data/data_1024.csv',
#                      delim_whitespace=True)
#     df = df.drop('Driver_ID', axis=1)
#     df = df.rename(index=str, columns={"Distance_Feature": "x", "Speeding_Feature": "y"})
#     df = df[['x','y']]
#     df = (df - df.min()) / (df.max() - df.min())
#     return df
#
#
# def load_a1():
#     df = pd.read_csv('data/a1.txt', names=['x', 'y'], delim_whitespace=True, dtype=np.float64)
#     df = (df - df.min()) / (df.max() - df.min())
#     return df
#
#
# def load_unbalance():
#     df = pd.read_csv('data/unbalance.txt', names=['x', 'y'], delim_whitespace=True, dtype=np.float64)
#     df = (df - df.min()) / (df.max() - df.min())
#     return df
#
#
# def load_dim2():
#     df = pd.read_csv('data/dim2.txt', names=['x', 'y'], delim_whitespace=True, dtype=np.float64)
#     df = (df - df.min()) / (df.max() - df.min())
#     return df
#
#
# def load_HTRU_2():
#     df = pd.read_csv('data/HTRU2/HTRU_2.csv', dtype=np.float64)
#     df = (df - df.min()) / (df.max() - df.min())
#     return df


def load_data():
    clusters = random.randint(1, 20)
    cluster_std = random.uniform(0.5, 8)
    print('K={0}'.format(clusters))
    X, y = make_blobs(n_samples=200*clusters,
                      centers=clusters,
                      cluster_std=cluster_std,
                      n_features=2,
                      center_box=(-50.0, 50.0))
    df = pd.DataFrame(data=X, columns=['x', 'y'])
    return df


# from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


def calculate_s_k(X, k):
    km = KMeans(n_clusters=k).fit(X)
    return km.inertia_


def run():
    pham_estimator = PhamDimovNguyen.KEstimator()
    # asanka_estimator = AsankaPerera.KEstimator()
    riddle_estimator = Riddle.KEstimator()
    torch_estimator = RiddleTorch.KEstimator()

    s_k_t = torch.zeros(51)
    df = load_data()
    s_k = dict()
    dim = len(df.columns)

    print(len(df))

    x_range = range(1, 51)

    for k in x_range:
        km = KMeans(n_clusters=k).fit(df)
        s_k[k] = km.inertia_
        s_k_t[k] = km.inertia_
        print(s_k[k])

    # asanka_estimator.fit_s_k(s_k, tolerance=1e-3)
    # print('Asanka : {0}'.format(asanka_estimator.K))

    pham_estimator.fit_s_k(s_k, max_k=50, dim=dim)
    print('PhamDN : {0}'.format(pham_estimator.K))

    riddle_estimator.fit_s_k(s_k, max_k=50)
    print('Riddle : {0}'.format(riddle_estimator.K))

    torch_estimator.fit(s_k_t)
    print('Torch  : {0}'.format(torch_estimator.K))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(df.x, df.y)
    ax1.set_title('Clusters')
    ax2.plot(np.log(np.arange(1, 51, dtype=np.float32)), np.log(s_k_t.numpy()[1:]))
    ax2.set_title('log-log Scree')
    plt.show()


if __name__ == '__main__':
    run()
