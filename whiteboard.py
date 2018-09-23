import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from KEstimators import PhamDimovNguyen
from KEstimators import AsankaPerera


def load_data_1024():
    df = pd.read_csv('data/data_1024.csv',
                     delim_whitespace=True)
    df = df.drop('Driver_ID', axis=1)
    df = df.rename(index=str, columns={"Distance_Feature": "x", "Speeding_Feature": "y"})
    df = df[['x','y']]
    df = (df - df.min()) / (df.max() - df.min())
    return df


def load_a1():
    df = pd.read_csv('data/a1.txt', names=['x', 'y'], delim_whitespace=True, dtype=np.float64)
    df = (df - df.min()) / (df.max() - df.min())
    return df


def load_unbalance():
    df = pd.read_csv('data/unbalance.txt', names=['x', 'y'], delim_whitespace=True, dtype=np.float64)
    df = (df - df.min()) / (df.max() - df.min())
    return df


def load_dim2():
    df = pd.read_csv('data/dim2.txt', names=['x', 'y'], delim_whitespace=True, dtype=np.float64)
    df = (df - df.min()) / (df.max() - df.min())
    return df


def load_HTRU_2():
    df = pd.read_csv('data/HTRU2/HTRU_2.csv', dtype=np.float64)
    df = (df - df.min()) / (df.max() - df.min())
    return df


def load_data():
    return load_dim2()



def run():
    pdn_estimator = PhamDimovNguyen.KEstimator()
    asanka_estimator = AsankaPerera.KEstimator()

    df = load_data()

    asanka_estimator.fit(df, tolerance=1e-4)
    print('Asanka : {0}'.format(asanka_estimator.K))

    pdn_estimator.fit(df, max_k=30)
    print('PhamDN : {0}'.format(pdn_estimator.K))

    # f_k = calculate_f_k(df, max_k=30)
    fig, ax1 = plt.subplots()
    # ax1.plot(range(1, len(f_k) + 1), f_k)
    ax1.scatter(df.x, df.y)
    plt.show()


if __name__ == '__main__':
    run()
