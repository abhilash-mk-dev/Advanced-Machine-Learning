import numpy as np
from sklearn.datasets import make_blobs


def gen_linear_data(n=100, std=1.0):
    X, y = make_blobs(n_samples = n, centers=2, random_state=42, cluster_std=std)
    y = np.where(y==0,-1,1)
    return X, y

