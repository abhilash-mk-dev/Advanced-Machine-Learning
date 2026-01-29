import numpy as np
from sklearn.datasets import load_breast_cancer

def gen_cancer_data(n=100):
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def gen_large_linear_data(n=200000):
    X = np.random.randn(n,20)
    y = ((X[:,0] > 0) ^ (X[:,1] > 0)).astype(int)
    return X, y