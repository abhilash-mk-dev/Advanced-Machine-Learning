import numpy as np

def gen_linear_data(n=100):
    X = np.random.randn(n,2)
    y = (X[:,0] + X[:,1]>0).astype(int)
    # print(X,y)
    return X, y

def gen_non_linear_data(n=100):
    X = np.random.randn(n,2)
    y = ((X[:,0] > 0) ^ (X[:,1] > 0)).astype(int)
    # print(X,y)
    return X, y