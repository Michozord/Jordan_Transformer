import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

rng = np.random.default_rng(seed=123)

def generate_matrix(d, block_size, lam = 0):
    J = lam * np.eye(d) + np.diag([1]*block_size + [0]*(d-block_size-1), k=1)
    S = np.random.rand(d,d)
    X = S @ J @ np.linalg.inv(S)
    return X

def generate_testset(d, dataset_size):
    X = np.ndarray(shape=(dataset_size * d, d, d))
    y = []

    for block_size in range(d):
        for i in range(dataset_size):
            X[i + block_size * dataset_size] = generate_matrix(d, block_size)
        y += [block_size] * dataset_size
    
    return X, y