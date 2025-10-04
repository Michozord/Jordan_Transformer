import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng(seed=123)

def generate_matrix(d, block_size, mode, lam = 0, range = None):
    J = lam * np.eye(d) + np.diag([1]*block_size + [0]*(d-block_size-1), k=1)
    if range is None:
        match mode:
            case "random"|"upper": range = 1
            case "int": range = 100
            case _: raise RuntimeError(f"Mode {mode} is not supported")

    def generate_S():
        while True:
            match mode:
                case "random": S = np.random.rand(d,d) * range
                case "int":    S = np.random.randint(0, range, size=(d,d))
                case "upper":  S = np.triu(np.random.rand(d,d)) * range
            if abs(np.linalg.det(S)) > 1e-6:
                return S
    
    S = generate_S()
    X = S @ J @ np.linalg.inv(S)
    return X

def generate_testset(d, dataset_size, mode="random"):
    X = np.ndarray(shape=(dataset_size * d, d, d))
    y = []

    for block_size in range(d):
        for i in range(dataset_size):
            X[i + block_size * dataset_size] = generate_matrix(d, block_size, mode)
        y += [block_size] * dataset_size
    
    return X, y