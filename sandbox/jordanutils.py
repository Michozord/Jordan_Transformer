import numpy as np
import scipy
from math import inf


class LabelsManager:

    def __init__(self, *block_sizes_list, dataset_size=100_000):
        self.dataset_size = dataset_size
        block_sizes_by_label = {}
        label_by_block_size = {}
        for label, block_sizes in enumerate(block_sizes_list):
            block_sizes_by_label[label] = (
                block_sizes_by_label.get(label, []) + block_sizes
            )
            for block_size in block_sizes:
                label_by_block_size[block_size] = label
        self.block_sizes_by_label = block_sizes_by_label
        self.label_by_block_size = label_by_block_size

    def get_dataset_sizes(self):
        sizes = {}
        for block_sizes in self.block_sizes_by_label.values():
            num = int(self.dataset_size / len(block_sizes))
            for block_size in block_sizes:
                sizes[block_size] = num
        return sizes


def generate_matrix(d, block_size, mode, eps=None, lam=1, value_range=None, schur=False):
    indexes = np.random.choice(d-1, size=block_size, replace=False)
    # indexes = list(range(block_size))
    super_diag = np.zeros(d-1)
    for index in indexes:
        super_diag[index] = 1
    J = lam * np.eye(d) + np.diag(super_diag, k=1)
    if eps is not None:
        J += eps * np.random.rand(d, d)
    if value_range is None:
        match mode:
            case "random" | "upper" | "ortho" | "lower":
                value_range = 1
            case "int":
                value_range = 100
            case _:
                raise RuntimeError(f"Mode {mode} is not supported")

    def generate_S():
        while True:
            match mode:
                case "random":
                    S = np.random.rand(d, d) * value_range
                case "int":
                    S = np.random.randint(0, value_range, size=(d, d))
                case "upper":
                    S = np.triu(np.random.rand(d, d)) * value_range
                case "lower":
                    S = np.tril(np.random.rand(d, d)) * value_range
                case "ortho":
                    A = np.random.rand(d, d)
                    Q, _ = np.linalg.qr(A)
                    S = Q
            if abs(np.linalg.cond(S)) < 1e5:
                return S

    S = generate_S()
    X = S @ J @ np.linalg.inv(S)
    # X = X / np.linalg.norm(X, ord="fro")
    if schur:
        return scipy.linalg.schur(X)[0]
    else:
        return X


def generate_testset(d, size_per_class, mode="random", eps=None, schur=False):
    X = np.ndarray(shape=(size_per_class * d, d, d))
    y = []

    idx = 0
    for label in range(d):
        for _ in range(size_per_class):
            X[idx] = generate_matrix(d, label, mode, eps=eps, schur=schur)
            idx += 1
            y.append(label)

    return X, y


if __name__ == "__main__":
    m = LabelsManager([1], [2, 3, 4])
    print(m.label_by_block_size)
    print(m.block_sizes_by_label)
    print(m.get_dataset_sizes())
