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


rng = np.random.default_rng(seed=123)


def generate_matrix(d, block_size, mode, lam=1, range=None, schur=False):
    J = lam * np.eye(d) + np.diag([1] * block_size + [0] * (d - block_size - 1), k=1)
    if range is None:
        match mode:
            case "random" | "upper" | "ortho" | "lower":
                range = 1
            case "int":
                range = 100
            case _:
                raise RuntimeError(f"Mode {mode} is not supported")

    def generate_S():
        while True:
            match mode:
                case "random":
                    S = np.random.rand(d, d) * range
                case "int":
                    S = np.random.randint(0, range, size=(d, d))
                case "upper":
                    S = np.triu(np.random.rand(d, d)) * range
                case "lower":
                    S = np.tril(np.random.rand(d, d)) * range
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


def generate_testset(d, labels: LabelsManager, mode="random", schur=False):
    dataset_sizes = labels.get_dataset_sizes()
    X = np.ndarray(shape=(sum(dataset_sizes.values()), d, d))
    y = []

    idx = 0
    for label in labels.block_sizes_by_label.keys():
        for block_size in labels.block_sizes_by_label[label]:
            for _ in range(dataset_sizes[block_size]):
                X[idx] = generate_matrix(d, block_size, mode, schur)
                idx += 1
                y.append(label)

    return X, y


if __name__ == "__main__":
    m = LabelsManager([1], [2, 3, 4])
    print(m.label_by_block_size)
    print(m.block_sizes_by_label)
    print(m.get_dataset_sizes())
