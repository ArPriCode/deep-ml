import numpy as np


def flip(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    return a[n - 1 - np.arange(n)]