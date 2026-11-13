import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    n = len(a)
    if n == 0:
        return a.copy()
    return a[(np.arange(n) + 1) % n]