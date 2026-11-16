import numpy as np
from itertools import product

def polynomial_basis(s, n):
    s = np.asarray(s, dtype=float)
    k = len(s)
    exponents = np.array(list(product(range(n + 1), repeat=k)))
    return np.prod(s ** exponents, axis=1)

def gradient_mc_poly(samples, n, alpha, n_passes):
    if not samples:
        return np.array([], dtype=float)
    k = len(samples[0][0])
    w = np.zeros((n + 1) ** k, dtype=float)
    for _ in range(n_passes):
        for s, G in samples:
            x = polynomial_basis(s, n)
            v_hat = np.dot(w, x)
            w += alpha * (G - v_hat) * x
    return w