import numpy as np
from scipy.optimize import linprog

def lad_fit(X, y):
    N, D = X.shape
    c = np.concatenate([np.zeros(D), np.ones(N)])
    
    A_ub = np.vstack([
        np.hstack([X, -np.eye(N)]),
        np.hstack([-X, -np.eye(N)])
    ])
    b_ub = np.concatenate([y, -y])
    
    bounds = [(None, None)] * D + [(0, None)] * N
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:D]

def huber_fit(X, y, delta, n_iter):
    N, D = X.shape
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    
    for _ in range(n_iter):
        r = y - X @ w
        abs_r = np.abs(r)
        
        s = np.ones(N)
        mask = abs_r > 0
        s[mask] = np.minimum(1.0, delta / abs_r[mask])
        
        W = np.diag(s)
        w = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
        
    return w