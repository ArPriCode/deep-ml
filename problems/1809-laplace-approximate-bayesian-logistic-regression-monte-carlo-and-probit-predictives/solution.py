import numpy as np


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def laplace_logreg(X, y, prior_var, n_iter=50):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    _, D = X.shape
    w = np.zeros(D, dtype=float)
    inv_prior = np.eye(D, dtype=float) / prior_var

    for _ in range(n_iter):
        mu = _sigmoid(X @ w)
        S = mu * (1.0 - mu)
        grad = X.T @ (y - mu) - w / prior_var
        H = X.T @ (S[:, None] * X) + inv_prior
        step = np.linalg.solve(H, grad)
        w = w + step
        if np.linalg.norm(step) < 1e-12:
            break

    mu = _sigmoid(X @ w)
    S = mu * (1.0 - mu)
    H = X.T @ (S[:, None] * X) + inv_prior
    Sigma = np.linalg.inv(H)
    return w, Sigma


def predictive(X_test, w_map, Sigma, method, n_samples=1000, seed=0):
    X_test = np.asarray(X_test, dtype=float)
    w_map = np.asarray(w_map, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if method == "plugin":
        return _sigmoid(X_test @ w_map)

    if method == "mc":
        D = w_map.shape[0]
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, D))
        L = np.linalg.cholesky(Sigma)
        w_samples = w_map + z @ L.T
        probs = _sigmoid(X_test @ w_samples.T)
        return np.mean(probs, axis=1)

    if method == "probit":
        m = X_test @ w_map
        v = np.sum((X_test @ Sigma) * X_test, axis=1)
        kappa = (1.0 + np.pi * v / 8.0) ** -0.5
        return _sigmoid(kappa * m)

    raise ValueError(f"Unknown method: {method}")