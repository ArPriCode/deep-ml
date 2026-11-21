import numpy as np


def _stable_bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Element-wise numerically stable binary cross-entropy with logits."""
    # max(x, 0) - x * z + log(1 + exp(-|x|))
    return np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


def gae_encode(A, X, W1, W2):
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)

    N = A.shape[0]
    A_tilde = A + np.eye(N, dtype=float)
    deg = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    # Normalised adjacency: D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)
    A_hat = (A_tilde * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

    H = np.maximum(0.0, A_hat @ X @ W1)
    Z = A_hat @ H @ W2
    return Z


def gae_logits(Z, pairs):
    Z = np.asarray(Z, dtype=float)
    if len(pairs) == 0:
        return np.array([], dtype=float)
    idx = np.asarray(pairs, dtype=int)
    return np.sum(Z[idx[:, 0]] * Z[idx[:, 1]], axis=1)


def gae_loss(A, Z, pos_weight=None):
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    N = A.shape[0]

    if pos_weight is None:
        pos_sum = np.sum(A)
        pos_weight = (N * N - pos_sum) / pos_sum

    logits = Z @ Z.T
    # Weighted BCE: pos_weight * y * (-log(sigmoid(x))) + (1 - y) * (-log(1 - sigmoid(x)))
    # Using stable identity:
    # -log(sigmoid(x)) = max(x, 0) - x + log(1 + exp(-|x|))
    # -log(1 - sigmoid(x)) = max(x, 0) + log(1 + exp(-|x|))
    log1p_term = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    loss_matrix = pos_weight * A * (log1p_term - logits) + (1.0 - A) * log1p_term

    return float(np.mean(loss_matrix))


def sampled_gae_loss(Z, pos_pairs, neg_pairs):
    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)

    logits = np.concatenate([pos_logits, neg_logits])
    targets = np.concatenate([
        np.ones(len(pos_logits), dtype=float),
        np.zeros(len(neg_logits), dtype=float),
    ])

    losses = _stable_bce_with_logits(logits, targets)
    return float(np.mean(losses))


def vgae_sample(mu, logvar, eps):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    eps = np.asarray(eps, dtype=float)
    return mu + np.exp(0.5 * logvar) * eps


def vgae_kl(mu, logvar):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    # KL(N(mu, diag(exp(logvar))) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_node = -0.5 * np.sum(1.0 + logvar - np.square(mu) - np.exp(logvar), axis=-1)
    return float(np.mean(kl_per_node))