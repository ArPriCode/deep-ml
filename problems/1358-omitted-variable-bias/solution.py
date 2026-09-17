import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return coefs


def omitted_variable_bias(
    X: np.ndarray, y: np.ndarray, omit_idx: int
) -> tuple[float, float, float]:
    """Calculates full model coefficient, short model coefficient, and omitted variable bias.

    Args:
        X (np.ndarray): (n, 2) design matrix.
        y (np.ndarray): (n,) target.
        omit_idx (int): Column index to omit (0 or 1).

    Returns:
        tuple[float, float, float]: (full_kept, short, bias)
    """
    kept_idx = 1 - omit_idx

    # 1. Full regression: y ~ intercept + X_kept + X_omitted
    full_coefs = ols(X, y)
    full_kept = float(full_coefs[1 + kept_idx])
    beta_omitted = float(full_coefs[1 + omit_idx])

    # 2. Short regression: y ~ intercept + X_kept
    X_kept = X[:, [kept_idx]]
    short_coefs = ols(X_kept, y)
    short = float(short_coefs[1])

    # 3. Auxiliary regression: X_omitted ~ intercept + X_kept
    X_omitted = X[:, omit_idx]
    aux_coefs = ols(X_kept, X_omitted)
    delta = float(aux_coefs[1])

    # 4. Calculate bias: beta_omitted * delta
    bias = beta_omitted * delta

    return full_kept, short, bias