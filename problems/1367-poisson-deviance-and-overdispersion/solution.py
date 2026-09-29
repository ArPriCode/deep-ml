import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    
    safe_y = np.where(y > 0, y, 1)

    terms = np.where(
        y > 0,
        y * np.log(safe_y / mu) - (y - mu),
        mu
    )

    return float(2 * np.sum(terms))


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    
    chi_square = np.sum((y - mu) ** 2 / mu)
    df = len(y) - n_params

    return float(chi_square / df)