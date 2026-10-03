import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    s = np.linalg.svd(delta_W, compute_uv=False)

    energy = s ** 2
    total = energy.sum()

    if total == 0:
        return 0

    cumulative = np.cumsum(energy)
    k = np.searchsorted(cumulative, energy_threshold * total) + 1

    return int(k)