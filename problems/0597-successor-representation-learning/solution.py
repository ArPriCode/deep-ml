import numpy as np

def learn_successor_representation(
    experience: list,
    n_states: int,
    gamma: float,
    alpha_sr: float,
    alpha_w: float
) -> tuple:
    M = np.zeros((n_states, n_states), dtype=float)
    w = np.zeros(n_states, dtype=float)

    for s, r, next_state, done in experience:
        w[s] += alpha_w * (r - w[s])

        target = np.zeros(n_states)
        target[s] = 1.0

        if not done:
            target += gamma * M[next_state]

        M[s] += alpha_sr * (target - M[s])

    V = M @ w

    M = np.round(M, 4).tolist()
    w = np.round(w, 4).tolist()
    V = np.round(V, 4).tolist()

    return M, w, V