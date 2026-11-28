import numpy as np


def qb_route(scores, beta, k):
    scores = np.asarray(scores, dtype=float)
    beta = np.asarray(beta, dtype=float)
    T, N = scores.shape
    if (T * k) % N != 0:
        raise ValueError("T * k must be divisible by N")
    C = (T * k) // N

    biased = scores - beta
    order = np.argsort(-biased, axis=1, kind="stable")
    idx = order[:, :k]

    sorted_biased = np.take_along_axis(biased, order, axis=1)
    a_t = sorted_biased[:, k]

    margins = scores - a_t[:, None]
    sorted_margins = np.sort(margins, axis=0)[::-1]
    beta_local = sorted_margins[C, :]

    return idx, beta_local


def qb_train(steps, k, ema=0.0):
    N = np.asarray(steps[0][0]).shape[1]
    beta = np.zeros(N, dtype=float)
    global_maxvio = []
    micro_maxvio = []

    for global_batch in steps:
        global_counts = np.zeros(N, dtype=float)
        micro_vios = []
        beta_locals = []

        for micro_batch in global_batch:
            scores = np.asarray(micro_batch, dtype=float)
            idx, beta_local = qb_route(scores, beta, k)
            beta_locals.append(beta_local)

            counts = np.bincount(idx.ravel(), minlength=N).astype(float)
            global_counts += counts

            c_bar_micro = counts.mean()
            micro_vios.append(np.max(counts / c_bar_micro - 1.0))

        c_bar_global = global_counts.mean()
        g_vio = float(np.max(global_counts / c_bar_global - 1.0))
        m_vio = float(np.mean(micro_vios))

        global_maxvio.append(round(g_vio, 4) + 0.0)
        micro_maxvio.append(round(m_vio, 4) + 0.0)

        beta_bar = np.mean(beta_locals, axis=0)
        beta = ema * beta + (1.0 - ema) * beta_bar
        beta = beta - np.mean(beta)

    return {
        "global_maxvio": global_maxvio,
        "micro_maxvio": micro_maxvio,
        "beta": (np.round(beta, 4) + 0.0).tolist(),
    }