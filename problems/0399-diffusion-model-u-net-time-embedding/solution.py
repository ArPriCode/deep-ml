import numpy as np

def unet_time_embedding(timesteps: list, embed_dim: int, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, max_period: int = 10000) -> np.ndarray:
    t = np.asarray(timesteps, dtype=float)

    half_dim = embed_dim // 2
    i = np.arange(half_dim)

    freqs = np.exp(-np.log(max_period) * i / half_dim)
    args = t[:, None] * freqs[None, :]

    emb = np.concatenate([np.sin(args), np.cos(args)], axis=1)

    h = emb @ W1 + b1
    h = h * (1 / (1 + np.exp(-h)))

    return h @ W2 + b2