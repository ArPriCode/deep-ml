import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    X = np.array(X, dtype=float)
    W_q_lin = np.array(W_q_lin, dtype=float)
    W_k_lin = np.array(W_k_lin, dtype=float)
    W_v_lin = np.array(W_v_lin, dtype=float)
    
    W_q = W_q_lin.T
    W_k = W_k_lin.T
    W_v = W_v_lin.T
    
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    d_out = W_q_lin.shape[0]
    scores = (Q @ K.T) / np.sqrt(d_out)
    
    seq_len = X.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores[mask] = -1e9
    
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = attn_weights @ V
    return output.tolist()