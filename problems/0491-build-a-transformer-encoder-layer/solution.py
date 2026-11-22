import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def layer_norm(x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    Q = X @ weights['W_q']
    K = X @ weights['W_k']
    V = X @ weights['W_v']

    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = softmax(scores)
    attn_output = attn_weights @ V

    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    mha_output = attn_output @ weights['W_o']

    norm1 = layer_norm(X + mha_output, weights['gamma1'], weights['beta1'])

    ffn_hidden = np.maximum(0, norm1 @ weights['W1'] + weights['b1'])
    ffn_output = ffn_hidden @ weights['W2'] + weights['b2']

    output = layer_norm(norm1 + ffn_output, weights['gamma2'], weights['beta2'])

    return output