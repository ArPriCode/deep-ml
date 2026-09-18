import numpy as np

def superposition_reconstruct(W, b, X):
    """
    Compute reconstructed features for the toy superposition model.
    
    Args:
        W: array of shape (n_hidden, n_features)
        b: array of shape (n_features,)
        X: array of shape (batch_size, n_features)
        
    Returns:
        list of lists of shape (batch_size, n_features) with reconstructed features
    """
    W = np.array(W)
    b = np.array(b)
    X = np.array(X)
    
    # x_hat = ReLU(X @ W^T @ W + b)
    # Using matrix multiplication across the batch:
    # (batch_size, n_features) @ (n_features, n_hidden) @ (n_hidden, n_features)
    reconstruction = np.maximum(0, X @ W.T @ W + b)
    
    return reconstruction.tolist()