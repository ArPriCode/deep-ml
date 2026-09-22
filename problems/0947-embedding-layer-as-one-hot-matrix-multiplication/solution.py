import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.
    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)
    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    # Get numbers for shape
    vocab_size = W.shape[0]
    
    # Create the one-hot matrix H of shape (num_tokens, vocab_size)
    H = np.eye(vocab_size)[token_ids]
    
    # Perform matrix multiplication H @ W
    return H @ W