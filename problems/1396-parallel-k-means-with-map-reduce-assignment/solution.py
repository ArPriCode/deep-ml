import numpy as np

def kmeans_partial(X_chunk, centroids):
    k = centroids.shape[0]
    d = X_chunk.shape[1]
    
    if X_chunk.shape[0] == 0:
        return np.zeros((k, d), dtype=X_chunk.dtype), np.zeros(k, dtype=np.int64)
    
    distances = ((X_chunk[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    labels = np.argmin(distances, axis=-1)
    
    sums = np.zeros((k, d), dtype=X_chunk.dtype)
    np.add.at(sums, labels, X_chunk)
    
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    
    return sums, counts

def parallel_kmeans(X, centroids, n_chunks, n_iters):
    centroids = np.asarray(centroids, dtype=float).copy()
    k = centroids.shape[0]
    
    for _ in range(n_iters):
        chunks = np.array_split(X, n_chunks)
        
        total_sums = np.zeros_like(centroids)
        total_counts = np.zeros(k, dtype=np.int64)
        
        for chunk in chunks:
            s, c = kmeans_partial(chunk, centroids)
            total_sums += s
            total_counts += c
        
        non_empty = total_counts > 0
        centroids[non_empty] = total_sums[non_empty] / total_counts[non_empty][:, None]
        
    return centroids