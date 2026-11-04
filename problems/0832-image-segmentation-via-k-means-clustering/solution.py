import numpy as np

def segment_image(image, k: int, max_iters: int = 100):
    img_np = np.array(image, dtype=np.float64)
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, c)
    
    centroids = pixels[:k].copy()
    
    for _ in range(max_iters):
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.copy(centroids)
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                new_centroids[i] = np.mean(cluster_pixels, axis=0)
                
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
        
    segmented_pixels = centroids[labels]
    segmented_pixels = np.round(segmented_pixels, 4)
    
    return segmented_pixels.reshape(h, w, c).tolist()