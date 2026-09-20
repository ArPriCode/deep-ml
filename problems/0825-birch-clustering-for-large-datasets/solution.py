import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.
    X: array-like of shape (n_samples, n_features)
    threshold: float, max allowed subcluster radius
    Returns: list of centroids (each a list of floats), sorted lexicographically.
    """
    X = np.array(X, dtype=float)
    if X.size == 0:
        return []

    # Store subclusters as dictionaries containing CF tuple components: N, LS, SS
    subclusters = []

    def compute_radius(N, LS, SS):
        # Calculate per-dimension variance: (SS_d / N) - (LS_d / N)^2
        variance = (SS / N) - (LS / N) ** 2
        # Clamp tiny negative variances due to floating-point imprecision to 0
        variance = np.maximum(variance, 0.0)
        # Radius is the square root of the sum of variances across dimensions
        return np.sqrt(np.sum(variance))

    for x in X:
        if not subclusters:
            # Step 1: First point creates the first subcluster
            subclusters.append({
                'N': 1,
                'LS': x.copy(),
                'SS': x ** 2
            })
            continue

        # Step 2: Find the existing subcluster whose centroid is closest to x
        best_idx = -1
        min_dist = float('inf')

        for i, sc in enumerate(subclusters):
            centroid = sc['LS'] / sc['N']
            dist = np.linalg.norm(x - centroid)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # Step 3: Tentatively absorb x into the closest subcluster
        target_sc = subclusters[best_idx]
        tentative_N = target_sc['N'] + 1
        tentative_LS = target_sc['LS'] + x
        tentative_SS = target_sc['SS'] + (x ** 2)

        tentative_radius = compute_radius(tentative_N, tentative_LS, tentative_SS)

        if tentative_radius <= threshold:
            # Keep the absorption
            target_sc['N'] = tentative_N
            target_sc['LS'] = tentative_LS
            target_sc['SS'] = tentative_SS
        else:
            # Create a new subcluster from x
            subclusters.append({
                'N': 1,
                'LS': x.copy(),
                'SS': x ** 2
            })

    # Extract final centroids
    centroids = [(sc['LS'] / sc['N']).tolist() for sc in subclusters]

    # Sort centroids in lexicographical order
    centroids.sort()

    return centroids