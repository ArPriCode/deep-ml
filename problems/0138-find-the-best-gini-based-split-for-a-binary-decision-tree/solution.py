import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises
    weighted Gini impurity."""
    n_samples, n_features = X.shape
    best_gini = float('inf')
    best_feature = -1
    best_threshold = None

    def calculate_gini(labels):
        if len(labels) == 0:
            return 0.0
        p1 = np.mean(labels == 1)
        p0 = 1 - p1
        return 1.0 - (p0**2 + p1**2)

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            n_left = len(y_left)
            n_right = len(y_right)

            gini_left = calculate_gini(y_left)
            gini_right = calculate_gini(y_right)

            weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

            if weighted_gini < best_gini - 1e-9:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = float(threshold)

    return best_feature, best_threshold