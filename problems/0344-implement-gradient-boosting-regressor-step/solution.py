import numpy as np


def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    X = np.array(X)
    y = np.array(y)
    current_predictions = np.array(current_predictions)

    residuals = y - current_predictions
    n_samples, n_features = X.shape

    best_mse = float("inf")
    best_stump_preds = np.full(n_samples, np.mean(residuals))

    for j in range(n_features):
        feature_vals = X[:, j]
        unique_vals = np.sort(np.unique(feature_vals))

        if len(unique_vals) <= 1:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = feature_vals <= threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            left_mean = np.mean(residuals[left_mask])
            right_mean = np.mean(residuals[right_mask])

            stump_preds = np.zeros(n_samples)
            stump_preds[left_mask] = left_mean
            stump_preds[right_mask] = right_mean

            mse = np.mean((residuals - stump_preds) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_stump_preds = stump_preds

    updated_predictions = current_predictions + learning_rate * best_stump_preds
    return np.round(updated_predictions, 4).tolist()