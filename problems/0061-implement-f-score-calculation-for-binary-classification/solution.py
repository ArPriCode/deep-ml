import numpy as np


def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = beta**2
    denom = (beta_sq * precision) + recall

    if denom == 0:
        return 0.0

    score = (1 + beta_sq) * (precision * recall) / denom
    return round(float(score), 3)