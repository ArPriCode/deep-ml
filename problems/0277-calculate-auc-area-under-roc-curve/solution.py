import numpy as np


def calculate_auc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_true_sorted == 0)[threshold_indices]

    tpr = np.r_[0, tps / n_pos]
    fpr = np.r_[0, fps / n_neg]

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    else:
        return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))