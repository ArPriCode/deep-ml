import numpy as np

def compute_log_prob_ratios(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    mask: np.ndarray
) -> dict:
    per_token_log_ratios = (policy_log_probs - reference_log_probs) * mask
    sequence_log_ratios = np.sum(per_token_log_ratios, axis=1)
    mean_sequence_log_ratio = float(np.mean(sequence_log_ratios))
    
    return {
        'per_token_log_ratios': per_token_log_ratios,
        'sequence_log_ratios': sequence_log_ratios,
        'mean_sequence_log_ratio': mean_sequence_log_ratio
    }