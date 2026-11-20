import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta):
    policy_chosen_logps = np.asarray(policy_chosen_logps, dtype=np.float64)
    policy_rejected_logps = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_chosen_logps = np.asarray(ref_chosen_logps, dtype=np.float64)
    ref_rejected_logps = np.asarray(ref_rejected_logps, dtype=np.float64)
    
    h = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
    target = 1.0 / (2.0 * beta)
    loss = np.mean((h - target) ** 2)
    
    return round(float(loss), 6)