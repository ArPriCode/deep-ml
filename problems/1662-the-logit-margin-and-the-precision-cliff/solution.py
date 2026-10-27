import numpy as np

def logit_margin(z, t):
    z = np.asarray(z, dtype=np.float64)
    other_z = np.delete(z, t)
    max_z = np.max(other_z)
    lse = max_z + np.log(np.sum(np.exp(other_z - max_z)))
    return float(z[t] - lse)

def precision_cliff(dtype):
    def test_m(m):
        exp_neg_m = dtype(np.exp(-m, dtype=np.float64))
        denom = dtype(dtype(1) + exp_neg_m)
        val = dtype(dtype(1) / denom)
        return val == dtype(1)

    low = 0.0
    high = 100.0
    while (high - low) > 1e-9:
        mid = (low + high) / 2.0
        if test_m(mid):
            high = mid
        else:
            low = mid
            
    return round(float(high), 4)