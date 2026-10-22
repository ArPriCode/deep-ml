import numpy as np
from scipy import stats

def two_sample_t_test(sample1: list[float], sample2: list[float], alpha: float = 0.05) -> dict:
    s1 = np.array(sample1, dtype=float)
    s2 = np.array(sample2, dtype=float)
    
    n1, n2 = len(s1), len(s2)
    m1, m2 = np.mean(s1), np.mean(s2)
    v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    
    se = np.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se
    
    df = (v1 / n1 + v2 / n2)**2 / ((v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    reject_null = bool(p_val < alpha)
    
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / s_pooled
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'degrees_of_freedom': float(df),
        'reject_null': reject_null,
        'cohens_d': float(cohens_d)
    }