import math

def pooled_stats(m1, s1, n1, m2, s2, n2):
    combined_mean = (n1 * m1 + n2 * m2) / (n1 + n2)
    
    ss1 = (n1 - 1) * (s1 ** 2)
    ss2 = (n2 - 1) * (s2 ** 2)
    
    ss_mean = n1 * ((m1 - combined_mean) ** 2) + n2 * ((m2 - combined_mean) ** 2)
    
    tss = ss1 + ss2 + ss_mean
    
    combined_sd = math.sqrt(tss / (n1 + n2 - 1))
    
    return combined_mean, combined_sd