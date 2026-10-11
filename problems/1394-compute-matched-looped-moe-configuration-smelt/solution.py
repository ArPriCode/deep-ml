import math

def block_flops(d, d_ff, top_k):
    return 4 * d * d + top_k * 3 * d * d_ff

def block_params(d, d_ff, n_experts):
    return 4 * d * d + n_experts * 3 * d * d_ff

def smelt_match(d, n_blocks, n_experts, top_k, ff_mult, kv_heads, head_dim, loop_fraction=0.5):
    lambda_val = 1 + loop_fraction
    
    applications = round(n_blocks * lambda_val)
    
    d_narrowed = int(d / math.sqrt(lambda_val))
    d_new = (d_narrowed // 8) * 8
    d_ff_new = ff_mult * d_new
    
    d_ff_base = ff_mult * d
    base_block_p = block_params(d, d_ff_base, n_experts)
    base_total_p = n_blocks * base_block_p
    
    experts_new = 1
    while n_blocks * block_params(d_new, d_ff_new, experts_new) < base_total_p:
        experts_new += 1
        
    kv_heads_new = max(1, round(kv_heads / lambda_val))
    
    base_flops_total = n_blocks * block_flops(d, d_ff_base, top_k)
    looped_flops_total = applications * block_flops(d_new, d_ff_new, top_k)
    flops_ratio = round(looped_flops_total / base_flops_total, 4)
    
    looped_total_p = n_blocks * block_params(d_new, d_ff_new, experts_new)
    params_ratio = round(looped_total_p / base_total_p, 4)
    
    base_kv_elements = n_blocks * (2 * kv_heads * head_dim)
    looped_kv_elements = applications * (2 * kv_heads_new * head_dim)
    kv_ratio = round(looped_kv_elements / base_kv_elements, 4)
    
    return {
        "d_new": d_new,
        "d_ff_new": d_ff_new,
        "experts_new": experts_new,
        "kv_heads_new": kv_heads_new,
        "applications": applications,
        "flops_ratio": flops_ratio,
        "params_ratio": params_ratio,
        "kv_ratio": kv_ratio
    }