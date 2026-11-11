def attention_memory(T: int, H: int, W: int, num_heads: int, bytes_per_element: int) -> list:
    N = T * H * W
    S = H * W
    full_bytes = num_heads * N * N * bytes_per_element
    factorized_bytes = num_heads * (T * S * S + S * T * T) * bytes_per_element
    return [full_bytes, factorized_bytes]