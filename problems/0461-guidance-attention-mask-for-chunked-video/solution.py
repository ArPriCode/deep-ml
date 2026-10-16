import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int], 
    current_chunk: int
) -> np.ndarray:
    total_tokens = sum(chunk_sizes)
    mask = np.zeros((total_tokens, total_tokens), dtype=bool)
    
    history_tokens = sum(chunk_sizes[:current_chunk])
    curr_chunk_size = chunk_sizes[current_chunk]
    curr_chunk_end = history_tokens + curr_chunk_size
    
    if history_tokens > 0:
        mask[:history_tokens, :history_tokens] = True
        mask[history_tokens:curr_chunk_end, :history_tokens] = True
        
    causal_mask = np.tril(np.ones((curr_chunk_size, curr_chunk_size), dtype=bool))
    mask[history_tokens:curr_chunk_end, history_tokens:curr_chunk_end] = causal_mask
    
    return mask