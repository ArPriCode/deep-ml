import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1
        
    pe = np.zeros((position, d_model))
    pos = np.arange(position)[:, np.newaxis]
    
    div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)
    
    pe[:, 0::2] = np.sin(pos / div_term)
    
    if d_model % 2 != 0:
        pe[:, 1::2] = np.cos(pos / div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(pos / div_term)
        
    return pe.astype(np.float16)