import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    adjusted_alpha = base_alpha / n_tilings
    prediction_before = sum(weights[i] for i in active_tiles)
    
    updated_weights = list(weights)
    for idx in active_tiles:
        updated_weights[idx] += adjusted_alpha * (target - prediction_before)
        
    prediction_after = sum(updated_weights[i] for i in active_tiles)
    
    return {
        'adjusted_alpha': round(adjusted_alpha, 4),
        'prediction_before': round(prediction_before, 4),
        'prediction_after': round(prediction_after, 4),
        'updated_weights': [round(w, 4) for w in updated_weights]
    }