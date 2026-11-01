# --- cell 1 ---
import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    '''
    # ReLU Activation Function: max(0, x)
    return np.maximum(0, x)