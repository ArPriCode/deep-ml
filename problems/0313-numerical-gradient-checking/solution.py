import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    """
    x = np.array(x, dtype=np.float64)
    analytical_grad = np.array(analytical_grad, dtype=np.float64)
    
    numerical_grad = np.zeros_like(x)
    
    # Iterate over each element/dimension of the input vector/tensor
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        # Evaluate f(x + epsilon)
        x[idx] = old_val + epsilon
        fx_plus = f(x)
        
        # Evaluate f(x - epsilon)
        x[idx] = old_val - epsilon
        fx_minus = f(x)
        
        # Restore original value
        x[idx] = old_val
        
        # Centered finite difference formula
        numerical_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        it.iternext()
        
    # Calculate relative error: ||num_grad - analytical_grad|| / (||num_grad|| + ||analytical_grad||)
    diff_norm = np.linalg.norm(numerical_grad - analytical_grad)
    num_norm = np.linalg.norm(numerical_grad)
    analytical_norm = np.linalg.norm(analytical_grad)
    
    denominator = num_norm + analytical_norm
    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = diff_norm / denominator
        
    return numerical_grad, float(relative_error)