from tinygrad import Tensor

def dropout(x, p, training):
    if not training or p == 0:
        return x
    
    mask = (Tensor.rand(*x.shape) >= p).cast(x.dtype)
    return x * mask * (1.0 / (1.0 - p))