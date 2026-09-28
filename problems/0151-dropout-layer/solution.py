import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not 0 <= self.p < 1:
            raise ValueError("p must be between 0 and 1")

        if not training:
            return x

        self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)

        return x * self.mask / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")

        return grad * self.mask / (1 - self.p)