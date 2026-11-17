import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    lookahead = parameter - momentum * velocity
    grad = grad_fn(lookahead)
    new_velocity = momentum * velocity + learning_rate * grad
    new_parameter = parameter - new_velocity
    return np.round(new_parameter, 5), np.round(new_velocity, 5)