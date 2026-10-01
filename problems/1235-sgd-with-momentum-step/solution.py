import torch

def momentum_step(w, grad, v, lr, mu):
    v_new = mu * v + grad
    w_new = w - lr * v_new
    return w_new, v_new