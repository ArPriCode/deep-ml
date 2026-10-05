import math

def rotation_layer(X, angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return [
        [x * c - y * s, x * s + y * c]
        for x, y in X
    ]