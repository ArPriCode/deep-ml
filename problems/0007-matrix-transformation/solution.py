import numpy as np

def transform_matrix(A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]) -> list[list[int | float]]:
    A_np = np.array(A, dtype=float)
    T_np = np.array(T, dtype=float)
    S_np = np.array(S, dtype=float)
    
    # Check if T and S are invertible by checking their determinants
    if np.linalg.det(T_np) == 0 or np.linalg.det(S_np) == 0:
        return -1
    
    # Compute T_inv * A * S
    T_inv = np.linalg.inv(T_np)
    transformed_matrix = np.dot(np.dot(T_inv, A_np), S_np)
    
    return transformed_matrix.tolist()