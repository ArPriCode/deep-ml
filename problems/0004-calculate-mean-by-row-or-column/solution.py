import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    matrix_np = np.array(matrix)
    
    if mode == 'row':
        return np.mean(matrix_np, axis=1).tolist()
    elif mode == 'column':
        return np.mean(matrix_np, axis=0).tolist()
    else:
        raise ValueError("Mode must be either 'row' or 'column'")
		