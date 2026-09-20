import numpy as np

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # Calculate eigenvalues using NumPy
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Sort eigenvalues from highest to lowest
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    
    # Return as a list of float values
    return [float(val) for val in sorted_eigenvalues]