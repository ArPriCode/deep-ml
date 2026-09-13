def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    """Transpose a 2D matrix by swapping rows and columns."""
    if not a:
        return []
    
    rows, cols = len(a), len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]
    