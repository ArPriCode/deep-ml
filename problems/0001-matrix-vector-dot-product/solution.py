def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float] | int:
    # Handle empty matrix case
    if not a:
        return -1 if len(b) > 0 else []

    # Check dimension compatibility (number of columns in 'a' must equal length of 'b')
    if len(a[0]) != len(b):
        return -1
    
    # Compute the dot product for each row
    c = []
    for row in a:
        row_sum = sum(x * y for x, y in zip(row, b))
        c.append(row_sum)
        
    return c
	