def dp_drills(weights, values, capacity, m, n, s1, s2):
    n_items = len(weights)
    dp_k = [[0] * (capacity + 1) for _ in range(n_items + 1)]
    for i in range(1, n_items + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp_k[i][w] = max(
                    dp_k[i - 1][w],
                    dp_k[i - 1][w - weights[i - 1]] + values[i - 1]
                )
            else:
                dp_k[i][w] = dp_k[i - 1][w]
    knapsack_val = dp_k[n_items][capacity]

    dp_g = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0:
                dp_g[i][j] = 1
            else:
                dp_g[i][j] = dp_g[i - 1][j] + dp_g[i][j - 1]
    grid_paths_val = dp_g[m - 1][n - 1]

    len1, len2 = len(s1), len(s2)
    dp_l = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp_l[i][j] = dp_l[i - 1][j - 1] + 1
            else:
                dp_l[i][j] = max(dp_l[i - 1][j], dp_l[i][j - 1])
    lcs_val = dp_l[len1][len2]

    return {
        'knapsack': knapsack_val,
        'grid_paths': grid_paths_val,
        'lcs': lcs_val,
        'complexity': {
            'knapsack': 'O(n*W) time, O(n*W) space',
            'grid_paths': 'O(m*n) time, O(m*n) space',
            'lcs': 'O(m*n) time, O(m*n) space'
        }
    }