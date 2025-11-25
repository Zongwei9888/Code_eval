import numpy as np

# Pure numpy implementation of block_diag to avoid scipy dependency
def block_diag(*arrays):
    """
    Create a block diagonal matrix from provided arrays.
    
    Args:
        *arrays: Variable number of 2D arrays to place on the diagonal.
        
    Returns:
        Block diagonal matrix with input arrays on the diagonal.
    """
    if len(arrays) == 0:
        return np.array([[]])
    
    # Get shapes of all arrays
    shapes = [np.atleast_2d(a).shape for a in arrays]
    
    # Calculate total size
    total_rows = sum(s[0] for s in shapes)
    total_cols = sum(s[1] for s in shapes)
    
    # Create output matrix filled with zeros
    result = np.zeros((total_rows, total_cols), dtype=np.result_type(*arrays))
    
    # Fill in the diagonal blocks
    row_offset = 0
    col_offset = 0
    for arr, (rows, cols) in zip(arrays, shapes):
        result[row_offset:row_offset + rows, col_offset:col_offset + cols] = np.atleast_2d(arr)
        row_offset += rows
        col_offset += cols
    
    return result

# Gaussian Linear Task (10 params, 10 data)
M_E_gaussian = np.block([
    [np.eye(10), np.zeros((10, 10))],
    [np.eye(10), np.eye(10)]
])

# Two Moons / Gaussian Mixture (2 params, 10 data)
M_E_two_moons = np.block([
    [np.eye(2), np.zeros((2, 10))],
    [np.ones((10, 2)), np.tril(np.ones((10, 10)))]
])

# SLCP (4 params, 8 data)
M_E_slcp = np.block([
    [np.eye(4), np.zeros((4, 8))],
    [np.ones((8, 4)), block_diag(*[np.tril(np.ones((2, 2))) for _ in range(4)])]
])

# Tree Structure (10 variables total)
M_E_tree = np.eye(10)
M_E_tree[0, 1:3] = True
M_E_tree[1, 3:5] = True
M_E_tree[2, 5:7] = True

# HMM (10 params, 10 data)
M_E_hmm = np.block([
    [np.eye(10) + np.diag(np.ones(9), k=-1), np.zeros((10, 10))],
    [np.eye(10), np.eye(10)]
])
