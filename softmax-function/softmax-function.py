import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x_array = np.asarray(x, dtype=float)
    x_max = np.max(x_array, axis=-1, keepdims=True)
    x_shifted = x_array - x_max
    
    # Exponentiate
    exp_x = np.exp(x_shifted)
    
    # Normalize: sum to 1 along last axis (Hint 2)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Write code here
    pass