import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    A = np.asarray(x)
    B = np.asarray(y)
    dot = np.dot(A,B)
    return dot

    # Write code here
    pass