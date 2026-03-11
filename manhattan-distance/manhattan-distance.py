import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = x.dtype)
    
    dist = np.linalg.norm(x-y, ord=1)
    return dist
    pass