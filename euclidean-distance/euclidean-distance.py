import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype= float)
    y= np.asarray(y, dtype = float)
    # Write code here
    diff = x-y
    dist = np.linalg.norm(diff)
    return dist
    pass