import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x= np.asarray(x, dtype=float)
    fx= 1/(1+np.exp(-x))
    swish=x*fx
    return swish
    # Write code here
    pass