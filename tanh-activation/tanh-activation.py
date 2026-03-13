import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x)
    x = (np.exp(x)-np.exp(-x))/ (np.exp(x)+np.exp(-x))
    return x
    pass