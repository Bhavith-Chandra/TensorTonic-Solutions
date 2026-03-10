import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    A = np.asarray(v, dtype=float)
    n = A.shape[0]
    D=np.zeros((n,n), dtype=A.dtype)
    for i in range(n):
        D[i,i]= A[i]
    # Write code here
    return D
    pass
