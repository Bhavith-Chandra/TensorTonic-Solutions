import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        return None
    N,M = A.shape
    if N != M:
        return None

    det_A= np.linalg.det(A)
    if abs(det_A) < 1e-10 :
         return None
 

    A_inv = np.linalg.inv(A)

    return A_inv
    # Write code here
    pass
