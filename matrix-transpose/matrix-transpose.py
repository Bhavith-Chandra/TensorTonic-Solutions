import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)          
    N, M = A.shape
    B = np.zeros((M, N), dtype=A.dtype)

    for i in range(N):           # iterate over rows
        for j in range(M):       # iterate over columns
            B[j, i] = A[i, j]    # swap indices (i, j) -> (j, i)

    return B

    pass
