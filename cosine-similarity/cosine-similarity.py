import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    A= np.asarray(a)
    B= np.asarray(b)
    norm_a= np.linalg.norm(A)
    norm_b= np.linalg.norm(B)
    if norm_a ==0 or norm_b==0:
        return 0
        
    sim = np.dot(A,B)/(norm_a*norm_b)
    return sim
    # Write code here
    pass