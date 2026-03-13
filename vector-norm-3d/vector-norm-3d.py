import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v,dtype = float)
    if v.ndim == 1:                                   # LINE 2
        if v.shape[0] != 3:                           # LINE 3
            return None                               # LINE 4
        return float(np.sqrt(np.sum(v**2)))           # LINE 5
    
    # Batch case  
    if v.ndim == 2:                                   # LINE 6
        if v.shape[1] != 3:                           # LINE 7
            return None                               # LINE 8
        return np.sqrt(np.sum(v**2, axis=1))         # LINE 9
    
    return None                                       # LINE 10
