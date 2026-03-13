import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v=np.asarray(v, dtype=float)
    if v.ndim == 1:                                   # LINE 2
        if v.shape[0] != 3:                           # LINE 3
            return None
        norm = np.sqrt(np.sum(v**2))                  # LINE 5
        if norm > 1e-10:                              # LINE 6
            return v / norm                           # LINE 7
        return v.copy()

    if v.ndim == 2:                                   # LINE 9
        if v.shape[1] != 3:                           # LINE 10
            return None                               # LINE 11
        norms = np.sqrt(np.sum(v**2, axis=1, keepdims=True))  # LINE 12
        norms[norms <= 1e-10] = 1                     # LINE 13
        return v / norms                              # LINE 14
    
    return None                                       
    # LINE 4
    # Your code here
    pass