import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    dtanh = dh_next * (1 - h_t**2)  
    dh_prev = dtanh @ W_hh
    dW_hh = h_prev.T @ dtanh
    return dh_prev, dW_hh
    # YOUR CODE HERE
    pass