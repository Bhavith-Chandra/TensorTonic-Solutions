import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    input = x_t @W_xh.T
    hidden = h_prev@W_hh.T
    pre_act = input + hidden + b_h
    h=np.tanh(pre_act)
    return h
    pass