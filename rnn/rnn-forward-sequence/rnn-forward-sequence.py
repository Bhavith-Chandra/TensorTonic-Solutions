import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size, seq_len, _ = X.shape
    h_prev = h_0                                 # (batch_size, hidden_dim)

    hidden_states = []                           # will collect T tensors of shape (B, H)

    for t in range(seq_len):
        x_t = X[:, t, :]                         # (batch_size, input_dim)

        # RNN cell logic for one time step
        input_term  = x_t @ W_xh.T              # (B, H)
        hidden_term = h_prev @ W_hh.T           # (B, H)
        pre_act = input_term + hidden_term + b_h
        h_t = np.tanh(pre_act)                  # (B, H), values in [-1, 1]

        hidden_states.append(h_t)
        h_prev = h_t

    # Stack along time axis: list length = seq_len, each (B, H)
    # np.stack -> (seq_len, B, H), then transpose to (B, seq_len, H)
    h_all = np.stack(hidden_states, axis=1)      # (batch_size, seq_len, hidden_dim)

    h_final = h_all[:, -1, :]                    # last time step (B, H)

    return h_all, h_final
    # YOUR CODE HERE
    pass