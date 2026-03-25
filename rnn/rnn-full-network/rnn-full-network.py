import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        batch_size, seq_len, _ = X.shape
        
        # Initialize h_0 if not provided (zeros)
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_dim))
        
        h_prev = h_0
        hidden_states = []
        
        # Step 1: RNN forward pass (like your rnn_forward)
        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, input_dim)
            
            # RNN cell: h_t = tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
            input_part = x_t @ self.W_xh.T
            hidden_part = h_prev @ self.W_hh.T
            h_t = np.tanh(input_part + hidden_part + self.b_h)
            
            hidden_states.append(h_t)
            h_prev = h_t
        
        # Stack: (batch, seq_len, hidden_dim)
        h_all = np.stack(hidden_states, axis=1)
        h_final = h_all[:, -1, :]
        
        # Step 2: Output projection Y = W_hy @ h_all + b_y
        # Reshape for batch matrix multiply: (batch*seq_len, hidden) @ (hidden, output)
        batch_seq, hidden_dim = h_all.shape[0], h_all.shape[2]
        h_flat = h_all.reshape(-1, hidden_dim)  # (batch*seq_len, hidden)
        y_flat = h_flat @ self.W_hy.T + self.b_y  # (batch*seq_len, output)
        Y = y_flat.reshape(batch_size, seq_len, -1)  # (batch, seq_len, output)
        
        return Y, h_final
