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
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        batch_size, T, _ = X.shape
        h_dim, _ = self.W_hh.shape
        X = np.stack(X, axis=1)
        y = [None] * T
        h_t = h_0 if h_0 else np.zeros((batch_size, h_dim))
        for t in range(T):
            h_t = np.tanh(h_t @ self.W_hh.T + X[t] @ self.W_xh.T + self.b_h)
            y[t] = h_t @ self.W_hy.T + self.b_y
        return np.stack(np.array(y), axis=1), h_t