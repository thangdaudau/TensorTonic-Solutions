import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last, C_last)."""
        # YOUR CODE HERE
        batch_size, T, input_dim = X.shape
        hidden_dim = self.hidden_dim
        X = np.stack(X, axis=1)
        y = [None] * T
        h = np.zeros((batch_size, hidden_dim))
        C = np.zeros((batch_size, hidden_dim))

        for t in range(T):
            hx = np.concat([h, X[t]], axis=1)
            f_t = sigmoid(hx @ self.W_f.T + self.b_f)
            i_t = sigmoid(hx @ self.W_i.T + self.b_i)
            c_t = np.tanh(hx @ self.W_c.T + self.b_c)
            o_t = sigmoid(hx @ self.W_o.T + self.b_o)

            C_t = f_t * C + i_t * c_t
            h_t = o_t * np.tanh(C_t)
            y[t] = h_t @ self.W_y.T + self.b_y
            h, C = h_t, C_t
            
        return np.stack(np.array(y), axis=1), h, C