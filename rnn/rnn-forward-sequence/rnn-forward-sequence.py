import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    batch_size, T, _ = X.shape
    X = np.stack(X, axis=1)
    h = [None] * T + [h_0]
    for t in range(T):
        h[t] = np.tanh(h[t - 1] @ W_hh.T + X[t] @ W_xh.T + b_h)
    return np.stack(np.array(h[:-1]), axis=1), h[-2]