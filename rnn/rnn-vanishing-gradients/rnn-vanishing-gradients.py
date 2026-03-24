import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    norm = np.linalg.norm(W_hh, ord=2)
    g = [0] * T
    g[0] = 1
    for t in range(1, T):
        g[t] = g[t - 1] * norm
    return g