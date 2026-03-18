import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng == None:
        rng = np.random
    if len(x.shape) == 2:
        n, m = x.shape
        dropout_pattern = np.array([np.array([1 / (1 - p) if rng.random() < 1 - p else 0.0 for _ in range(m)]) for _ in range(n)])
    else:
        n, = x.shape
        dropout_pattern = np.array([1 / (1 - p) if rng.random() < 1 - p else 0.0 for _ in range(n)])
    new_x = x * dropout_pattern
    return new_x, dropout_pattern