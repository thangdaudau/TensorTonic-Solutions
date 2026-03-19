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
    dropout_pattern = rng.binomial(1, 1 - p, x.shape) / (1 - p)
    new_x = x * dropout_pattern
    return new_x, dropout_pattern