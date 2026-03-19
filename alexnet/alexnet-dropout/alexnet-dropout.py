import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Apply dropout to input."""
    # YOUR CODE HERE
    mask = np.random.binomial(1, 1 - p * int(training), x.shape)
    return mask