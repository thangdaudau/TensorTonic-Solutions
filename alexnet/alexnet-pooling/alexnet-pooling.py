import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling (shape simulation)."""
    # YOUR CODE HERE
    batches, h, w, d = x.shape
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    return np.ndarray((batches, h_out, w_out, d))