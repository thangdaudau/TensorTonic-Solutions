import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # YOUR CODE HERE
    kernel = 11
    stride = 4
    padding = 0
    filter = 96
    
    batches, h, w, c = image.shape
    h_out = (h - kernel + 2 * padding) // stride + 1
    w_out = (w - kernel + 2 * padding) // stride + 1
    rng = np.random.default_rng()
    W = rng.random((kernel, kernel, c, filter))
    b = rng.random((filter,))

    col = np.lib.stride_tricks.sliding_window_view(image, (kernel, kernel), axis=(1, 2))
    col = col[:, ::stride, ::stride, :, :]
    col = col.reshape(batches * h_out * w_out, -1)
    W_col = W.reshape(-1, filter)

    ret = (col @ W_col + b).reshape(batches, h_out, w_out, filter)

    
    return np.zeros((batches, h_out + 1, w_out + 1, filter))