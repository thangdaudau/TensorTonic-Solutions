import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # YOUR CODE HERE
    kernel = 11
    stride = 4
    # padding = 0
    filter = 96
    
    batches, h, w, d = image.shape
    h_out = (h - kernel + 2 * 0 + stride - 1) // stride + 1
    w_out = (w - kernel + 2 * 0 + stride - 1) // stride + 1
    ret = np.zeros((batches, h_out, w_out, filter))
    
    rng = np.random.default_rng()
    W = rng.random((kernel, kernel, d, filter))
    b = rng.random((filter,))

    for bat in range(batches):
        for i in range(h_out):
            for j in range(w_out):
                ret[bat, i, j] = b.copy()
                for n in range(kernel):
                    for m in range(kernel):
                        h_in = i * stride + n
                        w_in = j * stride + m
                        if h_in < h and w_in < w:
                            ret[bat, i, j] += image[bat, h_in, w_in] @ W[n, m]
    return ret