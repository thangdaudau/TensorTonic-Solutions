import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    batches, h, w, c = x.shape
    h_out, w_out, c_out = h // 2, w // 2, c
    
    y = np.ndarray((batches, h_out, w_out, c_out))
    for b in range(batches):
        for i in range(h_out):
            for j in range(w_out):
                y[b, i, j] = np.maximum.reduce([
                    x[b, 2 * i, 2 * j],
                    x[b, 2 * i + 1, 2 * j],
                    x[b, 2 * i, 2 * j + 1],
                    x[b, 2 * i + 1, 2 * j + 1]
                ])
    return y