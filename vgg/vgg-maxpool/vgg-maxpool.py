import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    batches, h, w, c = x.shape
    return x.reshape(batches, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))