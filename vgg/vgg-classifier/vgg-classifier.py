import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    # Your implementation here
    def relu(x):
        return np.maximum(x, 0)
    batches, h, w, c = features.shape
    x = features.reshape((batches, h * w * c))
    h1 = relu(x @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)
    y = h2 @ W3 + b3
    return y