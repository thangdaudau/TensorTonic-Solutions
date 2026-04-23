import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    # YOUR CODE HERE
    relu = lambda x: np.maximum(x, 0)

    x = np.array(x)
    out = relu(x @ conv1)

    skip1 = out
    h1 = relu(out @ W1_b1)
    h1 = h1 @ W2_b1
    out = relu(h1 + skip1)

    skip2 = out @ Ws_b2
    h2 = relu(out @ W1_b2)
    h2 = h2 @ W2_b2
    out = relu(h2 + skip2)

    logits = out @ fc

    return logits