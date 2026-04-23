import numpy as np
EPS = 1e-5
def relu(x):
    return np.maximum(x, 0)
def batch_norm(x, gamma, beta):
    mean = x.mean(0)
    var = x.var(0)
    return gamma * (x - mean) / np.sqrt(var + EPS) + beta
def conv(x, W):
    return x @ W
def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    x = np.array(x)
    if mode == 'post':
        y = relu(batch_norm(conv(x, W1), gamma1, beta1))
        y = relu(batch_norm(conv(y, W2), gamma2, beta2) + x)
    else:
        y = conv(relu(batch_norm(x, gamma1, beta1)), W1)
        y = conv(relu(batch_norm(y, gamma2, beta2)), W2) + x
    return {"output": y, "mode": mode}