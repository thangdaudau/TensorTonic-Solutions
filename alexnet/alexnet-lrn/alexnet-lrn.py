import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    # YOUR CODE HERE
    batches, h, w, d = x.shape
    new_x = np.ndarray(x.shape)
    for k in range(batches):
        for i in range(h):
            for j in range(w):
                for t in range(d):
                    l = max(0, t - d // 2)
                    r = min(d - 1, i + d // 2) + 1
                    a = x[k, i, j, l:r]
                    sa = sum(a * a)
                    new_x[k, i, j, t] = x[k, i, j, t] / np.pow(k + alpha * sa, beta)
    return new_x