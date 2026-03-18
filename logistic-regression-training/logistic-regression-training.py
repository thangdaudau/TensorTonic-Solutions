import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        # L = -np.average(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        dL_dw = X.T @ (p - y) / n_samples
        dL_db = np.average(p - y)

        w -= lr * dL_dw
        b -= lr * dL_db
        
    return w, b