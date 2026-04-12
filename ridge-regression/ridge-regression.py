import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y).T
    N, d = X.shape
    return np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T @ y