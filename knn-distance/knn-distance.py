import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    if X_test.size == 0:
        return np.empty((0, k), dtype=int)
    def take(x):
        d = (x - X_train)**2
        if len(d.shape) == 2:
            d = d.sum(1)
        k_eff = min(k, X_train.shape[0])
        return np.concatenate([np.argpartition(d, k_eff - 1)[:k_eff], -np.ones(k - k_eff)])
    return np.array(list(map(take, X_test)), dtype='int')