import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    _, cnt = np.unique(y, return_counts=True)
    prob = cnt / len(y)
    return np.sum(-prob * np.log2(prob))