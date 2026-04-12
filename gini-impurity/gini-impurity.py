import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    if not y_left and not y_right:
        return 0.0
    NL = len(y_left)
    NR = len(y_right)
    _, cntL = np.unique(y_left, return_counts=True)
    _, cntR = np.unique(y_right, return_counts=True)
    probL = cntL / NL
    probR = cntR / NR
    giniL = 1 - np.sum(probL * probL)
    giniR = 1 - np.sum(probR * probR)
    return (NL * giniL + NR * giniR) / (NL + NR)