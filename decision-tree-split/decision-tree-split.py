import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    N, d = X.shape
    def Gini(S):
        ySubset = y[S]
        s = ySubset.shape[0]
        if s == 0:
            return 0, float('inf')
        is2 = 1 / (s * s)
        c0 = np.count_nonzero(ySubset == 0)
        c1 = s - c0
        gini = 1 - c0 * c0 * is2 - c1 * c1 * is2
        return s, gini
    
    feature = -1
    threshold = 0
    bestImpurity = float('inf')

    index = list(range(N))
    for j in range(d):
        index.sort(key=lambda idx: X[idx, j])
        for i in range(N - 1):
            idx = index[i]
            nxt = index[i + 1]
            if y[idx] == y[nxt]:
                continue
            currentThreshold = (X[idx, j] + X[nxt, j]) * 0.5
            mask = X[:, j] > currentThreshold
            sL, giniL = Gini(mask)
            sR, giniR = Gini(~mask)
            gini = sL * giniL + sR * giniR
            if gini < bestImpurity:
                bestImpurity = gini
                feature = j
                threshold = currentThreshold

    return feature, threshold