import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    N, d = X_train.shape
    c, Nc = np.unique(y_train, return_counts=True)
    logPc = np.log(Nc / N)

    Xc = [X_train[y_train == ci] for ci in c]
    mean = np.array([xc.mean(axis=0) for xc in Xc])
    var = np.array([xc.var(axis=0) for xc in Xc]) + 1e-9
    
    ans = []
    for x in X_test:
        logPosterior = logPc + np.sum(-0.5 * np.log(var) - (x - mean)**2 / (2 * var), axis=1)
        ans += [c[np.argmax(logPosterior)]]
    return ans