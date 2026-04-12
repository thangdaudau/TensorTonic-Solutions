import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    N, d = X_train.shape

    N, d = X_train.shape
    c, Nc = np.unique(y_train, return_counts=True)
    logPc = np.log(Nc / N)

    Xc = [X_train[y_train == ci] for ci in c]
    count = np.array([xc.sum(axis=0) for xc in Xc])
    theta = (count + 1) / (Nc + 2).reshape(c.size, 1)
    
    what = []
    for x in X_test:
        logPosterior = logPc + np.sum(x * np.log(theta) + (1 - x) * np.log(1 - theta), axis=1)
        what += [logPosterior]
    return what