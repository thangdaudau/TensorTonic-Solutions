import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.array(X)
    if len(X.shape) == 1:
        n_feature, = X.shape
        if strategy == 'mean':
            mean = np.mean([X[i] for i in range(n_feature) if not np.isnan(X[i])])
            if np.isnan(mean):
                mean = 0
            for i in range(n_feature):
                if np.isnan(X[i]):
                    X[i] = mean
        else:
            med = np.median([X[i] for i in range(n_feature) if not np.isnan(X[i])])
            if np.isnan(med):
                med = 0
            for i in range(n_feature):
                if np.isnan(X[i]):
                    X[i] = med
    else:
        n_sample, n_feature = X.shape
        if strategy == 'mean':
            for j in range(n_feature):
                mean = np.mean([X[i, j] for i in range(n_sample) if not np.isnan(X[i, j])])
                if np.isnan(mean):
                    mean = 0
                for i in range(n_sample):
                    if X[i, j] != X[i, j]:
                        X[i, j] = mean
        else:
            for j in range(n_feature):
                med = np.median([X[i, j] for i in range(n_sample) if not np.isnan(X[i, j])])
                if np.isnan(med):
                    med = 0
                for i in range(n_sample):
                    if X[i, j] != X[i, j]:
                        X[i, j] = med
    return X
    