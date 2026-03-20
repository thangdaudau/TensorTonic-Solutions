import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        func = np.nanmean if strategy == 'mean' else np.nanmedian
        val = func(X)
        if np.isnan(val):
            val = 0
        return np.where(np.isnan(X), val, X)

    # 2D
    func = np.nanmean if strategy == 'mean' else np.nanmedian
    vals = func(X, axis=0)
    vals = np.where(np.isnan(vals), 0, vals)

    inds = np.where(np.isnan(X))
    X[inds] = vals[inds[1]]
    return X