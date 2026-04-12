import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.array(X)
    N, d = X.shape
    
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / (N - 1)
    
    eival, eivec = np.linalg.eigh(C)
    idx = np.argsort(eival)[::-1]
    
    W = eivec.T[idx[:k]]
    
    return Xc @ W.T