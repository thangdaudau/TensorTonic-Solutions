import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    c, nc = np.unique(y_train, return_counts=True)
    i = np.argmax(nc)
    return np.full(len(X_test), c[i])