import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    # Write code here
    ans = {}
    for k in train_dist.keys() & serving_dist.keys():
        p = np.array(train_dist[k]) + eps
        q = np.array(serving_dist[k]) + eps
        psi = np.sum((q - p) * np.log(q / p)).item()
        skewed = psi >= threshold
        ans[k] = {"psi": psi, "skewed": skewed}
    return ans
    