import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    p, a = np.array(points), np.array(assignments)
    return [p[a == i].mean(0).tolist() if (a == i).any() else [0, 0] for i in range(k)]