import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    centroids = np.array(centroids)
    points = np.array(points)
    return [np.sum((centroids - p)**2, axis=1).argmin().item() for p in points]