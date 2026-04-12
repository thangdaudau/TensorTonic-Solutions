import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points = np.array(points)
    assignments = np.array(assignments)
    
    c = np.unique(assignments)
    newc = [np.mean(points[assignments == ci], axis=0) for ci in c]

    ret = [[0, 0] for _ in range(k)]
    for i in range(len(c)):
        ret[c[i]] = newc[i].tolist()
    return ret