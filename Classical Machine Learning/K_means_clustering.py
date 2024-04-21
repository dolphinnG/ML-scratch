

import numpy as np


def k_means(data, k, n_epoch=100):
    rng = np.random.default_rng(69)
    centroids = data[rng.choice(len(data), k, replace=False)] 
    euclid_distances = np.zeros((len(data), k))
    group_idx = np.zeros(len(data))
    for _ in range(n_epoch): #epochs
        for i in range(k): # calculate distances to each centroid
            euclid_distances[:, i] = np.sum((data - centroids[i]) ** 2, 1) # no need sqrt
        group_idx = np.argmin(euclid_distances, 1)
        for i in range(k): # update centroids
            centroids[i] = np.mean(data[group_idx == i], 0)
    return group_idx, centroids