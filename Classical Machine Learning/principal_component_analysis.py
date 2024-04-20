import numpy as np

from singular_value_decomposition import svd_naive

def pca_naive(data, num_components):
    # Perform SVD on the data
    U, S, V = svd_naive(data)

    # Select the top 'num_components' eigenvectors
    components = V[:num_components]

    # Project the data onto the space spanned by selected eigenvectors
    projected_data = np.dot(data, components.T)

    return projected_data

