
import numpy as np

def gaussian_elemination(A, b):
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]
            A[j] -= ratio * A[i]
            b[j] -= ratio * b[i]
            print(A, b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]
    return x


def gaussian_elimination_rref_square(A, b):
    assert A.shape[0] == A.shape[1]
    #square matrix only
    n = len(b)
    A_copy = A.copy()
    b_copy = b.copy()
    A_copy = np.hstack((A_copy, b_copy))
    np.set_printoptions(precision=3)
    for i in range(n):
        # Forward elimination
        for j in range(i+1, n):
            ratio = A_copy[j][i] / A_copy[i][i]
            A_copy[j] -= ratio * A_copy[i]
        # Backward elimination
        for j in range(i-1, -1, -1):
            ratio = A_copy[j][i] / A_copy[i][i]
            A_copy[j] -= ratio * A_copy[i]

    # Normalize rows
    for i in range(n):
        A_copy[i] /= A_copy[i][i]
        
    return A_copy[:, :-b_copy.shape[1]], A_copy[:, -b_copy.shape[1]:]

def l2_norm(x):
    return np.sqrt(np.sum(x**2)) 

def calculate_vector_projection(v, u):
    return np.dot(v, u) / np.dot(u, u) * u

def do_gram_schmidt_process(A, n = 0):
    Q = np.zeros_like(A)
    # Copy the first n columns from A to Q
    Q[:, :n] = A[:, :n]
    for col_idx in range(n, A.shape[1]):
        v = A[:, col_idx]
        for i in range(col_idx):
            # repeatedly deduct the projection of v onto the previous vectors
            u = Q[:, i]
            v = v - calculate_vector_projection(v, u)
        normalized = v / l2_norm(v)
        Q[:, col_idx] = normalized 
    return Q

def add_orthogonal_columns(V, k):
    m, n = V.shape
    # Create a larger matrix A with k additional random columns
    A = np.hstack((V, np.random.rand(m, k)))
    # Apply the Gram-Schmidt process starting from the column after the last column of V
    Q = do_gram_schmidt_process(A, n)
    return Q

def pad_diagonal_matrix(S_diag, shape):
    m, n = shape
    print(S_diag.shape, shape)
    assert np.all(np.array(shape) - np.array(S_diag.shape) >= np.array([0,0])), "shape must be less than diagonal matrix shape"
    S_padded = np.pad(S_diag, ((0, m - S_diag.shape[0]), (0, n - S_diag.shape[1])))

    return S_padded

