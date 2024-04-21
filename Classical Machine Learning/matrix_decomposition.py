

import numpy as np
from sympy import symbols, Matrix, eye, solve, simplify, N, re

from utils.linear_algebra import add_orthogonal_columns, do_gram_schmidt_process, pad_diagonal_matrix

from sympy import re

# def compute_Vs_from_Us(U, sigma, W):
#     s_inv = np.diag(1/np.diag(sigma))
#     # g_Us = W @ U @ s_inv
#     g_Vs = W.T @ U[:, :-1] @ s_inv
#     return g_Vs
# def compute_Us_from_Vs(V, sigma, W):
#     s_inv = np.diag(1/np.diag(sigma))
#     g_Us = W @ V @ s_inv
#     # g_Vs = W.T @ V[:, :-1] @ s_inv
#     return g_Us

def eigen_decomposition_by_QR_approximation(A, n_iter=1000):
    """The final Q and R will not reconstruct A doing Q@R, 
    but still have the eigenvalues approximated on the diagonal of R.
    Q_total is the eigenvectors, R is the eigenvalues diagonal matrix  
    
    returns: R, Q_total
    """
    Q_total = np.eye(A.shape[0])
    Q, R = QR_decomposition(A)
    Q_total = Q_total @ Q
    for i in range(n_iter):
        Q, R = QR_decomposition(R @ Q)
        Q_total = Q_total @ Q
    return R, Q_total #Q_total is the eigenvectors, R is the eigenvalues diagonal matrix


def QR_decomposition(A):
    Q = do_gram_schmidt_process(A)
    R = Q.T @ A
    return Q, R

def lu_decomposition(A):
    """The function then enters a loop that iterates over each column j of U, except the last one. For each column j, it does the following:
It creates an identity matrix L of the same size as A.
It calculates a vector gamma of multipliers for the Gaussian elimination. 
The elements of gamma are the elements of the j-th column of U below the diagonal, divided by the diagonal element U[j, j].
It subtracts gamma times the j-th row of U from the rows of U below the j-th row. 
This is done by subtracting gamma from the j-th column of L below the diagonal, and then multiplying L and U . 
This operation transforms U into an upper triangular matrix.
It restores the j-th column of L below the diagonal to gamma, effectively storing the multipliers used in the Gaussian elimination.
It appends L to the list Ls."""

    n = A.shape[0]
    U = A.copy()
    Ls = []

    for j in range(n-1):
        L = np.eye(n)
        gamma = U[j+1:, j] / U[j, j]
        L[j+1:, j] = -gamma
        U = L @ U

        L[j+1:, j] = gamma #storing the inverse of L
        # the inverse of row operation matrices Ls are obtained by 
        # just flipping the sign of the off diagonal elements
        Ls.append(L)

    return U, Ls

def inverse_iteration(A, lambda_approx, num_iterations=10, shift=.01):
    # A = np.array([[4, 1], [2, 3]]).astype(float) #Specify the known eigenvalue
    # Identity matrix
    I = np.eye(A.shape[0])
    # Compute A - lambda * I
    """In the context of the inverse iteration method, the matrix (A - λI) becomes singular 
    when λ is an eigenvalue of A. This is because (A - λI) has a zero determinant when λ is an eigenvalue, 
    which is a characteristic property of eigenvalues.
    To avoid this issue, a common approach is to add a small shift σ to the eigenvalue λ when forming the matrix (A - λI). 
    This shift moves λ away from the exact eigenvalue, preventing (A - λI) from becoming singular. 
    The modified matrix becomes (A - (λ+σ)I), 
    and the inverse iteration method can proceed without encountering a singular matrix."""
    B = (A - (lambda_approx - shift) * I).astype(float)
    # Initial guess for the eigenvector
    x = np.random.rand(A.shape[0])
    x = x / np.linalg.norm(x)  # Normalize the initial vector

    # Inverse Iteration
    for _ in range(num_iterations):    
        x = np.linalg.solve(B, x)  # Solve Bx = x_old    
        x = x / np.linalg.norm(x)  # Normalize the vector
    # print("Approximate eigenvector:", x)
    return x

def eigendecomposition_test_return_eigenvalues_diagmatrix(A):    
    lamda = symbols('λ')        
    """N(x, 50) is used to evaluate each element of A to 50 decimal places, 
    and N(re(e)) is used to convert the eigenvalues to numerical values. 
    This should help to reduce numerical issues in the computation of the eigenvalues."""
    A = Matrix(A).applyfunc(lambda x: N(x, 50))  # Use higher precision    
    I = eye(A.shape[0])        
    # # Characteristic equation det(A - λI) = 0    
    characteristic_eq_matrix = A - lamda * I    
    poly = characteristic_eq_matrix.det()  
    eigenvalues = solve(poly, lamda)
    print('eigenvalues', eigenvalues)
    eigenvalues = np.array([N(re(e)) for e in eigenvalues])    
    eigenvalues = np.sort(eigenvalues)[::-1]    
# eigenvectors = []    
# for e_val in eigenvalues:        
# # Solve (A - λI)v = 0        
#     eig_vec = (A - e_val * I).nullspace() 
#     #unable to cal null space due to numerical issues
#     eigenvectors.append(eig_vec)        
# V = Matrix([vec[0] for vec in eigenvectors if vec]).T        
    D = Matrix.diag(*eigenvalues)    
    return D