

import numpy as np
from matrix_decomposition import eigen_decomposition_by_QR_approximation, inverse_iteration
from utils.linear_algebra import add_orthogonal_columns, pad_diagonal_matrix


def svd_naive(A):
    
    AAt = A @ A.T
    AtA = A.T @ A
    
    #AAt decomposition:
    # D = eigendecomposition_test_return_eigenvalues_diagmatrix(AAt)
    # eig_values_AAt = np.diag(D)
    # eig_values_AAt = [re(val) for val in eig_values_AAt]
    rA_dim, cA_dim = A.shape
    psd_matrix = AAt if rA_dim <= cA_dim else AtA
    R, Q = eigen_decomposition_by_QR_approximation(psd_matrix, 1000)
    eig_values_AAt = np.diag(R)
    eig_vectors_AAt = []
    for eig_val in eig_values_AAt:
        eig_vecz = inverse_iteration(AAt, eig_val, 1000)
        eig_vectors_AAt.append(eig_vecz)
    # print(eig_vectors_AAt)   
    
    #AAt decomposition:
    # D = eigendecomposition_test_return_eigenvalues_diagmatrix(AtA)
    # eig_values_AtA = np.diag(D)
    # eig_values_AtA = [re(val) for val in eig_values_AtA]
    eig_vectors_AtA = []
    for eig_val in eig_values_AAt: #both eigenvalues are the same
        eig_vecz = inverse_iteration(AtA, eig_val, 1000)
        eig_vectors_AtA.append(eig_vecz)   
    # print(eig_vectors_AtA)
    
    #AtA decomposition: #implementation of QR algo but only for square matrices
    # R, Q = eigen_decomposition_by_QR_approximation(AtA, 1000)
    # V = Q
    
    
    sigma = np.diag(np.sqrt(np.array([e for e in eig_values_AAt if e != 0 ]).astype(float)))
    # print(sigma)
    if sigma.shape != A.shape:
        sigma = pad_diagonal_matrix(sigma, A.shape)
    # sigma_R = np.sqrt(R)
    eig_vectors_AAt = [vec.reshape(-1, 1) for vec in eig_vectors_AAt]
    U = np.hstack(eig_vectors_AAt)
    eig_vectors_AtA = [vec.reshape(-1, 1) for vec in eig_vectors_AtA]
    V = np.hstack(eig_vectors_AtA)
    
    s_inv = np.diag(1/np.diag(sigma))
    if rA_dim < cA_dim:
        additional_cols =  A.shape[1] - V.shape[1]
        V = add_orthogonal_columns(V, additional_cols) #wide matrix A case
        # U = compute_Us_from_Vs(V, sigma, A)
        g_Us = A @ V[:,:-additional_cols] @ s_inv
        U = g_Us
        
    elif rA_dim > cA_dim:
        additional_cols =  A.shape[0] - V.shape[1]
        U = add_orthogonal_columns(U, additional_cols) #tall matrix A case
        # V = compute_Vs_from_Us(U, sigma, A)
        g_Vs = A.T @ U[:, :-additional_cols] @ s_inv
        V = g_Vs
    else:
        # V = compute_Vs_from_Us(U, sigma, A)
        g_Vs = A.T @ U @ s_inv
        V = g_Vs
# #https://math.stackexchange.com/questions/2359992/how-to-resolve-the-sign-issue-in-a-svd-problem
    return U, sigma, V 


def sign_flip(X, U, V, S, tol = 1):

    m, n = X.shape
    U_flipped = U.copy()
    V_flipped = V.copy()
    # Step 1
    s_lefts = []
    # Y = X - U @ S @ V.T
    Y = X
    for k in range(len(U)):
        # Y = X - U[:,:k+1] @ S[:k+1,:k+1] @ V[:,:k+1].T
        # print('step 1: ', k)     
        s_left = np.sum(np.sign(U[:,k].T @ Y)*((U[:,k].T @ Y)**2))
        # s_left = np.sum(np.sign(U[:,k].T @ Y)*(np.abs(U[:,k].T @ Y)))
        s_lefts.append(s_left)
        
    # Step 2
    s_rights = []
    for k in range(len(V)):
        # print('step 2: ', k)
        # Y = X - U[:,:k+1] @ S[:k+1,:k+1] @ V[:,:k+1].T
        s_right = np.sum(np.sign(V[:,k].T @ Y.T)*((V[:,k].T @ Y.T)**2))
        # s_right = np.sum(np.sign(V[:,k].T @ Y.T)*(np.abs(V[:,k].T @ Y.T)))
        s_rights.append(s_right)

    # Step 3
    print('s_lefts: ', s_lefts)
    print('s_rights: ', s_rights)
    for k in range(len(U)):
        s_left = s_lefts[k]
        s_right = s_rights[k]
        if np.abs(s_left) < tol or np.abs(s_right) < tol:
            print('skipped: k={0}, left={1}, right={2}'.format(k, s_left, s_right))
            # continue
        if s_left * s_right < 0 and not (np.abs(s_left) < tol or np.abs(s_right) < tol):
            if np.abs(s_left) < np.abs(s_right):
                # s_left = s_lefts[k] = -s_left
                s_lefts[k] = -1
                s_rights[k] = 1
            else:
                # s_right = s_rights[k] = -s_right
                s_lefts[k] = 1
                s_rights[k] = -1
        else:
            s_lefts[k] = 1
            s_rights[k] = 1
                
        U_flipped[:, k] = U[:, k] * np.sign(s_lefts[k])
        V_flipped[:, k] = V[:, k] * np.sign(s_rights[k])
    print('s_lefts: ', s_lefts)
    print('s_rights: ', s_rights)
    return U_flipped, V_flipped