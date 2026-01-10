import numpy as np
from matrix_calculation import matrix_multiply_A_transpose_A, matrix_vector_multiply_optimized

def calculate_eigens(W_matrix):
    """Same as before - uses np.linalg.eigh which is allowed"""
    lambdas, V = np.linalg.eigh(W_matrix)
    lambdas = lambdas[::-1]
    V = V[:, ::-1]
    return lambdas, V

def calculate_svd_matrices_optimized(A_norm, lambdas, V_full):
    """
    Optimized SVD calculation with improved U computation
    """
    # Singular values
    sigmas = np.sqrt(np.maximum(lambdas, 0))
    non_zero_sigmas_idx = sigmas > 1e-10
    S_vector = sigmas[non_zero_sigmas_idx]
    
    # V matrix
    V_matrix = V_full[:, non_zero_sigmas_idx]
    
    # Optimized U calculation using matrix-vector multiplication
    M, N = A_norm.shape
    rank = len(S_vector)
    U_matrix = np.zeros((M, rank), dtype=A_norm.dtype)
    
    for i in range(rank):
        sigma_i = S_vector[i]
        v_i = V_matrix[:, i]
        
        # Use optimized matrix-vector multiply instead of full matrix multiply
        A_times_v_i = matrix_vector_multiply_optimized(A_norm, v_i)
        u_i = (1.0 / sigma_i) * A_times_v_i
        U_matrix[:, i] = u_i
    
    return U_matrix, S_vector, V_matrix

# Keep original function name for compatibility
calculate_svd_matrices = calculate_svd_matrices_optimized