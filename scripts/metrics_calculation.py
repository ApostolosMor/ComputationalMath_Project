import numpy as np

def matrix_multiply_optimized(A, B, block_size=64):
    """
    Block-based matrix multiplication for better cache performance
    Similar to composite integration methods
    """
    # Check compatibility
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Dimensions not compatible: A {A.shape}, B {B.shape}")
    
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=A.dtype)
    
    # Process in blocks (like composite trapezoidal)
    for i0 in range(0, M, block_size):
        i1 = min(i0 + block_size, M)
        for j0 in range(0, N, block_size):
            j1 = min(j0 + block_size, N)
            # Pre-allocate block accumulator
            block_acc = np.zeros((i1-i0, j1-j0), dtype=A.dtype)
            
            for k0 in range(0, K, block_size):
                k1 = min(k0 + block_size, K)
                # Extract small blocks
                A_block = A[i0:i1, k0:k1]
                B_block = B[k0:k1, j0:j1]
                
                # Multiply small blocks with vectorization-like approach
                for ii in range(i1-i0):
                    for kk in range(k1-k0):
                        a_val = A_block[ii, kk]
                        # Inner loop with manual optimization
                        row_mult = a_val * B_block[kk, :]
                        for jj in range(j1-j0):
                            block_acc[ii, jj] += row_mult[jj]
            
            # Copy block result to output
            for ii in range(i1-i0):
                for jj in range(j1-j0):
                    C[i0+ii, j0+jj] = block_acc[ii, jj]
    
    return C

def matrix_transpose_optimized(A, block_size=64):
    """
    Cache-optimized transposition using block processing
    """
    M, N = A.shape
    A_T = np.zeros((N, M), dtype=A.dtype)
    
    # Process in blocks
    for i0 in range(0, M, block_size):
        i1 = min(i0 + block_size, M)
        for j0 in range(0, N, block_size):
            j1 = min(j0 + block_size, N)
            # Transpose block
            for i in range(i0, i1):
                for j in range(j0, j1):
                    A_T[j, i] = A[i, j]
    
    return A_T

def matrix_multiply_A_transpose_A(A):
    """
    Specialized computation of W = AᵀA exploiting symmetry
    Direct computation without full transpose
    """
    M, N = A.shape
    W = np.zeros((N, N), dtype=A.dtype)
    
    # Compute only upper triangle (W is symmetric)
    for i in range(N):
        # Diagonal element
        diag_sum = 0.0
        for k in range(M):
            a_val = A[k, i]
            diag_sum += a_val * a_val
        W[i, i] = diag_sum
        
        # Off-diagonal elements (upper triangle)
        for j in range(i + 1, N):
            sum_val = 0.0
            for k in range(M):
                sum_val += A[k, i] * A[k, j]
            W[i, j] = sum_val
            W[j, i] = sum_val  # Fill symmetric position
    
    return W

def matrix_vector_multiply_optimized(A, v):
    """
    Optimized matrix-vector multiplication for U calculation
    """
    M, N = A.shape
    result = np.zeros(M, dtype=A.dtype)
    
    # Process with better memory access pattern
    for i in range(M):
        sum_val = 0.0
        # Process in chunks of 4 for slight optimization
        for j in range(0, N, 4):
            j_end = min(j + 4, N)
            for jj in range(j, j_end):
                sum_val += A[i, jj] * v[jj]
        result[i] = sum_val
    
    return result

# Keep original functions for compatibility but use optimized ones
matrix_multiply = matrix_multiply_optimized
matrix_transpose = matrix_transpose_optimized