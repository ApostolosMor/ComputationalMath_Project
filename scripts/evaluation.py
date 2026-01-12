import numpy as np

def calculate_mse(original_channel, compressed_channel):

    # float64 for exact computations
    A_orig = original_channel.astype(np.float64)
    A_comp = compressed_channel.astype(np.float64)

    M = len(original_channel)
    N = len(original_channel[0])

    total = 0.0

    for i in range(M):
        for j in range(N):
            diff = (A_orig[i,j]) - (A_comp[i,j])
            total += diff * diff


    # Mean Squared Error
    mse = total / (M * N)
    return mse

def calculate_compression_ratio(M, N, k):
    original_size = M * N
    
    # k στήλες του U (M*k), k ιδιάζουσες τιμές (k), k στήλες του V (N*k)
    # Total: M*k + k + N*k = k * (M + N + 1)
    compressed_size = k * (M + N + 1)
    
    if compressed_size == 0:
        return np.inf #for safety 
        
    compression_ratio = original_size / compressed_size
    
    return compression_ratio
