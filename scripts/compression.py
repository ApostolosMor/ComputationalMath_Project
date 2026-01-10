import numpy as np
from PIL import Image

def reconstruct_channel_optimized(U, S_vector, V, k):
    """
    Optimized reconstruction using direct outer product sum
    Avoids full matrix multiplies for U_k @ diag(S_k) @ V_k.T
    """
    M, N = U.shape[0], V.shape[0]
    A_k = np.zeros((M, N), dtype=np.float64)
    
    # Direct summation: A_k = Σ σ_i * u_i * v_iᵀ
    for i in range(min(k, len(S_vector))):
        sigma = S_vector[i]
        u_i = U[:, i]
        v_i = V[:, i]
        
        # Compute outer product efficiently
        for row in range(M):
            u_val = sigma * u_i[row]
            # Use small inner loop for columns
            for col in range(N):
                A_k[row, col] += u_val * v_i[col]
    
    # Denormalize and clip
    A_k_denorm = A_k * 255.0
    A_k_clipped = np.clip(A_k_denorm, 0, 255).astype(np.uint8)
    return A_k_clipped

def reconstruct_channels_progressive(U, S, V, k_values):
    """
    Reconstruct for multiple k values efficiently
    Reuses computations from smaller k to build larger k
    """
    k_values_sorted = sorted(k_values)
    results = {}
    
    # Start with empty reconstruction
    M, N = U.shape[0], V.shape[0]
    current_reconstruction = np.zeros((M, N), dtype=np.float64)
    current_k = 0
    
    for target_k in k_values_sorted:
        # Add components from current_k to target_k
        for i in range(current_k, min(target_k, len(S))):
            sigma = S[i]
            u_i = U[:, i]
            v_i = V[:, i]
            
            # Add this component to current reconstruction
            for row in range(M):
                u_val = sigma * u_i[row]
                for col in range(N):
                    current_reconstruction[row, col] += u_val * v_i[col]
        
        # Store result for this k
        A_k_denorm = current_reconstruction * 255.0
        A_k_clipped = np.clip(A_k_denorm, 0, 255).astype(np.uint8)
        results[target_k] = A_k_clipped
        
        current_k = target_k
    
    return results

def merge_and_save_image(R_k, G_k, B_k, k, original_shape):
    """Same as before"""
    compressed_image_np = np.dstack((R_k, G_k, B_k))
    compressed_image = Image.fromarray(compressed_image_np, 'RGB')
    output_filename = f'compressed_k{k}.png'
    compressed_image.save(output_filename)
    print(f"Εικόνα k={k} αποθηκεύτηκε ως: {output_filename}")
    return compressed_image_np

# Use optimized reconstruction
reconstruct_channel = reconstruct_channel_optimized