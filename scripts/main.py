import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if 'scripts' not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_split import load_and_split_image, normalize_and_prepare_w, IMAGE_PATH
from svd_core import calculate_eigens, calculate_svd_matrices
from compression import reconstruct_channels_progressive, merge_and_save_image
from evaluation import calculate_mse, calculate_compression_ratio
from matrix_calculation import matrix_multiply_A_transpose_A

RANKS_TO_TEST = [5, 20, 50, 100]

def process_channel_optimized(A_channel, channel_name):
    print(f"\n--- Επεξεργασία {channel_name} Καναλιού (Βήματα 2, 3, 4) ---")
    
    # Βήμα 2: Ομαλοποίηση
    A_norm = A_channel / 255.0
    
    # Optimized: Direct computation of W = AᵀA without full transpose
    W = matrix_multiply_A_transpose_A(A_norm)
    
    # Βήμα 3: Ιδιοτιμές/Ιδιοδιανύσματα
    lambdas, V_full = calculate_eigens(W)
    
    # Βήμα 4: SVD matrices (optimized)
    U, S_vector, V = calculate_svd_matrices(A_norm, lambdas, V_full)
    
    rank = len(S_vector)
    print(f" Διαστάσεις {channel_name}: {A_norm.shape}")
    print(f" Rank: {rank}")
    print(f" Μεγαλύτερη σ1: {S_vector[0]:.4f}")
    
    return U, S_vector, V, A_norm

def run_compression_pipeline_optimized():
    print("=========================================")
    print(f"Ξεκινά η ΒΕΛΤΙΣΤΟΠΟΙΗΜΕΝΗ SVD Συμπίεση: {IMAGE_PATH}")
    print("=========================================")
    
    try:
        # Φόρτωση εικόνας
        R_channel, G_channel, B_channel, original_shape = load_and_split_image(IMAGE_PATH)
        
        # Process each channel with optimized methods
        print("\n--- Υπολογισμός SVD για όλα τα κανάλια ---")
        U_R, S_R, V_R, R_norm = process_channel_optimized(R_channel, "Κόκκινο")
        U_G, S_G, V_G, G_norm = process_channel_optimized(G_channel, "Πράσινο")
        U_B, S_B, V_B, B_norm = process_channel_optimized(B_channel, "Μπλε")
        
        # Use progressive reconstruction for efficiency
        print("\n--- Βήματα 5, 6 & Αξιολόγηση ---")
        M, N, _ = original_shape
        results_table = []
        compressed_images = []
        
        # Progressive reconstruction for each channel
        print("Προοδευτική ανακατασκευή καναλιών...")
        R_results = reconstruct_channels_progressive(U_R, S_R, V_R, RANKS_TO_TEST)
        G_results = reconstruct_channels_progressive(U_G, S_G, V_G, RANKS_TO_TEST)
        B_results = reconstruct_channels_progressive(U_B, S_B, V_B, RANKS_TO_TEST)
        
        # Combine and evaluate
        for k in RANKS_TO_TEST:
            print(f"Αξιολόγηση για k = {k}...")
            
            # Get precomputed reconstructions
            R_k = R_results[k]
            G_k = G_results[k]
            B_k = B_results[k]
            
            # Merge and save
            compressed_image_np = merge_and_save_image(R_k, G_k, B_k, k, original_shape)
            compressed_images.append(compressed_image_np)
            
            # Calculate MSE
            mse_r = calculate_mse(R_channel, R_k)
            mse_g = calculate_mse(G_channel, G_k)
            mse_b = calculate_mse(B_channel, B_k)
            avg_mse = (mse_r + mse_g + mse_b) / 3.0
            
            # Compression ratio
            cr = calculate_compression_ratio(M, N, k)
            
            results_table.append({
                'k': k,
                'CR': cr,
                'MSE': avg_mse
            })
        
        # Display results (same as before)
        print("\n=========================================")
        print("📊 ΠOΣΟΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ΣΥΜΠΙΕΣΗΣ")
        print("=========================================")
        print(f"{'k':<5} | {'Λόγος Συμπίεσης (CR)':<25} | {'Μέσο Σφάλμα (MSE)':<20}")
        print("-" * 52)
        for res in results_table:
            print(f"{res['k']:<5} | {res['CR']:.2f} : 1.00{'':<18} | {res['MSE']:.2f}{'':<20}")
        
        # Visualization
        fig, axes = plt.subplots(1, len(RANKS_TO_TEST) + 1, figsize=(18, 5))
        original_img_norm = np.dstack((R_norm, G_norm, B_norm))
        axes[0].imshow(original_img_norm)
        axes[0].set_title(f"Original\n({original_shape[0]}x{original_shape[1]})")
        
        for i, k in enumerate(RANKS_TO_TEST):
            axes[i + 1].imshow(compressed_images[i])
            axes[i + 1].set_title(f"k = {k}")
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nΗ βελτιστοποιημένη διαδικασία συμπίεσης SVD ολοκληρώθηκε επιτυχώς.")
        
    except Exception as e:
        print(f"\nΔιαδικασία απέτυχε λόγω σφάλματος: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_compression_pipeline_optimized()