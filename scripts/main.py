import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Προσθέτουμε τον φάκελο 'scripts' στο PATH για να βρει τα modules (απαραίτητο αν τρέχουμε από διαφορετικό φάκελο)
if 'scripts' not in sys.path:
    # Προσθέτουμε τη διαδρομή του φακέλου 'scripts'
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# --- ΕΙΣΑΓΩΓΗ ΣΥΝΑΡΤΗΣΕΩΝ (IMPORTS) ---
# Συναρτήσεις από τα άλλα scripts
from image_split import load_and_split_image, normalize_and_prepare_w, IMAGE_PATH
from svd_core import calculate_eigens, calculate_svd_matrices
from compression import reconstruct_channel, merge_and_save_image
from evaluation import calculate_mse, calculate_compression_ratio
from metrics_calculation import matrix_multiply,matrix_transpose,matrix_scalar_multiply

# --- 1. Ορισμός Σταθερών Εκτέλεσης ---
# Οι βαθμίδες προσέγγισης k που θα χρησιμοποιήσουμε για τη συμπίεση
RANKS_TO_TEST = [5, 20, 50, 100]

def process_channel(A_channel, channel_name):
    """
    Εκτελεί τα Βήματα 2, 3, και 4 για ένα συγκεκριμένο κανάλι χρώματος.
    Υπολογίζει U, S, V και επιστρέφει το ομαλοποιημένο κανάλι.
    
    Επιστρέφει: U, S_vector, V, A_norm
    """
    print(f"\n--- Επεξεργασία {channel_name} Καναλιού (Βήματα 2, 3, 4) ---")
    
    # Βήμα 2: Ομαλοποίηση και Υπολογισμός W = A^T A
    A_norm, W = normalize_and_prepare_w(A_channel)
    
    # Βήμα 3: Υπολογισμός Ιδιοτιμών/Ιδιοδιανυσμάτων
    # Χρησιμοποιούμε την np.linalg.eigh (επιτρεπόμενη έτοιμη συνάρτηση)
    lambdas, V_full = calculate_eigens(W)
    
    # Βήμα 4: Υπολογισμός U, Sigma, V (αυτο-υλοποίηση του U)
    U, S_vector, V = calculate_svd_matrices(A_norm, lambdas, V_full)
    
    # Έλεγχος: Εμφάνιση του rank και της μεγαλύτερης ιδιάζουσας τιμής
    rank = len(S_vector)
    print(f"  Διαστάσεις {channel_name}: {A_norm.shape}")
    print(f"  Rank (Αριθμός μη μηδενικών σ): {rank}")
    print(f"  Μεγαλύτερη σ1: {S_vector[0]:.4f}")
    
    return U, S_vector, V, A_norm

def run_compression_pipeline():
    """
    Κεντρική λειτουργία που συνδέει όλα τα βήματα της SVD συμπίεσης.
    """
    
    print("=========================================")
    print(f"Ξεκινά η SVD Συμπίεση για εικόνα: {IMAGE_PATH}")
    print("=========================================")
    
    try:
        # --- Α) Βήμα 1: Φόρτωση και Διαχωρισμός Εικόνας ---
        # R_channel, G_channel, B_channel είναι ΜΗ ομαλοποιημένα (0-255)
        R_channel, G_channel, B_channel, original_shape = load_and_split_image(IMAGE_PATH)

        # --- Β) Βήματα 2, 3, 4: Υπολογισμός SVD Matrices (U, S, V) για κάθε κανάλι ---
        # R_norm, G_norm, B_norm είναι τα ομαλοποιημένα κανάλια (0-1)
        U_R, S_R, V_R, R_norm = process_channel(R_channel, "Κόκκινο")
        U_G, S_G, V_G, G_norm = process_channel(G_channel, "Πράσινο")
        U_B, S_B, V_B, B_norm = process_channel(B_channel, "Μπλε")

        # --- Γ) Βήματα 5 & 6: Ανακατασκευή, Αποθήκευση και Αξιολόγηση ---
        print("\n--- Βήματα 5, 6 & Αξιολόγηση ---")
        
        M, N, _ = original_shape
        results_table = [] # Για αποθήκευση αποτελεσμάτων
        compressed_images = []
        
        for k in RANKS_TO_TEST:
            print(f"Ανακατασκευή και Αξιολόγηση για k = {k}...")
            
            # Βήμα 5: Ανακατασκευή κάθε καναλιού (Επιστρέφει UNINT8, 0-255)
            R_k = reconstruct_channel(U_R, S_R, V_R, k)
            G_k = reconstruct_channel(U_G, S_G, V_G, k)
            B_k = reconstruct_channel(U_B, S_B, V_B, k)
            
            # Βήμα 6: Επανένωση και Αποθήκευση
            compressed_image_np = merge_and_save_image(R_k, G_k, B_k, k, original_shape)
            compressed_images.append(compressed_image_np)

            # --- Υπολογισμός Μετρικών (evaluation.py) ---
            
            # 1. Μέσο Τετραγωνικό Σφάλμα (MSE)
            # Υπολογίζεται για κάθε κανάλι και λαμβάνεται ο μέσος όρος
            mse_r = calculate_mse(R_channel, R_k)
            mse_g = calculate_mse(G_channel, G_k)
            mse_b = calculate_mse(B_channel, B_k)
            avg_mse = (mse_r + mse_g + mse_b) / 3.0
            
            # 2. Λόγος Συμπίεσης (CR)
            cr = calculate_compression_ratio(M, N, k)
            
            # Αποθήκευση αποτελεσμάτων
            results_table.append({
                'k': k, 
                'CR': cr, 
                'MSE': avg_mse
            })

        # --- Δ) Παρουσίαση Αποτελεσμάτων Πίνακα ---
        print("\n=========================================")
        print("📊 ΠOΣΟΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ΣΥΜΠΙΕΣΗΣ")
        print("=========================================")
        print(f"{'k':<5} | {'Λόγος Συμπίεσης (CR)':<25} | {'Μέσο Σφάλμα (MSE)':<20}")
        print("-" * 52)
        for res in results_table:
            # Εμφάνιση Λόγου Συμπίεσης ως CR : 1.00
            print(f"{res['k']:<5} | {res['CR']:.2f} : 1.00{'':<18} | {res['MSE']:.2f}{'':<20}")
            
        # --- Ε) Οπτικοποίηση Αποτελεσμάτων ---
        fig, axes = plt.subplots(1, len(RANKS_TO_TEST) + 1, figsize=(18, 5))
        
        # Αρχική Εικόνα (χρησιμοποιούμε τα ομαλοποιημένα κανάλια για εμφάνιση)
        original_img_norm = np.dstack((R_norm, G_norm, B_norm))
        axes[0].imshow(original_img_norm)
        axes[0].set_title(f"Original\n({original_shape[0]}x{original_shape[1]})")
        
        # Συμπιεσμένες Εικόνες (χρησιμοποιούμε τα UINT8)
        for i, k in enumerate(RANKS_TO_TEST):
            axes[i + 1].imshow(compressed_images[i])
            axes[i + 1].set_title(f"k = {k}")
            
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Ολοκλήρωση
        print("\nΗ διαδικασία συμπίεσης SVD ολοκληρώθηκε επιτυχώς.")
        print("Οι ανακατασκευασμένες εικόνες και τα αποτελέσματα είναι έτοιμα.")

    except Exception as e:
        print(f"\nΔιαδικασία απέτυχε λόγω σφάλματος: {e}")
        print("Βεβαιωθείτε ότι όλα τα scripts και η εικόνα βρίσκονται στις σωστές διαδρομές.")
        # Έξοδος για αποφυγή προβλημάτων
        exit()

if __name__ == '__main__':
    # Βεβαιωθείτε ότι η εικόνα (mario_clean.png) βρίσκεται στον βασικό φάκελο,
    # ένα επίπεδο πάνω από τον φάκελο 'scripts'.
    
    run_compression_pipeline()