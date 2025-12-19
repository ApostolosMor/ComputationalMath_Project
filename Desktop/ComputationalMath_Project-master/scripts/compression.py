import numpy as np
from PIL import Image

def reconstruct_channel(U, S_vector, V, k):
    """
    Εφαρμόζει το Βήμα 5: Ανακατασκευή του πίνακα A_k για δοσμένο βαθμό k.
    
    A_k = sum_{i=1}^{k} (sigma_i * u_i * v_i^T)
    
    U: Ο πίνακας U (M x rank)
    S_vector: Οι ιδιάζουσες τιμές (rank)
    V: Ο πίνακας V (N x rank)
    k: Ο βαθμός προσέγγισης (αριθμός ιδιάζουσων τιμών που χρησιμοποιούνται).
    
    Επιστρέφει: Ο πίνακας A_k (M x N)
    """
    
    # Επειδή η SVD έχει ταξινομημένες τις τιμές, χρησιμοποιούμε μόνο τις πρώτες k
    
    # 1. Επιλογή των k κορυφαίων συνιστωσών
    U_k = U[:, :k]
    S_k = S_vector[:k]
    V_k = V[:, :k]
    
    # 2. Ανακατασκευή του πίνακα A_k
    # Ο ανακατασκευασμένος πίνακας A_k είναι: U_k @ np.diag(S_k) @ V_k.T
    
    # np.diag(S_k) δημιουργεί τον διαγώνιο πίνακα Σ_k
    Sigma_k = np.diag(S_k)
    
    # Πολλαπλασιασμός: U_k (M x k) @ Sigma_k (k x k) -> (M x k)
    # Αποτέλεσμα @ V_k.T (k x N) -> (M x N)
    A_k = U_k @ Sigma_k @ V_k.T
    #Temp = matrix_multiply(U_k, Sigma_k) 
    # Πολλαπλασιασμός: A_k = Temp * V_k.T
    #A_k = matrix_multiply(Temp, V_k.T)
    
    # 3. Απο-ομαλοποίηση (Denormalization) και Κλιμάκωση
    # Οι τιμές A_k είναι στο [0, 1]. Τις ξαναφέρνουμε στο [0, 255].
    A_k_denorm = A_k * 255.0
    
    # Εξασφαλίζουμε ότι οι τιμές είναι ακέραιες (uint8) και εντός του εύρους [0, 255]
    A_k_clipped = np.clip(A_k_denorm, 0, 255).astype(np.uint8)
    
    return A_k_clipped

def merge_and_save_image(R_k, G_k, B_k, k, original_shape):
    """
    Εφαρμόζει το Βήμα 6: Επανένωση των τριών καναλιών και αποθήκευση της εικόνας.
    """
    
    # Στοίβαξη των 3 καναλιών σε ένα 3D array
    # np.dstack στοιβάζει τα arrays κατά τη τρίτη διάσταση (M x N x 3)
    compressed_image_np = np.dstack((R_k, G_k, B_k))
    
    # Μετατροπή του NumPy array σε αντικείμενο εικόνας PIL
    compressed_image = Image.fromarray(compressed_image_np, 'RGB')
    
    # Αποθήκευση εικόνας
    output_filename = f'compressed_k{k}.png'
    compressed_image.save(output_filename)
    
    print(f"Εικόνα k={k} αποθηκεύτηκε ως: {output_filename}")
    
    return compressed_image_np


# --- ΔΟΚΙΜΑΣΤΙΚΟ ΜΕΡΟΣ ---
if __name__ == '__main__':
    print("Το compression.py περιέχει συναρτήσεις για Βήματα 5 & 6.")
    print("Χρειάζεται να εισαχθεί σε κεντρικό script.")