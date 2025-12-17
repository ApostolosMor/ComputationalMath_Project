import numpy as np

def calculate_mse(original_channel, compressed_channel):
    """
    Υπολογίζει το Μέσο Τετραγωνικό Σφάλμα (MSE) μεταξύ δύο πινάκων (καναλιών).
    
    Τύπος: MSE = (1 / (M*N)) * sum((A - A_k)^2)
    
    Σημείωση: Ο original_channel είναι ο ΜΗ ομαλοποιημένος πίνακας (0-255), 
    αλλά το compressed_channel (A_k) είναι ήδη κλιμακωμένο πίσω στο 0-255 
    μέσα στο compression.py (np.uint8). Πρέπει να εξασφαλίσουμε ότι οι πράξεις 
    γίνονται σε τύπο δεδομένων floating point (π.χ. float64).
    """
    
    # Μετατροπή των πινάκων σε float64 για ακριβείς υπολογισμούς
    A_orig = original_channel.astype(np.float64)
    A_comp = compressed_channel.astype(np.float64)
    
    # Υπολογισμός της διαφοράς
    diff = A_orig - A_comp
    
    # Υπολογισμός του τετραγώνου της διαφοράς
    squared_diff = np.square(diff)
    
    # Υπολογισμός του μέσου όρου
    mse = np.mean(squared_diff)
    
    return mse

def calculate_compression_ratio(M, N, k):
    """
    Υπολογίζει τον Λόγο Συμπίεσης (CR) για έναν πίνακα M x N με βαθμό προσέγγισης k.
    
    CR = (M * N) / (k * (M + N + 1))
    """
    
    # Αριθμός στοιχείων αρχικής εικόνας (για ένα κανάλι)
    original_size = M * N
    
    # Αριθμός στοιχείων συμπιεσμένης εικόνας (για ένα κανάλι)
    # Χρειάζονται k στήλες του U (M*k), k ιδιάζουσες τιμές (k), k στήλες του V (N*k)
    # Συνολικά: M*k + k + N*k = k * (M + N + 1)
    compressed_size = k * (M + N + 1)
    
    if compressed_size == 0:
        return np.inf # Αδιανόητο αλλά για λόγους ασφαλείας
        
    compression_ratio = original_size / compressed_size
    
    return compression_ratio

# --- ΔΟΚΙΜΑΣΤΙΚΟ ΜΕΡΟΣ ---
if __name__ == '__main__':
    print("Το evaluation.py περιέχει συναρτήσεις για MSE και CR.")
    
    # Δοκιμή CR
    M_test, N_test = 500, 300
    k_test = 10
    cr = calculate_compression_ratio(M_test, N_test, k_test)
    print(f"\nCR για 500x300 και k=10: {cr:.2f}") # Περίπου 14.97