import numpy as np

def calculate_mse(original_channel, compressed_channel):

    # Μετατροπή των πινάκων σε float64 για ακριβείς υπολογισμούς
    A_orig = original_channel.astype(np.float64)
    A_comp = compressed_channel.astype(np.float64)

    # Αριθμός γραμμών του πίνακα
    M = len(original_channel)
    # Αριθμός στηλών του πίνακα
    N = len(original_channel[0])

    total = 0.0

    for i in range(M):
        for j in range(N):
            # Διαφορά αντίστοιχων στοιχείων
            diff = (A_orig[i,j]) - (A_comp[i,j])
            # Τετραγωνισμός της διαφοράς και πρόσθεση στο άθροισμα
            total += diff * diff


    # Υπολογισμός του Mean Squared Error
    mse = total / (M * N)
    return mse

def calculate_compression_ratio(M, N, k):
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
