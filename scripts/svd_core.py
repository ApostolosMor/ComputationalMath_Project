import numpy as np
from metrics_calculation import matrix_multiply, matrix_transpose

def calculate_eigens(W_matrix):
    # Βήμα 3: Υπολογισμός ιδιοτιμών (lambdas) και ιδιοδιανυσμάτων (V)
    # Χρησιμοποιούμε την np.linalg.eigh όπως ρητά επιτρέπεται από τις οδηγίες
    lambdas, V = np.linalg.eigh(W_matrix)
    
    # Ταξινόμηση σε φθίνουσα σειρά (από τη μεγαλύτερη στη μικρότερη)
    #Πίνακας ιδιοτιμών
    lambdas = lambdas[::-1]
    #Πίνακας ιδιοδιανυσμάτων
    V = V[:, ::-1] 
    
    return lambdas, V

def calculate_svd_matrices(A_norm, lambdas, V_full):
    # Βήμα 4α: Υπολογισμός ιδιάζουσων τιμών σ_i = sqrt(λ_i)- χρησιμοποιώ το np.maximu() για να αποφύγω το σφάλμα στρογγυλοποίησης
    sigmas = np.sqrt(np.maximum(lambdas, 0))
    
    # Επιλογή μη μηδενικών τιμών (προσδιορισμός rank) - επιλέγω μόνο όσες δεν είναι  ή πολύ μικρές 
    # Βαθμός πίνακα = πλήθος από ανεξάρτητες πληροφορίες που περιέχονται στο κανάλι της εικόνας
    non_zero_sigmas_idx = sigmas > 1e-10 
    S_vector = sigmas[non_zero_sigmas_idx]
    
    # Ο πίνακας V περιέχει ως στήλες τα διανύσματα v_i (Βήμα 4β)
    V_matrix = V_full[:, non_zero_sigmas_idx]
    
    # Βήμα 4γ: Υπολογισμός του πίνακα U κατά στήλες: u_i = (1/σ_i) * (A * v_i)
    M, N = A_norm.shape #Διαστάσεις του πίνακα
    U_matrix = np.zeros((M, len(S_vector))) #Δημιουργεία κενού πίνακα με μηδενικά
    
    for i in range(len(S_vector)): #Για κάθε στήλη  u_i ξεχωριστά 
        # Επιλέγουμε την i-οστή ιδιάζουσα τιμή και το αντίστοιχο i-οστό ιδιοδιάνυσμα του V 
        sigma_i = S_vector[i]
        # Επιλογή της i-οστής στήλης του V
        v_i = V_matrix[:, i]
        
        # Μετατροπή του διανύσματος v_i σε πίνακα (N x 1) για τον custom πολλαπλασιασμό
        v_i_2d = v_i.reshape(-1, 1)
        
        # Χρήση της custom matrix_multiply
        A_times_v_i_2d = matrix_multiply(A_norm, v_i_2d)
        
        # Επαναφορά σε μορφή διανύσματος (flatten) και υπολογισμός του u_i
        u_i = (1.0 / sigma_i) * A_times_v_i_2d.flatten()
        
        # Τοποθέτηση του u_i στην i-οστή στήλη του U
        U_matrix[:, i] = u_i
        
    return U_matrix, S_vector, V_matrix