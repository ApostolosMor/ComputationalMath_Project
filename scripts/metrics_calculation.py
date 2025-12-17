import numpy as np

def matrix_multiply(A, B):
    """
    Υπολογίζει το γινόμενο C = A @ B χρησιμοποιώντας εμφωλευμένους βρόχους (loops).
    
    ΠΡΟΣΟΧΗ: Αυτή η μέθοδος είναι πολύ αργή για μεγάλους πίνακες (όπως κανάλια εικόνας).
    """
    
    # Έλεγχος συμβατότητας διαστάσεων
    # Ο Α πρέπει να είναι M x K και ο Β πρέπει να είναι K x N
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Οι διαστάσεις δεν είναι συμβατές για πολλαπλασιασμό: "
            f"A ({A.shape[0]}x{A.shape[1]}) και B ({B.shape[0]}x{B.shape[1]})."
        )
    
    M = A.shape[0]  # Αριθμός γραμμών του C
    K = A.shape[1]  # Κοινή διάσταση (εσωτερικός βρόχος)
    N = B.shape[1]  # Αριθμός στηλών του C
    
    # Δημιουργία του πίνακα αποτελέσματος C (M x N) με αρχικοποίηση στο μηδέν
    C = np.zeros((M, N), dtype=A.dtype)
    
    # Εμφωλευμένοι βρόχοι για τον υπολογισμό του C[i, j]
    for i in range(M):      # Διασχίζει τις γραμμές του Α
        for j in range(N):  # Διασχίζει τις στήλες του Β
            sum_val = 0
            for k in range(K):  # Διασχίζει την κοινή διάσταση (K)
                # Ορισμός του C[i, j] = sum_{k} A[i, k] * B[k, j]
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val
            
    return C

def matrix_transpose(A):
    """
    Υπολογίζει τον ανάστροφο πίνακα A^T.
    Μετατρέπει τις γραμμές σε στήλες χειροκίνητα.
    """
    M, N = A.shape
    # Ο ανάστροφος έχει διαστάσεις N x M
    A_T = np.zeros((N, M), dtype=np.float64)
    
    for i in range(M):
        for j in range(N):
            # Η θέση (i,j) γίνεται (j,i)
            A_T[j, i] = A[i, j]
            
    return A_T

