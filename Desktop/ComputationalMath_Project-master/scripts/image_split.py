import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from metrics_calculation import matrix_multiply, matrix_transpose

IMAGE_PATH = 'mario_clean.png' 

def load_and_split_image(image_path):
    try:
        # Φόρτωση εικόνας
        img = Image.open(image_path).convert('RGB')
        # Μετατροπή σε πίνακα NumPy
        img_np = np.array(img, dtype=np.float64)
        
        # Εξαγωγή των καναλιών RGB
        A_R = img_np[:, :, 0]
        A_G = img_np[:, :, 1]
        A_B = img_np[:, :, 2]
        
        print(f"Εικόνα φορτώθηκε. Διαστάσεις: {img_np.shape}")
        
        # Επιστροφή μη ομαλοποιημένων καναλιών και το σχήμα
        return A_R, A_G, A_B, img_np.shape

    # Εξαγωγή του σφάλματος 
    except FileNotFoundError:
        print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε η εικόνα στη διαδρομή: {image_path}")
        raise 
    except Exception as e:
        print(f"ΣΦΑΛΜΑ κατά τη φόρτωση της εικόνας: {e}")
        raise

def normalize_and_prepare_w(A_channel):
    
    # Ομαλοποίηση (Normalization): Μετατροπή των τιμών [0-255] σε [0-1]
    A_norm = A_channel / 255.0
    
    # Υπολογισμός W = A^T A
    W = matrix_multiply(matrix_transpose(A_norm), A_norm)
    
    return A_norm, W