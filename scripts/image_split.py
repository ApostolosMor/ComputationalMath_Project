import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from metrics_calculation import matrix_multiply, matrix_transpose


# --- 1. Ορισμός Σταθερών και Φόρτωση Εικόνας ---
IMAGE_PATH = 'mario_clean.png' 

def load_and_split_image(image_path):
    """
    Φορτώνει μια έγχρωμη εικόνα, την μετατρέπει σε πίνακα NumPy, 
    και εξάγει τα κανάλια RGB (μη ομαλοποιημένα).
    """
    try:
        # Φόρτωση εικόνας (Pillow)
        img = Image.open(image_path).convert('RGB') # Προσθήκη convert('RGB') για σταθερότητα
        
        # Μετατροπή σε πίνακα NumPy
        img_np = np.array(img, dtype=np.float64)
        
        # Εξαγωγή των καναλιών RGB (Βήμα 1)
        A_R = img_np[:, :, 0]
        A_G = img_np[:, :, 1]
        A_B = img_np[:, :, 2]
        
        print(f"Εικόνα φορτώθηκε. Διαστάσεις: {img_np.shape}")
        
        # Επιστρέφουμε τα μη ομαλοποιημένα κανάλια και το σχήμα
        return A_R, A_G, A_B, img_np.shape
        
    except FileNotFoundError:
        print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε η εικόνα στη διαδρομή: {image_path}")
        raise # Εξαγωγή του σφάλματος για να το πιάσει το καλούν script
    except Exception as e:
        print(f"ΣΦΑΛΜΑ κατά τη φόρτωση της εικόνας: {e}")
        raise

def normalize_and_prepare_w(A_channel):
    """
    Ομαλοποιεί ένα κανάλι (A_channel) και υπολογίζει τον πίνακα W = A^T A (Βήμα 2).
    
    Επιστρέφει: Το ομαλοποιημένο κανάλι και τον πίνακα W.
    """
    
    # Ομαλοποίηση (Normalization): Μετατροπή των τιμών [0-255] σε [0-1]
    A_norm = A_channel / 255.0
    
    # Βήμα 2: Υπολογισμός W = A^T A
    W = matrix_multiply(matrix_transpose(A_norm), A_norm)
    
    return A_norm, W


# # --- 2. Κύριο Μέρος Προγράμματος (Δοκιμαστικό) ---
# if __name__ == '__main__':
    
#     try:
#         # 1. Φόρτωση και διαχωρισμός
#         R_channel, G_channel, B_channel, original_shape = load_and_split_image(IMAGE_PATH)
        
#         # 2. Ομαλοποίηση και Υπολογισμός W για το Κόκκινο κανάλι
#         R_norm, W_R = normalize_and_prepare_w(R_channel)
        
#         print("\nΤα κανάλια RGB έχουν εξαχθεί, ομαλοποιηθεί και υπολογιστεί ο W.")
#         print(f"Διαστάσεις W_R: {W_R.shape}")
        
#         # Εμφάνιση αρχικής εικόνας για έλεγχο
#         # Χρησιμοποιούμε τα ομαλοποιημένα κανάλια για εμφάνιση (τιμές 0-1)
#         plt.imshow(np.dstack((R_norm, G_channel/255.0, B_channel/255.0)))
#         plt.title('Αρχική Εικόνα (mario_clean.png)')
#         plt.axis('off')
#         plt.show()
        
#     except Exception as e:
#         print(f"\nΑδυναμία ολοκλήρωσης του image_split.py.")