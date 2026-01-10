import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Remove unused imports to metrics_calculation since we compute W directly

IMAGE_PATH = 'mario_clean.png'

def load_and_split_image(image_path):
    """Same as before"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.float64)
        A_R = img_np[:, :, 0]
        A_G = img_np[:, :, 1]
        A_B = img_np[:, :, 2]
        print(f"Εικόνα φορτώθηκε. Διαστάσεις: {img_np.shape}")
        return A_R, A_G, A_B, img_np.shape
    except FileNotFoundError:
        print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε η εικόνα στη διαδρομή: {image_path}")
        raise
    except Exception as e:
        print(f"ΣΦΑΛΜΑ κατά τη φόρτωση της εικόνας: {e}")
        raise

def normalize_and_prepare_w(A_channel):
    """
    Optimized: Only normalize here, W calculation moved to main
    to use optimized AᵀA computation
    """
    A_norm = A_channel / 255.0
    # W will be computed separately using optimized function
    return A_norm  # Return only normalized matrix