import cv2
import numpy as np
from deepface import DeepFace

#  AI AGE

def predict_age(image):
    try:
        result = DeepFace.analyze(image, actions=['age'], enforce_detection=False)
        return int(result[0]['age'])
    except:
        return None

#  FFT (Rol 3'ten)

def compute_fft(image):
    dft = np.fft.fft2(image)
    return np.fft.fftshift(dft)


def apply_high_pass_filter(dft_shift, radius=30):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
    mask[mask_area] = 0

    return dft_shift * mask


def calculate_spectral_energy(dft_shift):
    magnitude = np.abs(dft_shift)
    return np.sum(magnitude ** 2)


#  FREQUENCY AGE

def estimate_frequency_age(dft_shift):
    energy = calculate_spectral_energy(dft_shift)
    return int(min(max(energy / 1e6, 0), 100))



#  MAIN 

def process(face_image):
    

    # 1. AI age 
    ai_age_original = predict_age(face_image)

    # 2. FFT için grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    dft_shift = compute_fft(gray)

    # 3. Frequency ages
    freq_age_original = estimate_frequency_age(dft_shift)

    aged_dft = apply_high_pass_filter(dft_shift)
    freq_age_transformed = estimate_frequency_age(aged_dft)

    # 4. Output (FR-43 + FR-29)
    return {
        "ages": {
            "original": {
                "ai_age": ai_age_original,
                "frequency_age": freq_age_original
            },
            "transformed": {
                "ai_age": None,
                "frequency_age": freq_age_transformed
            }
        },
        "comparison": {
            "age_difference_ai": None,
            "age_difference_frequency": (
                freq_age_transformed - freq_age_original
            )
        }
    }
