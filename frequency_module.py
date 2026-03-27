import cv2
import numpy as np
import matplotlib.pyplot as plt
def compute_fft(image):
    # 1. Resmi frekans uzayına taşı [cite: 298]
    dft = np.fft.fft2(image)
    
    # 2. Alçak frekansları (merkezi) ortaya çek [cite: 298]
    dft_shift = np.fft.fftshift(dft)
    
    # 3. İnsan gözünün görebileceği bir spektrum (görsel) hazırla [cite: 302]
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    return dft_shift, magnitude_spectrum
def apply_high_pass_filter(dft_shift, radius=30):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2  # Merkezin koordinatları
    
    # Merkeze bir "kara delik" koy (Düşük frekansları sil) [cite: 300]
    mask = np.ones((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask[mask_area] = 0
    
    # Filtreyi uygula
    filtered_dft = dft_shift * mask
    return filtered_dft
def reconstruct_image(filtered_dft):
    # Frekansları eski yerine koy ve tersine çevir 
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back