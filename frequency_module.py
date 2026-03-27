import cv2
import numpy as np
import matplotlib.pyplot as plt
def compute_fft(image):
    # 1. Resmi frekans uzayına taşı 
    dft = np.fft.fft2(image)
    
    # 2. Alçak frekansları (merkezi) ortaya çek 
    dft_shift = np.fft.fftshift(dft)
    
    # 3. İnsan gözünün görebileceği bir spektrum (görsel) hazırla 
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    return dft_shift, magnitude_spectrum
def apply_high_pass_filter(dft_shift, radius=30):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2  # Merkezin koordinatları
    
    # Merkeze bir "kara delik" koy (Düşük frekansları sil) 
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
def apply_low_pass_filter(dft_shift, radius=30):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2
    
    # Merkezin dışında kalan her şeyi silen bir maske oluştur 
    mask = np.zeros((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask[mask_area] = 1
    
    # Filtreyi uygula
    filtered_dft = dft_shift * mask
    return filtered_dft
def calculate_spectral_energy(dft_shift):
    # Toplam spektral enerjiyi hesapla
    magnitude = np.abs(dft_shift)
    total_energy = np.sum(magnitude**2)
    
    
    return total_energy
if __name__ == "__main__":
    # 1. Test resmini oku (Grayscale olması FFT için şart) 
    # 'test_images/test_face.jpg' dosyasının klasörde olduğundan emin ol
    img = cv2.imread('test_images/test_face.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        # 2. Motoru çalıştır: FFT 
        dft_shift, spec = compute_fft(img)
        
        # 3. Yaşlandırma testi: High-Pass 
        aged_dft = apply_high_pass_filter(dft_shift, radius=30)
        aged_img = reconstruct_image(aged_dft)
        
        # 4. Gençleştirme testi: Low-Pass
        young_dft = apply_low_pass_filter(dft_shift, radius=30)
        young_img = reconstruct_image(young_dft)
        
        # 5. Görselleştirme 
        plt.figure(figsize=(12, 6))
        plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Orijinal')
        plt.subplot(132), plt.imshow(aged_img, cmap='gray'), plt.title('Aged (High-Pass)')
        plt.subplot(133), plt.imshow(young_img, cmap='gray'), plt.title('De-Aged (Low-Pass)')
        plt.show()
    else:
        print("Hata: 'test_images/test_face.jpg' bulunamadı!")