import base64
import cv2
import numpy as np


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return float(max(min_value, min(max_value, value)))


def normalize_strength(intensity: float) -> float:
    return clamp(float(intensity) / 100.0)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert input image to grayscale if needed.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 2:
        return image

    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError("Unsupported image shape.")


def compute_fft(image: np.ndarray):
    """
    Compute 2D FFT and shifted frequency representation.
    """
    gray = ensure_grayscale(image)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    return gray, fft, fft_shifted


def compute_magnitude_spectrum(fft_shifted: np.ndarray) -> np.ndarray:
    """
    Create log-scaled magnitude spectrum image.
    """
    magnitude = np.abs(fft_shifted)
    spectrum = np.log1p(magnitude)
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return spectrum.astype(np.uint8)


def create_circular_mask(shape, radius: int, high_pass: bool = False) -> np.ndarray:
    """
    Create circular low-pass or high-pass mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    distance_sq = (x - ccol) ** 2 + (y - crow) ** 2
    region = distance_sq <= radius ** 2

    if high_pass:
        mask = np.ones((rows, cols), dtype=np.float32)
        mask[region] = 0.0
    else:
        mask = np.zeros((rows, cols), dtype=np.float32)
        mask[region] = 1.0

    return mask


def reconstruct_image(filtered_fft_shifted: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from filtered shifted FFT.
    """
    fft_ishift = np.fft.ifftshift(filtered_fft_shifted)
    image_back = np.fft.ifft2(fft_ishift)
    image_back = np.abs(image_back)
    image_back = cv2.normalize(image_back, None, 0, 255, cv2.NORM_MINMAX)
    return image_back.astype(np.uint8)


def apply_frequency_filter(image: np.ndarray, radius: int, mode: str = "low") -> np.ndarray:
    """
    Apply low-pass or high-pass filter in frequency domain.
    """
    gray, _, fft_shifted = compute_fft(image)

    if mode == "low":
        mask = create_circular_mask(gray.shape, radius, high_pass=False)
    elif mode == "high":
        mask = create_circular_mask(gray.shape, radius, high_pass=True)
    else:
        raise ValueError("Mode must be 'low' or 'high'.")

    filtered_fft = fft_shifted * mask
    result = reconstruct_image(filtered_fft)
    return result


def apply_aging_filter(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simulate aging using controlled texture enhancement.
    Keeps the result natural-looking and in color.
    """
    if image is None:
        raise ValueError("Input image is None.")

    intensity = float(np.clip(intensity, 0.0, 1.0))

    gray = ensure_grayscale(image)

    radius = int(16 + intensity * 18)
    high_pass = apply_frequency_filter(image, radius, mode="high")
    detail = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    detail_strength = 0.22 + 0.28 * intensity
    enhanced_gray = cv2.addWeighted(gray, 1.0, detail, detail_strength, 0)

    alpha = 1.08 + 0.15 * intensity
    beta = -6 - int(10 * intensity)
    enhanced_gray = cv2.convertScaleAbs(enhanced_gray, alpha=alpha, beta=beta)

    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    blend_ratio = 0.30 + 0.25 * intensity
    result = cv2.addWeighted(image, 1.0 - blend_ratio, enhanced_bgr, blend_ratio, 0)

    sharpen_strength = 0.18 + 0.12 * intensity
    blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_deaging_filter(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simulate de-aging with aggressive skin smoothing while preserving
    edges and structure.  At 100 % intensity the effect should be
    unmistakably visible (youthful, porcelain-like skin).
    """
    if image is None:
        raise ValueError("Input image is None.")

    intensity = float(np.clip(intensity, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 1) Multi-pass bilateral smoothing (main skin smoother)
    # ------------------------------------------------------------------
    #   d: filter diameter – larger = smoother
    #   sigma_color / sigma_space: higher = more aggressive averaging
    d = int(9 + 8 * intensity)                         # 9 → 17
    if d % 2 == 0:
        d += 1                                         # must be odd
    sigma_color = int(50 + 100 * intensity)            # 50 → 150
    sigma_space = int(50 + 100 * intensity)            # 50 → 150

    smooth = image.copy()
    passes = 1 + int(2 * intensity)                    # 1 → 3 passes
    for _ in range(passes):
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)

    # ------------------------------------------------------------------
    # 2) Frequency-domain low-pass to soften fine wrinkle texture
    # ------------------------------------------------------------------
    gray = ensure_grayscale(image)
    rows, cols = gray.shape
    min_dim = min(rows, cols)
    lp_radius = int(min_dim * (0.08 + 0.12 * intensity))  # 8–20 % of image
    lp_mask = create_circular_mask(gray.shape, lp_radius, high_pass=False)

    # Apply LP per channel to keep colour
    freq_smooth = np.zeros_like(image, dtype=np.float64)
    for ch in range(image.shape[2]):
        fft_ch = np.fft.fftshift(np.fft.fft2(image[:, :, ch].astype(np.float64)))
        fft_ch *= lp_mask
        freq_smooth[:, :, ch] = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_ch)))
    freq_smooth = np.clip(freq_smooth, 0, 255).astype(np.uint8)

    # Blend bilateral + frequency smooth
    freq_weight = 0.15 + 0.25 * intensity              # 15–40 %
    smooth = cv2.addWeighted(smooth, 1.0 - freq_weight,
                             freq_smooth, freq_weight, 0)

    # ------------------------------------------------------------------
    # 3) Edge preservation – keep structure from the original using a
    #    high-frequency residual (unsharp-mask style)
    # ------------------------------------------------------------------
    blur_for_edges = cv2.GaussianBlur(image, (0, 0), 3.0)
    edge_detail = cv2.subtract(image, blur_for_edges)
    edge_strength = 0.4 + 0.4 * intensity              # 0.4 → 0.8
    smooth = cv2.addWeighted(smooth, 1.0, edge_detail, edge_strength, 0)

    # ------------------------------------------------------------------
    # 4) Optional brightness / skin-tone lift in LAB space
    # ------------------------------------------------------------------
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_boost = 3.0 + 7.0 * intensity                    # +3 → +10
    lab[:, :, 0] = np.clip(lab[:, :, 0] + l_boost, 0, 255)
    smooth = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # 5) Final blend with original (never fully replace at low intensity)
    # ------------------------------------------------------------------
    blend_ratio = 0.40 + 0.55 * intensity              # 40–95 %
    result = cv2.addWeighted(image, 1.0 - blend_ratio,
                             smooth, blend_ratio, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_aging(image: np.ndarray, intensity: float) -> np.ndarray:
    strength = normalize_strength(intensity)
    return apply_aging_filter(image, intensity=strength)


def apply_deaging(image: np.ndarray, intensity: float) -> np.ndarray:
    strength = normalize_strength(intensity)
    return apply_deaging_filter(image, intensity=strength)


def apply_fft_filter(image: np.ndarray, intensity: float) -> tuple[np.ndarray, np.ndarray]:
    strength = normalize_strength(intensity)
    radius = int(8 + strength * 52)
    filtered = apply_frequency_filter(image, radius=radius, mode="high")
    spectrum = compute_magnitude_spectrum(compute_fft(image)[2])
    filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    return filtered_bgr, spectrum


def compute_energy_analysis(image: np.ndarray, radius: int = 30) -> dict:
    """
    Compute total, low-frequency, and high-frequency energy ratios.
    """
    gray, _, fft_shifted = compute_fft(image)

    magnitude = np.abs(fft_shifted)
    power_spectrum = magnitude ** 2

    low_mask = create_circular_mask(gray.shape, radius, high_pass=False)
    high_mask = create_circular_mask(gray.shape, radius, high_pass=True)

    total_energy = float(np.sum(power_spectrum))
    low_energy = float(np.sum(power_spectrum * low_mask))
    high_energy = float(np.sum(power_spectrum * high_mask))

    if total_energy == 0:
        low_ratio = 0.0
        high_ratio = 0.0
    else:
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy

    return {
        "total_energy": total_energy,
        "low_frequency_energy": low_energy,
        "high_frequency_energy": high_energy,
        "low_frequency_ratio": low_ratio,
        "high_frequency_ratio": high_ratio,
        "radius": radius,
    }


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode image as PNG base64 string.
    """
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Image could not be encoded to PNG.")

    return base64.b64encode(buffer).decode("utf-8")