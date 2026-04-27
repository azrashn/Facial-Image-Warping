import base64
import cv2
import numpy as np


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return float(max(min_value, min(max_value, value)))


def normalize_strength(intensity: float) -> float:
    """
    Accepts both 0-1 and 0-100 intensity values.
    Frontend sends 0-100, Swagger may send 0-1.
    """
    intensity = float(intensity)
    if intensity > 1.0:
        intensity = intensity / 100.0
    return clamp(intensity)


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
    Frequency-based aging:
    - preserves original colors
    - enhances fine high-frequency texture/wrinkle-like details
    - slightly increases contrast without creating muddy artifacts
    """
    if image is None:
        raise ValueError("Input image is None.")

    intensity = float(np.clip(intensity, 0.0, 1.0))

    # Work on luminance only, so colors are preserved
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # High-frequency details from luminance channel
    radius = int(14 + intensity * 40)
    high_pass = apply_frequency_filter(l, radius=radius, mode="high")
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    # Remove broad muddy texture, keep only fine detail
    detail_blur = cv2.GaussianBlur(high_pass, (0, 0), 1.0)
    detail = high_pass.astype(np.float32) - detail_blur.astype(np.float32)

    # Add controlled wrinkle/detail texture to luminance
    l_float = l.astype(np.float32)
    detail_strength = 1.15 + 1.25 * intensity
    aged_l = l_float + detail * detail_strength

    # Mild contrast and slight darkening for aging effect
    contrast = 1.02 + 0.10 * intensity
    darkness = 2.0 + 5.0 * intensity
    aged_l = aged_l * contrast - darkness

    # mikro wrinkle noise
    noise = np.random.normal(0, 5 * intensity, l.shape).astype(np.float32)
    aged_l = aged_l.astype(np.float32) + noise

    aged_l = np.clip(aged_l, 0, 255).astype(np.uint8)

    # Merge back with original color channels
    aged_lab = cv2.merge([aged_l, a, b])
    aged_color = cv2.cvtColor(aged_lab, cv2.COLOR_LAB2BGR)

    # Subtle sharpening, not too aggressive
    blurred = cv2.GaussianBlur(aged_color, (0, 0), 1.0)
    sharpened = cv2.addWeighted(
    aged_color,
    1.18 + 0.18 * intensity,
    blurred,
    -0.18 - 0.18 * intensity,
    0,
)

    # Final blend keeps result natural and colored
    blend_ratio = 0.35 + 0.30 * intensity
    result = cv2.addWeighted(image, 1.0 - blend_ratio, sharpened, blend_ratio, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_deaging_filter(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Frequency-based de-aging:
    - reduces high-frequency skin texture
    - smooths skin without fully blurring eyes/lips/edges
    - preserves color and facial structure
    """
    if image is None:
        raise ValueError("Input image is None.")

    intensity = float(np.clip(intensity, 0.0, 1.0))

    # Edge-preserving smoothing
    d = int(7 + 8 * intensity)
    if d % 2 == 0:
        d += 1

    sigma_color = int(45 + 90 * intensity)
    sigma_space = int(45 + 90 * intensity)

    smooth = image.copy()
    passes = 1 + int(2 * intensity)

    for _ in range(passes):
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)

    # Low-pass frequency smoothing per color channel
    rows, cols = image.shape[:2]
    min_dim = min(rows, cols)
    lp_radius = int(min_dim * (0.10 + 0.10 * intensity))
    lp_mask = create_circular_mask((rows, cols), lp_radius, high_pass=False)

    freq_smooth = np.zeros_like(image, dtype=np.float64)

    for ch in range(3):
        channel = image[:, :, ch].astype(np.float64)
        fft_ch = np.fft.fftshift(np.fft.fft2(channel))
        fft_ch *= lp_mask
        restored = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_ch)))
        freq_smooth[:, :, ch] = restored

    freq_smooth = np.clip(freq_smooth, 0, 255).astype(np.uint8)

    # Combine bilateral smoothing + frequency low-pass
    smooth_mix = cv2.addWeighted(
        smooth,
        0.65,
        freq_smooth,
        0.35,
        0,
    )

    # Preserve strong edges from original image
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 1.5)
    edge_mask = np.clip(edge_mask[..., None], 0.0, 1.0)

    # Where edges exist, keep more original image
    preserved = (
        smooth_mix.astype(np.float32) * (1.0 - edge_mask * 0.75)
        + image.astype(np.float32) * (edge_mask * 0.75)
    ).astype(np.uint8)

    # Slight brightness lift for youthful effect
    lab = cv2.cvtColor(preserved, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + (2 + 6 * intensity), 0, 255)
    preserved = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Final natural blend
    blend_ratio = 0.45 + 0.40 * intensity
    result = cv2.addWeighted(image, 1.0 - blend_ratio, preserved, blend_ratio, 0)

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