"""
fft_module.py — FFT-based Aging / De-aging & Magnitude Spectrum
================================================================
Production-grade frequency-domain image processing:
  - Aging:    High-pass filter to amplify edges/wrinkles/detail
  - De-aging: Gaussian low-pass filter for skin smoothing with edge preservation
  - Magnitude spectrum visualization  (log-scaled, normalized)

All operations work per-channel on BGR images and produce uint8 output.
"""

import cv2
import numpy as np
from typing import Tuple


# ============================================================
# 1.  Filter Mask Generators
# ============================================================
def _create_gaussian_lpf(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Creates a 2D Gaussian Low-Pass Filter mask in the frequency domain.

    H(u,v) = exp( -D(u,v)² / (2 * cutoff²) )

    where D(u,v) is the distance from the center of the spectrum.

    Args:
        shape:  (rows, cols) of the frequency domain.
        cutoff: Standard deviation of the Gaussian (in pixels).
                Larger → keeps more frequencies → less smoothing.

    Returns:
        (rows, cols) float64 mask in [0, 1].
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    d_sq = u * u + v * v  # D(u,v)²

    # Avoid division by zero
    sigma_sq = max(cutoff * cutoff, 1e-10)
    mask = np.exp(-d_sq / (2.0 * sigma_sq))

    return mask


def _create_gaussian_hpf(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Creates a 2D Gaussian High-Pass Filter mask.

    H_hp(u,v) = 1 - H_lp(u,v)

    Args:
        shape:  (rows, cols).
        cutoff: Gaussian sigma — smaller → more aggressive high-pass.

    Returns:
        (rows, cols) float64 mask in [0, 1].
    """
    return 1.0 - _create_gaussian_lpf(shape, cutoff)


def _create_butterworth_lpf(
    shape: Tuple[int, int], cutoff: float, order: int = 2
) -> np.ndarray:
    """
    Butterworth Low-Pass Filter — smoother roll-off than ideal filter.

    H(u,v) = 1 / (1 + (D(u,v) / cutoff)^(2n))

    Used for edge-preserving smoothing in de-aging.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    d = np.sqrt(u * u + v * v)

    cutoff = max(cutoff, 1e-10)
    mask = 1.0 / (1.0 + (d / cutoff) ** (2 * order))

    return mask


# ============================================================
# 2.  Per-Channel FFT Processing
# ============================================================
def _fft_filter_channel(
    channel: np.ndarray,
    mask: np.ndarray,
    blend_factor: float = 1.0,
) -> np.ndarray:
    """
    Applies a frequency-domain filter to a single grayscale channel.

    Pipeline:  channel → FFT2 → shift → multiply mask → unshift → IFFT2

    Args:
        channel:      (H, W) uint8 or float64.
        mask:         (H, W) frequency-domain filter mask.
        blend_factor: 0.0 = original, 1.0 = fully filtered.

    Returns:
        (H, W) uint8 filtered channel.
    """
    f = np.fft.fft2(channel.astype(np.float64))
    f_shift = np.fft.fftshift(f)

    # Apply filter
    f_filtered = f_shift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_filtered)
    result = np.real(np.fft.ifft2(f_ishift))

    # Blend with original
    original = channel.astype(np.float64)
    blended = original * (1.0 - blend_factor) + result * blend_factor

    return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# 3.  Aging (Wrinkle / Detail Enhancement)
# ============================================================
def _apply_aging(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    Aging effect via high-frequency amplification.

    Strategy:
      1. Extract high-frequency detail using a Gaussian HPF
      2. Add the amplified detail back to the original image
      3. Intensity controls the amplification factor

    This enhances edges, wrinkles, pores — simulating aged skin texture.

    Args:
        image:     BGR uint8.
        intensity: 0.0 – 1.0

    Returns:
        BGR uint8 aged image.
    """
    h, w = image.shape[:2]

    # Cutoff: larger face dimension / 8  — captures wrinkle-scale detail
    base_cutoff = max(h, w) / 8.0
    # Intensity scales how much high-freq is amplified
    # At intensity=0.5, moderate enhancement; at 1.0, strong crinkle effect
    amplification = 1.0 + intensity * 3.0  # range [1.0, 4.0]

    # Build high-pass mask
    hpf = _create_gaussian_hpf((h, w), cutoff=base_cutoff)

    # Create the enhancement mask:  1 + amplification * HPF
    # This keeps all frequencies but boosts high ones
    enhancement_mask = 1.0 + (amplification - 1.0) * hpf

    result = image.copy()
    channels = cv2.split(result)
    enhanced = []

    for ch in channels:
        f = np.fft.fft2(ch.astype(np.float64))
        f_shift = np.fft.fftshift(f)

        # Multiply spectrum by enhancement mask
        f_enhanced = f_shift * enhancement_mask

        f_ishift = np.fft.ifftshift(f_enhanced)
        ch_result = np.real(np.fft.ifft2(f_ishift))
        enhanced.append(np.clip(ch_result, 0, 255).astype(np.uint8))

    return cv2.merge(enhanced)


# ============================================================
# 4.  De-aging (Skin Smoothing with Edge Preservation)
# ============================================================
def _apply_deaging(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    De-aging effect via frequency-domain skin smoothing.

    Strategy:
      1. Apply Butterworth LPF to smooth skin texture (remove high-freq noise/pores)
      2. Preserve edges using a detail mask (difference of Gaussians approach)
      3. Blend smoothed result with original based on intensity

    Butterworth chosen over Gaussian LPF for less ringing at boundaries.

    Args:
        image:     BGR uint8.
        intensity: 0.0 – 1.0

    Returns:
        BGR uint8 de-aged (smoother) image.
    """
    h, w = image.shape[:2]

    # Cutoff controls how much detail is removed
    # Higher cutoff → more frequencies pass → less smoothing
    # Lower cutoff → fewer frequencies pass → more smoothing
    # Scale cutoff inversely with intensity
    max_cutoff = max(h, w) / 4.0  # mild smoothing
    min_cutoff = max(h, w) / 12.0  # aggressive smoothing
    cutoff = max_cutoff - intensity * (max_cutoff - min_cutoff)

    # Butterworth LPF (order=2 for smooth roll-off)
    lpf = _create_butterworth_lpf((h, w), cutoff=cutoff, order=2)

    # Edge preservation mask: keep very low frequencies (structural edges)
    # by blending LPF result with original
    edge_cutoff = max(h, w) / 20.0
    edge_mask = _create_gaussian_lpf((h, w), cutoff=edge_cutoff)

    # Combined mask: LPF + boosted edges
    # This smooths mid-to-high freq (skin texture) while preserving low-freq (structure)
    combined_mask = lpf + 0.3 * intensity * (edge_mask - lpf)
    combined_mask = np.clip(combined_mask, 0, 1)

    result = image.copy()
    channels = cv2.split(result)
    smoothed = []

    for ch in channels:
        filtered = _fft_filter_channel(ch, combined_mask, blend_factor=1.0)
        # Blend filtered with original using intensity
        original = ch.astype(np.float64)
        blended = original * (1.0 - intensity * 0.85) + filtered.astype(np.float64) * (intensity * 0.85)
        smoothed.append(np.clip(blended, 0, 255).astype(np.uint8))

    return cv2.merge(smoothed)


# ============================================================
# 5.  Public API
# ============================================================
def apply_fft_filter(
    image: np.ndarray,
    operation: str,
    intensity: float,
) -> np.ndarray:
    """
    Applies an FFT-based aging or de-aging filter to the image.

    Args:
        image:     BGR uint8 input.
        operation: 'aging' or 'de-aging'.
        intensity: 0.0 – 1.0

    Returns:
        BGR uint8 processed image.
    """
    if intensity <= 0.001:
        return image.copy()

    if operation == "aging":
        return _apply_aging(image, intensity)
    elif operation == "de-aging":
        return _apply_deaging(image, intensity)
    else:
        raise ValueError(f"Unknown FFT operation: '{operation}'")


def compute_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Computes the log-scaled FFT magnitude spectrum for visualization.

    Pipeline:
        BGR → Grayscale → FFT2 → FFTShift → log(1 + |F|) → normalize [0, 255]

    Args:
        image: BGR uint8 input.

    Returns:
        BGR uint8 magnitude spectrum image (grayscale rendered as 3-channel).
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2D FFT
    f = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f)

    # Magnitude spectrum with log scaling
    magnitude = np.abs(f_shift)
    log_magnitude = np.log1p(magnitude)  # log(1 + |F(u,v)|)

    # Normalize to [0, 255]
    mag_min = log_magnitude.min()
    mag_max = log_magnitude.max()
    if mag_max - mag_min > 1e-10:
        normalized = (log_magnitude - mag_min) / (mag_max - mag_min) * 255.0
    else:
        normalized = np.zeros_like(log_magnitude)

    spectrum_gray = normalized.astype(np.uint8)

    # Apply a colormap for better visualization (inferno-like)
    spectrum_color = cv2.applyColorMap(spectrum_gray, cv2.COLORMAP_INFERNO)

    return spectrum_color
