"""
metrics_module.py — Image Quality Metrics Calculator
=====================================================
Calculates MSE, PSNR, and SSIM between the original and processed images.
Handles shape/channel mismatches gracefully.

Uses scikit-image's reference implementations for numerical correctness.
"""

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
import cv2
from typing import Dict


def _ensure_compatible_shapes(
    original: np.ndarray, processed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensures both images have identical (H, W, C) shape for metric computation.

    Handles:
      - Resolution mismatch → resize processed to original's dimensions
      - Channel mismatch   → convert both to the same channel count
      - Grayscale (H,W) vs color (H,W,3) → expand/convert as needed
    """

    # --- Step 1: Normalize to 3D arrays (H, W, C) ---
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # --- Step 2: Channel count alignment ---
    orig_channels = original.shape[2] if original.ndim == 3 else 1
    proc_channels = processed.shape[2] if processed.ndim == 3 else 1

    if orig_channels != proc_channels:
        # Convert both to BGR (3 channels) as common ground
        if orig_channels == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        elif orig_channels == 1:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        if proc_channels == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)
        elif proc_channels == 1:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # --- Step 3: Spatial resolution alignment ---
    orig_h, orig_w = original.shape[:2]
    proc_h, proc_w = processed.shape[:2]

    if (orig_h, orig_w) != (proc_h, proc_w):
        processed = cv2.resize(
            processed,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return original, processed


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Mean Squared Error between two images.
    MSE = (1/N) * Σ(I_orig - I_proc)²

    Returns 0.0 for identical images.
    """
    original, processed = _ensure_compatible_shapes(original, processed)

    # skimage expects float64 for precision
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)

    mse_value = mean_squared_error(orig_f, proc_f)
    return round(float(mse_value), 4)


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio.
    PSNR = 10 * log10(MAX² / MSE)

    Returns float('inf') if images are identical (MSE = 0).
    For uint8 images, data_range = 255.
    """
    original, processed = _ensure_compatible_shapes(original, processed)

    # Determine the dynamic range from dtype
    if original.dtype == np.uint8:
        data_range = 255.0
    elif original.dtype == np.float32 or original.dtype == np.float64:
        data_range = 1.0 if original.max() <= 1.0 else 255.0
    else:
        data_range = 255.0

    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)

    # Handle identical images (MSE = 0 → PSNR = ∞)
    mse_val = mean_squared_error(orig_f, proc_f)
    if mse_val == 0.0:
        return float("inf")

    psnr_value = peak_signal_noise_ratio(
        orig_f, proc_f, data_range=data_range
    )
    return round(float(psnr_value), 4)


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Structural Similarity Index Measure.
    SSIM compares luminance, contrast, and structure between patches.

    Returns a value in [-1, 1] where 1 = identical.
    Uses channel_axis for multichannel images.
    """
    original, processed = _ensure_compatible_shapes(original, processed)

    if original.dtype == np.uint8:
        data_range = 255.0
    elif original.dtype == np.float32 or original.dtype == np.float64:
        data_range = 1.0 if original.max() <= 1.0 else 255.0
    else:
        data_range = 255.0

    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)

    # Determine minimum window size based on image dimensions
    min_dim = min(orig_f.shape[0], orig_f.shape[1])
    # SSIM default win_size is 7, must be odd and <= smallest image dimension
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    # Multichannel: use channel_axis parameter
    is_multichannel = orig_f.ndim == 3 and orig_f.shape[2] > 1

    ssim_value = structural_similarity(
        orig_f,
        proc_f,
        data_range=data_range,
        win_size=win_size,
        channel_axis=2 if is_multichannel else None,
    )
    return round(float(ssim_value), 4)


def compute_all_metrics(
    original: np.ndarray, processed: np.ndarray
) -> Dict[str, float]:
    """
    Computes all three quality metrics in a single call.

    Args:
        original:  Original image as numpy array (BGR, uint8).
        processed: Processed/warped image as numpy array (BGR, uint8).

    Returns:
        {"mse": float, "psnr": float, "ssim": float}
    """
    return {
        "mse": compute_mse(original, processed),
        "psnr": compute_psnr(original, processed),
        "ssim": compute_ssim(original, processed),
    }
