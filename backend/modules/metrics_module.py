"""Quality metric utilities for image comparison."""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def _validate_images(img1: np.ndarray, img2: np.ndarray) -> None:
    """Validate image inputs before metric computation."""
    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays.")
    if img1.size == 0 or img2.size == 0:
        raise ValueError("Input images must be non-empty.")
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")


def _normalize(image: np.ndarray) -> np.ndarray:
    """Cast to float64 and normalise to [0, 1]."""
    return image.astype(np.float64) / 255.0


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute Mean Squared Error (MSE) between two images.

    Operates on original [0, 255] pixel values so that the resulting MSE
    is in a human-readable range (e.g. 12.34 instead of 0.0000019).
    """
    _validate_images(img1, img2)
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mse_value = float(mean_squared_error(img1_f, img2_f))
    return {"mse": mse_value}


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Uses the original [0, 255] pixel scale with ``data_range=255.0``.
    """
    _validate_images(img1, img2)
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    psnr_value = float(peak_signal_noise_ratio(img1_f, img2_f, data_range=255.0))
    return {"psnr": psnr_value}


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute Structural Similarity Index (SSIM) between two images.

    Uses the original [0, 255] pixel scale with ``data_range=255.0``.
    """
    _validate_images(img1, img2)
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    ssim_value = float(
        structural_similarity(img1_f, img2_f, data_range=255.0, channel_axis=2)
    )
    return {"ssim": ssim_value}
