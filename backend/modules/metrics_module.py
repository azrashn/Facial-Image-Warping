"""Quality metric utilities for image comparison."""

from __future__ import annotations

from typing import Dict

import cv2
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


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale when needed."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Unsupported image format for grayscale conversion.")


def _compute_data_range(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a safe dynamic range value for metric functions."""
    min_value = float(min(np.min(img1), np.min(img2)))
    max_value = float(max(np.max(img1), np.max(img2)))
    range_value = max_value - min_value
    return range_value if range_value > 0.0 else 1.0


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute Mean Squared Error (MSE) between two images."""
    _validate_images(img1, img2)
    mse_value = float(mean_squared_error(img1, img2))
    return {"mse": mse_value}


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
    _validate_images(img1, img2)
    data_range = _compute_data_range(img1, img2)
    psnr_value = float(peak_signal_noise_ratio(img1, img2, data_range=data_range))
    return {"psnr": psnr_value}


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute Structural Similarity Index (SSIM) between two images."""
    _validate_images(img1, img2)
    gray_img1 = _to_grayscale(img1)
    gray_img2 = _to_grayscale(img2)
    data_range = _compute_data_range(gray_img1, gray_img2)
    ssim_value = float(structural_similarity(gray_img1, gray_img2, data_range=data_range))
    return {"ssim": ssim_value}
