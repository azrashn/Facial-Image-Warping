import base64
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


def _build_face_hair_mask(image: np.ndarray) -> np.ndarray:
    """
    Build a smooth float mask [0..1] covering the face and hair region.

    Uses MediaPipe FaceMesh to find the face oval, then extends the mask
    upward to include the hair area.  Returns a single-channel float32
    array of the same (H, W) as *image*.
    """
    try:
        from modules.warping_module import detect_face_landmarks
    except ModuleNotFoundError:
        from backend.modules.warping_module import detect_face_landmarks

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    lm = detect_face_landmarks(image)
    if lm is None:
        # Fallback: treat the whole image as face (old behaviour)
        logger.warning("_build_face_hair_mask: no landmarks – full image mask")
        return np.ones((h, w), dtype=np.float32)

    # ── Face oval convex hull ────────────────────────────────────────
    # MediaPipe face-mesh silhouette (FACEMESH_FACE_OVAL) indices
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ]

    # Clamp indices to available landmarks
    n_lm = lm.shape[0]
    face_pts = lm[[i for i in face_oval_indices if i < n_lm]]
    hull = cv2.convexHull(face_pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)

    # ── Extend upward for hair ───────────────────────────────────────
    # Find the top of the face oval, then extend a rectangle up to the
    # image top (or a generous margin) to cover the hair / forehead.
    top_y = int(face_pts[:, 1].min())
    left_x = int(face_pts[:, 0].min())
    right_x = int(face_pts[:, 0].max())

    # Widen slightly for hair that extends beyond face width
    hair_pad_x = int((right_x - left_x) * 0.25)
    hair_left = max(0, left_x - hair_pad_x)
    hair_right = min(w, right_x + hair_pad_x)
    hair_top = 0  # all the way to the image top

    hair_rect = np.array([
        [hair_left, hair_top],
        [hair_right, hair_top],
        [hair_right, top_y],
        [hair_left, top_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, hair_rect, 1.0)

    # Also add side regions next to face for sideburns / ears
    face_center_y = int(face_pts[:, 1].mean())
    side_pad = int((right_x - left_x) * 0.15)
    # Left side
    left_side = np.array([
        [max(0, left_x - side_pad), top_y],
        [left_x, top_y],
        [left_x, face_center_y],
        [max(0, left_x - side_pad), face_center_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, left_side, 1.0)
    # Right side
    right_side = np.array([
        [right_x, top_y],
        [min(w, right_x + side_pad), top_y],
        [min(w, right_x + side_pad), face_center_y],
        [right_x, face_center_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, right_side, 1.0)

    # ── Smooth edges for seamless blending ────────────────────────────
    ksize = max(3, int(min(h, w) * 0.06) | 1)  # ensure odd
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    mask = np.clip(mask, 0.0, 1.0)

    return mask


def apply_aging_filter(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Realistic aging simulation restricted to the **face and hair** only.
    Background and clothing are left untouched.

    Three combined effects:
      1. Wrinkle / skin-texture enhancement  (frequency + CLAHE)
      2. Hair whitening / graying            (HSV colour manipulation)
      3. Subtle aged-skin colour tint        (LAB shift)

    A MediaPipe face-mesh mask ensures effects are composited only onto
    the face + hair region.
    """
    if image is None:
        raise ValueError("Input image is None.")

    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]

    # ── Build face + hair mask ────────────────────────────────────────
    face_mask = _build_face_hair_mask(image)          # float32 [0..1]
    face_mask_3 = face_mask[..., np.newaxis]           # (H,W,1) for BGR ops

    # ── 1. WRINKLE & TEXTURE ENHANCEMENT ──────────────────────────────
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # High-pass frequency filter → fine skin detail
    radius = int(14 + intensity * 40)
    high_pass = apply_frequency_filter(l_ch, radius=radius, mode="high")
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    # Strip broad muddy texture, keep only crisp detail
    detail_blur = cv2.GaussianBlur(high_pass, (0, 0), 1.0)
    detail = high_pass.astype(np.float32) - detail_blur.astype(np.float32)

    # CLAHE for local contrast – makes existing creases pop
    clip_limit = 2.2 + 3.2 * intensity
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_ch)

    # Build wrinkled luminance
    l_float = l_ch.astype(np.float32)
    detail_strength = 1.1 + 1.7 * intensity
    aged_l = l_float + detail * detail_strength

    # Blend in CLAHE version
    clahe_blend = 0.18 + 0.24 * intensity
    aged_l = aged_l * (1.0 - clahe_blend) + clahe_l.astype(np.float32) * clahe_blend

    # Mild contrast push + slight darkening
    contrast = 1.02 + 0.10 * intensity
    darkness = 2.0 + 5.0 * intensity
    aged_l = aged_l * contrast - darkness

    # Micro wrinkle noise
    noise = np.random.normal(0, 4 * intensity, l_ch.shape).astype(np.float32)
    aged_l = aged_l + noise
    aged_l = np.clip(aged_l, 0, 255).astype(np.uint8)

    # Merge back with original colour channels
    aged_lab = cv2.merge([aged_l, a_ch, b_ch])
    wrinkled = cv2.cvtColor(aged_lab, cv2.COLOR_LAB2BGR)

    # Subtle sharpening to crisp up fine lines
    blurred = cv2.GaussianBlur(wrinkled, (0, 0), 1.0)
    sharp_s = 0.16 + 0.20 * intensity
    wrinkled = cv2.addWeighted(wrinkled, 1.0 + sharp_s, blurred, -sharp_s, 0)

    # Blend with original to keep it natural
    blend_ratio = 0.32 + 0.30 * intensity
    wrinkled = cv2.addWeighted(image, 1.0 - blend_ratio, wrinkled, blend_ratio, 0)

    # ★ Composite wrinkle effect onto original using face mask
    result = (
        image.astype(np.float32) * (1.0 - face_mask_3)
        + wrinkled.astype(np.float32) * face_mask_3
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── 2. HAIR WHITENING / GRAYING ───────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_ch_hsv = hsv[:, :, 2].astype(np.float32)
    s_ch_hsv = hsv[:, :, 1].astype(np.float32)
    h_ch_hsv = hsv[:, :, 0].astype(np.float32)

    # Dark-pixel mask (hair is typically dark)
    dark_thresh = 145 + int(50 * intensity)
    dark_mask = np.clip((dark_thresh - v_ch_hsv) / max(dark_thresh, 1), 0, 1)

    # Position weight: upper portion of image is more likely hair
    y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1)
    pos_weight = np.clip(1.0 - y_coords * 1.1, 0.12, 1.0)
    pos_weight = np.broadcast_to(pos_weight, (h, w)).copy()

    # Exclude skin-coloured pixels (hue ≈ 0-30 in OpenCV 0-180 scale)
    skin_region = (
        (h_ch_hsv >= 0) & (h_ch_hsv <= 30)
        & (s_ch_hsv > 30) & (v_ch_hsv > 70)
    )
    not_skin = 1.0 - skin_region.astype(np.float32)

    # Combined hair mask — intersected with face_mask to avoid clothing
    hair_mask = dark_mask * pos_weight * not_skin * face_mask

    # Morphological cleanup for a coherent region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hair_u8 = (np.clip(hair_mask, 0, 1) * 255).astype(np.uint8)
    hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_CLOSE, kernel)
    hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_OPEN, kernel)
    hair_mask = hair_u8.astype(np.float32) / 255.0

    # Smooth edges for natural blending
    hair_mask = cv2.GaussianBlur(hair_mask, (25, 25), 10)

    # Apply desaturation + brightness boost in detected hair region
    white_str = 0.48 + 0.22 * intensity
    hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float64)

    # Desaturate → gray / white
    hsv_result[:, :, 1] *= (1.0 - hair_mask * white_str)

    # Lighten hair toward silver-white
    bright_add = (32 + 62 * intensity) * hair_mask
    hsv_result[:, :, 2] = np.clip(hsv_result[:, :, 2] + bright_add, 0, 255)

    hsv_result = np.clip(hsv_result, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)

    # ── 3. SUBTLE AGED-SKIN COLOUR TINT (face only) ──────────────────
    # Build the tinted version
    lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab_out[:, :, 2] = np.clip(lab_out[:, :, 2] + 1.0 + 1.8 * intensity, 0, 255)
    lab_out[:, :, 0] = np.clip(lab_out[:, :, 0] - 0.8 * intensity, 0, 255)
    tinted = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ★ Composite tint onto result using face mask (no tint on clothes)
    result = (
        result.astype(np.float32) * (1.0 - face_mask_3)
        + tinted.astype(np.float32) * face_mask_3
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


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
def apply_cartoon_filter(image: np.ndarray) -> np.ndarray:
    """
    Cartoon / caricature filter:
    - detects edges with Canny
    - smooths colors using bilateral filtering
    - reduces color levels with quantization
    - overlays black edges on the quantized image
    """
    if image is None:
        raise ValueError("Input image is None.")

    # 1) Edge detection
    gray = ensure_grayscale(image)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray_blur, 120, 230)
    edges = cv2.GaussianBlur(edges, (0, 0), 0.7)

    # 2) Smooth colors while preserving edges
    smooth = cv2.bilateralFilter(image, d=7, sigmaColor=55, sigmaSpace=55)

    # 3) Color quantization
    quantized = (smooth // 24) * 24
    quantized = cv2.addWeighted(smooth, 0.35, quantized, 0.65, 0)

    # 4) Combine edges with quantized image
    edge_mask = (edges.astype(np.float32) / 255.0)[..., None] * 0.55
    edge_color = np.zeros_like(quantized, dtype=np.float32)
    cartoon = (
        quantized.astype(np.float32) * (1.0 - edge_mask)
        + edge_color * edge_mask
    )

    return np.clip(cartoon, 0, 255).astype(np.uint8)

def _normalized_landmarks_to_points(
    landmarks: list,
    indices: list[int],
    width: int,
    height: int,
) -> np.ndarray:
    points = []

    for idx in indices:
        lm = landmarks[idx]

        if isinstance(lm, dict):
            x = int(lm["x"] * width)
            y = int(lm["y"] * height)
        else:
            x = int(lm[0] * width)
            y = int(lm[1] * height)

        points.append([x, y])

    return np.array(points, dtype=np.int32)


def _landmark_to_point(lm, width: int, height: int) -> tuple[int, int]:
    if isinstance(lm, dict):
        return int(lm["x"] * width), int(lm["y"] * height)

    return int(lm[0] * width), int(lm[1] * height)


def _apply_color_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    hue: int,
    opacity: float,
    saturation_multiplier: float = 1.4,
    blur_sigma: float = 5.0,
    normalize_mask: bool = True,
) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    hue = int(np.clip(hue, 0, 179))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    mask_float = mask.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0
    mask_float = np.clip(mask_float, 0.0, 1.0)

    mask_bool = mask_float > 0.01

    hsv[:, :, 0][mask_bool] = (
        hsv[:, :, 0][mask_bool] * 0.28 + hue * 0.72
    )
    hsv[:, :, 1][mask_bool] = np.clip(
        hsv[:, :, 1][mask_bool] * saturation_multiplier + 6,
        0,
        255,
    )

    colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    soft_mask = cv2.GaussianBlur(mask_float, (0, 0), blur_sigma)
    max_value = float(soft_mask.max())
    if normalize_mask and max_value > 0:
        soft_mask /= max_value
    soft_mask = soft_mask[..., None] * opacity

    result = (
        colored.astype(np.float32) * soft_mask
        + image.astype(np.float32) * (1.0 - soft_mask)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def _face_oval_float_mask(landmarks: list, width: int, height: int) -> np.ndarray:
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109,
    ]
    points = _normalized_landmarks_to_points(landmarks, face_oval_indices, width, height)
    face_mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(face_mask, [points], 1.0)
    return cv2.GaussianBlur(face_mask, (0, 0), max(3.0, min(width, height) * 0.018))


def _add_eyeshadow_gradient(
    mask: np.ndarray,
    eye_top: np.ndarray,
    brow_lower: np.ndarray,
) -> None:
    height, width = mask.shape
    polygon = np.vstack([eye_top, brow_lower[::-1]])
    poly_mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 1.0)

    x_min, y_min, box_w, box_h = cv2.boundingRect(polygon.astype(np.int32))
    if box_w <= 1 or box_h <= 1:
        return

    x0 = max(0, x_min)
    y0 = max(0, y_min)
    x1 = min(width, x_min + box_w)
    y1 = min(height, y_min + box_h)

    eye_y = float(np.mean(eye_top[:, 1]))
    brow_y = float(np.mean(brow_lower[:, 1]))
    low_y = min(eye_y, brow_y)
    high_y = max(eye_y, brow_y)

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    vertical = np.clip((yy - low_y) / max(high_y - low_y, 1.0), 0.0, 1.0)
    if brow_y > eye_y:
        vertical = 1.0 - vertical

    center_x = float(np.mean(eye_top[:, 0]))
    sigma_x = max(float(np.ptp(eye_top[:, 0])) * 0.62, 1.0)
    lateral = np.exp(-0.5 * ((xx - center_x) / sigma_x) ** 2)

    alpha = poly_mask[y0:y1, x0:x1] * (vertical ** 1.15) * lateral
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], alpha)


def _add_blush_gradient(
    mask: np.ndarray,
    center: tuple[int, int],
    radius_x: float,
    radius_y: float,
    angle_degrees: float,
) -> None:
    height, width = mask.shape
    cx, cy = center
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.deg2rad(angle_degrees)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    x = xx - float(cx)
    y = yy - float(cy)
    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    gaussian = np.exp(-0.5 * ((xr / max(radius_x, 1.0)) ** 2 + (yr / max(radius_y, 1.0)) ** 2))
    gaussian[gaussian < 0.08] = 0.0
    mask[:] = np.maximum(mask, gaussian)


def apply_virtual_makeup(
    image: np.ndarray,
    landmarks: list,
    region: str = "lip",
    hue: int = 0,
    opacity: float = 0.5,
) -> np.ndarray:
    """
    Virtual makeup using landmark masks + HSV color manipulation + alpha blending.

    region:
    - lip
    - blush
    - eyeshadow
    """
    if image is None:
        raise ValueError("Input image is None.")

    if not landmarks:
        raise ValueError("Landmarks are required for makeup.")

    h, w = image.shape[:2]
    region = (region or "").strip().lower()

    mask = np.zeros((h, w), dtype=np.float32)
    normalize_mask = True

    if region in {"lip", "lips"}:
        outer_lip = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        ]
        inner_lip = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
            308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
        ]
        outer_points = _normalized_landmarks_to_points(landmarks, outer_lip, w, h)
        inner_points = _normalized_landmarks_to_points(landmarks, inner_lip, w, h)
        cv2.fillPoly(mask, [outer_points], 1.0)
        cv2.fillPoly(mask, [inner_points], 0.0)
        saturation_multiplier = 1.45
        blur_sigma = max(1.5, min(h, w) * 0.004)
        opacity = min(opacity, 0.75)

    elif region == "eyeshadow":
        left_eye_top = [33, 246, 161, 160, 159, 158, 157, 173, 133]
        right_eye_top = [362, 398, 384, 385, 386, 387, 388, 466, 263]
        left_brow_lower = [55, 65, 52, 53, 46]
        right_brow_lower = [285, 295, 282, 283, 276]

        left_eye_points = _normalized_landmarks_to_points(landmarks, left_eye_top, w, h)
        right_eye_points = _normalized_landmarks_to_points(landmarks, right_eye_top, w, h)
        left_brow_points = _normalized_landmarks_to_points(landmarks, left_brow_lower, w, h)
        right_brow_points = _normalized_landmarks_to_points(landmarks, right_brow_lower, w, h)

        eye_clearance = max(2, int(0.006 * h))
        left_eye_points[:, 1] -= eye_clearance
        right_eye_points[:, 1] -= eye_clearance

        _add_eyeshadow_gradient(mask, left_eye_points, left_brow_points)
        _add_eyeshadow_gradient(mask, right_eye_points, right_brow_points)

        saturation_multiplier = 1.24
        blur_sigma = max(3.0, min(h, w) * 0.008)
        opacity = min(opacity, 0.58)
        normalize_mask = False

    elif region == "blush":
        left_cheek_center = landmarks[205]
        right_cheek_center = landmarks[425]

        lx, ly = _landmark_to_point(left_cheek_center, w, h)
        rx, ry = _landmark_to_point(right_cheek_center, w, h)

        radius_x = max(10.0, min(w, h) * 0.105)
        radius_y = max(8.0, min(w, h) * 0.068)

        _add_blush_gradient(mask, (lx, ly), radius_x, radius_y, -10)
        _add_blush_gradient(mask, (rx, ry), radius_x, radius_y, 10)
        mask *= _face_oval_float_mask(landmarks, w, h)

        saturation_multiplier = 1.12
        blur_sigma = max(5.0, min(h, w) * 0.012)
        opacity = min(opacity, 0.34)
        normalize_mask = False

    else:
        raise ValueError("Region must be 'lip', 'blush', or 'eyeshadow'.")

    return _apply_color_with_mask(
        image=image,
        mask=mask,
        hue=hue,
        opacity=opacity,
        saturation_multiplier=saturation_multiplier,
        blur_sigma=blur_sigma,
        normalize_mask=normalize_mask,
    )
def create_face_region_mask(image: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Create a soft face mask using MediaPipe FaceMesh face oval landmarks.
    This allows aging/de-aging effects to be applied only to the face area.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if not landmarks:
        raise ValueError("Landmarks are required for face mask.")

    h, w = image.shape[:2]

    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]

    points = []

    for idx in face_oval_indices:
        lm = landmarks[idx]

        if isinstance(lm, dict):
            x = int(lm["x"] * w)
            y = int(lm["y"] * h)
        else:
            x = int(lm[0] * w)
            y = int(lm[1] * h)

        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Smooth edges so the effect blends naturally with the background
    mask = cv2.GaussianBlur(mask, (0, 0), 12)

    return mask


def blend_effect_with_mask(
    original: np.ndarray,
    effected: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Blend effected image onto original image using a soft mask.
    Background stays unchanged.
    """
    if original is None or effected is None or mask is None:
        raise ValueError("Original, effected image and mask are required.")

    mask_float = mask.astype(np.float32) / 255.0
    mask_float = mask_float[..., None]

    result = (
        effected.astype(np.float32) * mask_float
        + original.astype(np.float32) * (1.0 - mask_float)
    )

    return np.clip(result, 0, 255).astype(np.uint8)

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode image as PNG base64 string.
    """
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Image could not be encoded to PNG.")

    return base64.b64encode(buffer).decode("utf-8")
