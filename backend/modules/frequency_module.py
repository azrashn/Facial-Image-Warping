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


def _build_face_hair_mask(image: np.ndarray, landmarks: np.ndarray = None) -> np.ndarray:
    """
    Build a smooth float mask [0..1] covering the face and hair region.

    Uses MediaPipe FaceMesh to find the face oval, then extends the mask
    upward to include the hair area.  Returns a single-channel float32
    array of the same (H, W) as *image*.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    lm = landmarks
    if lm is None:
        try:
            from modules.warping_module import detect_face_landmarks
        except ModuleNotFoundError:
            from backend.modules.warping_module import detect_face_landmarks
        lm = detect_face_landmarks(image)
        
    if lm is None:
        logger.warning("_build_face_hair_mask: no landmarks; using full image mask")
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
    return mask


def _build_face_oval_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if landmarks is None:
        return np.ones((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]
    n_lm = landmarks.shape[0]
    face_pts = landmarks[[i for i in face_oval_indices if i < n_lm]]
    hull = cv2.convexHull(face_pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)

    ksize = max(9, int(min(h, w) * 0.04) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _build_feathered_face_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if landmarks is None:
        return np.ones((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]
    n_lm = landmarks.shape[0]
    face_pts = landmarks[[i for i in face_oval_indices if i < n_lm]]
    hull = cv2.convexHull(face_pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)

    # ── Extend the mask upward slightly to ensure the forehead gets full coloring ──
    # Since Gaussian blur fades out the mask at the top of the face oval (landmark 10),
    # shifting the top forehead landmarks upward by 18% of the face scale keeps the forehead
    # values high (near 1.0) after blurring.
    face_sz = _face_scale(landmarks)
    up_shift = np.array([0, -0.18 * face_sz], dtype=np.float32)
    extended_pts = []
    for idx in [109, 67, 103, 10, 332, 297, 338]:
        if idx < n_lm:
            extended_pts.append(landmarks[idx] + up_shift)
    for idx in [338, 297, 332, 10, 103, 67, 109]:
        if idx < n_lm:
            extended_pts.append(landmarks[idx])
            
    if len(extended_pts) > 0:
        extended_pts = np.array(extended_pts, dtype=np.int32)
        cv2.fillPoly(mask, [extended_pts], 1.0)

    ksize = max(45, int(min(h, w) * 0.20) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.45)
    
    # Normalize the mask to ensure the center of the face gets the full color change intensity
    max_val = np.max(mask)
    if max_val > 0.001:
        mask = mask / max_val
        
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _build_pure_skin_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if landmarks is None:
        return np.ones((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]
    n_lm = landmarks.shape[0]
    face_pts = landmarks[[i for i in face_oval_indices if i < n_lm]]
    hull = cv2.convexHull(face_pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)

    exclude_polys = []
    
    # Left eye
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    exclude_polys.append(landmarks[[i for i in left_eye if i < n_lm]].astype(np.int32))
    
    # Right eye
    right_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    exclude_polys.append(landmarks[[i for i in right_eye if i < n_lm]].astype(np.int32))
    
    # Lips
    lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95]
    exclude_polys.append(landmarks[[i for i in lips if i < n_lm]].astype(np.int32))
    
    # Left Eyebrow
    left_brow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    exclude_polys.append(landmarks[[i for i in left_brow if i < n_lm]].astype(np.int32))
    
    # Right Eyebrow
    right_brow = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
    exclude_polys.append(landmarks[[i for i in right_brow if i < n_lm]].astype(np.int32))

    for poly in exclude_polys:
        cv2.fillPoly(mask, [poly], 0.0)

    ksize = max(3, int(min(h, w) * 0.02) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _build_crepiness_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if landmarks is None:
        return np.ones((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    n_lm = landmarks.shape[0]
    
    left_cheeks = [33, 133, 117, 50, 187, 216, 92, 111, 118]
    right_cheeks = [263, 362, 346, 280, 411, 436, 322, 340, 347]
    
    left_poly = landmarks[[i for i in left_cheeks if i < n_lm]].astype(np.int32)
    right_poly = landmarks[[i for i in right_cheeks if i < n_lm]].astype(np.int32)
    
    cv2.fillPoly(mask, [left_poly, right_poly], 1.0)
    
    ksize = max(15, int(min(h, w) * 0.12) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _face_scale(lm: np.ndarray) -> float:
    if lm is None or len(lm) < 363:
        return 100.0
    return float(np.linalg.norm(lm[133] - lm[362]))

# ============================================================================
# PHOTOREALISTIC AGING PIPELINE — Enhanced Sub-systems 1-8
# ============================================================================

# Module-level shared configuration for sagging control points (30+ points).
# Format: (landmark_idx, dx_factor, dy_factor, sigma_factor)
# Positive dy = downward (gravity), positive dx = rightward.
_SAGGING_CONTROL_POINTS = [
    # ── Cheek descent & mid-face deflation ──
    (117,  0.000,  0.160,  0.24),    # Left Cheek (dropping lower cheek fat)
    (346,  0.000,  0.160,  0.24),    # Right Cheek (dropping lower cheek fat)
    (50,   0.000,  0.000,  0.20),    # Left zygomatic (ANCHORED to make cheekbone prominent)
    (280,  0.000,  0.000,  0.20),    # Right zygomatic (ANCHORED to make cheekbone prominent)
    # ── EXTREME BULLDOG JOWLS (Bulging out and hanging low) ──
    # Left side
    (136, -0.080,  0.220,  0.22),    # Mid-jaw left (pushed out + drop)
    (150, -0.100,  0.280,  0.25),    # Jowl peak left (massive drop + out)
    (149, -0.060,  0.200,  0.20),    # Inner jowl left
    (172, -0.070,  0.240,  0.24),    # Left lower cheek fat pad
    # Right side
    (365,  0.080,  0.220,  0.22),    # Mid-jaw right
    (379,  0.100,  0.280,  0.25),    # Jowl peak right
    (378,  0.060,  0.200,  0.20),    # Inner jowl right
    (397,  0.070,  0.240,  0.24),    # Right lower cheek fat pad
    # ── BULLDOG ANCHORS (Deep Marionette Fold Illusion) ──
    # Anchor the chin and the area directly under the mouth so jowls hang past them
    (152,  0.000,  0.000,  0.25),    # Chin center
    (175,  0.000,  0.000,  0.20),    # Chin left support
    (400,  0.000,  0.000,  0.20),    # Chin right support
    (61,   0.020, -0.020,  0.15),    # Left mouth corner (pull slightly IN and UP)
    (291, -0.020, -0.020,  0.15),    # Right mouth corner (pull slightly IN and UP)
    (43,   0.020, -0.010,  0.15),    # Left marionette anchor
    (273, -0.020, -0.010,  0.15),    # Right marionette anchor
    # ── Temple hollowing (inward pull) ──
    (54,   0.060, -0.016,  0.18),    # Left temple (stronger inward pull for prominent cheekbones)
    (284, -0.060, -0.016,  0.18),    # Right temple
    # ── Brow droop ──
    (70,   0.000,  0.070,  0.16),    # Left mid-brow
    (300,  0.000,  0.070,  0.16),    # Right mid-brow
    # ── Eyelid hooding (upper eyelid skin fold) ──
    (159,  0.000,  0.150,  0.12),    # Left Upper Eyelid (heavy droop)
    (386,  0.000,  0.150,  0.12),    # Right Upper Eyelid (heavy droop)
    (145,  0.000,  0.120,  0.10),    # Left eyelid outer
    (374,  0.000,  0.120,  0.10),    # Right eyelid outer
    # ── Under-eye bag puffiness ──
    (111,  0.000,  0.055,  0.12),    # Left under-eye
    (340,  0.000,  0.055,  0.12),    # Right under-eye
    (116, -0.012,  0.045,  0.10),    # Left under-eye outer
    (345,  0.012,  0.045,  0.10),    # Right under-eye outer
    # ── Nasolabial pull ──
    (92,  -0.016,  0.060,  0.16),    # Left nasolabial
    (322,  0.016,  0.060,  0.16),    # Right nasolabial
    # ── Anchor points (zero displacement — prevent drift) ──
    (1,    0.000,  0.000,  0.15),    # Nose Tip
    (133,  0.000,  0.000,  0.12),    # Left Eye Inner Corner
    (362,  0.000,  0.000,  0.12),    # Right Eye Inner Corner
    (10,   0.000,  0.000,  0.25),    # Forehead Top
    (4,    0.000,  0.000,  0.10),    # Nose bridge
    (168,  0.000,  0.000,  0.10),    # Between eyebrows
]


def _warp_landmarks_for_sagging(landmarks: np.ndarray, shape: tuple[int, int], intensity: float) -> np.ndarray:
    """Warp landmark positions to simulate gravitational sagging with 30+ control points."""
    if landmarks is None or intensity < 0.01:
        return landmarks

    face_sz = _face_scale(landmarks)
    # Non-linear curve for more dramatic late-stage sagging
    eff_intensity = intensity ** 1.6

    warped_landmarks = landmarks.copy()
    for i in range(len(landmarks)):
        lx, ly = landmarks[i]
        l_weight_sum = 0.0
        l_disp_x = 0.0
        l_disp_y = 0.0
        for idx, dx_f, dy_f, sig_f in _SAGGING_CONTROL_POINTS:
            if idx >= len(landmarks):
                continue
            cx, cy = landmarks[idx]
            dx = dx_f * face_sz * eff_intensity
            dy = dy_f * face_sz * eff_intensity
            sigma = sig_f * face_sz
            r2 = (lx - cx)**2 + (ly - cy)**2
            w_i = np.exp(-0.5 * r2 / (sigma**2))
            l_disp_x += w_i * dx
            l_disp_y += w_i * dy
            l_weight_sum += w_i
        denom = l_weight_sum + 0.15
        warped_landmarks[i, 0] += l_disp_x / denom
        warped_landmarks[i, 1] += l_disp_y / denom

    return warped_landmarks


def _apply_sagging_warp(image: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Apply gravitational sagging warp to the image using 30+ control points."""
    if landmarks is None or intensity < 0.01:
        return image

    h, w = image.shape[:2]
    face_sz = _face_scale(landmarks)
    eff_intensity = intensity ** 1.8

    y_grid, x_grid = np.mgrid[:h, :w].astype(np.float32)

    disp_x = np.zeros((h, w), dtype=np.float32)
    disp_y = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for idx, dx_f, dy_f, sig_f in _SAGGING_CONTROL_POINTS:
        if idx >= len(landmarks):
            continue
        cx, cy = landmarks[idx]
        dx = dx_f * face_sz * eff_intensity
        dy = dy_f * face_sz * eff_intensity
        sigma = sig_f * face_sz

        r2 = (x_grid - cx)**2 + (y_grid - cy)**2
        w_i = np.exp(-0.5 * r2 / (sigma**2))

        disp_x += w_i * dx
        disp_y += w_i * dy
        weight_sum += w_i

    denom = weight_sum + 0.15
    final_disp_x = disp_x / denom
    final_disp_y = disp_y / denom

    map_x = x_grid - final_disp_x
    map_y = y_grid - final_disp_y

    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warped


def _apply_structural_wrinkles(image: np.ndarray, landmarks: np.ndarray, face_mask: np.ndarray, intensity: float, fast_mode: bool = False) -> np.ndarray:
    """
    Render photorealistic structural wrinkles using supersampled valley/hill depth mapping.

    Wrinkle types rendered:
      1. Forehead furrows (5 horizontal lines, 3 in fast mode)
      2. Glabellar ("11") lines (2 vertical furrows between brows)
      3. Crow's feet (3 radiating per eye with sub-branches)
      4. Nasolabial folds (deep curved grooves)
      5. Marionette lines (mouth corners to jaw)
      6. Under-eye wrinkles (fine horizontal lines)
      7. Perioral / lip radial wrinkles (skipped in fast mode)
      8. Neck horizontal lines (skipped in fast mode)
    """
    if landmarks is None or intensity < 0.01:
        return image

    h, w = image.shape[:2]
    face_sz = _face_scale(landmarks)
    n_lm = landmarks.shape[0]

    # Supersampling for anti-aliased wrinkle rendering (reduced in fast mode)
    ss_scale = 2.0 if fast_mode else 4.0
    ssh, ssw = int(h * ss_scale), int(w * ss_scale)

    W_valley = np.zeros((ssh, ssw), dtype=np.float32)
    W_hill = np.zeros((ssh, ssw), dtype=np.float32)
    y_grid, x_grid = np.mgrid[:h, :w].astype(np.float32)

    # ── Helper: generate wavy interpolated curve with 3-octave waviness ──
    def generate_wavy_curve(pts: np.ndarray, num_points: int = 120, wave_amp: float = 0.004, wave_freq: float = 12.0) -> np.ndarray:
        t_orig = np.linspace(0, 1, len(pts))
        t_new = np.linspace(0, 1, num_points)
        x_new = np.interp(t_new, t_orig, pts[:, 0])
        y_new = np.interp(t_new, t_orig, pts[:, 1])
        interpolated = np.stack([x_new, y_new], axis=1)

        wavy = interpolated.copy()
        for j in range(1, num_points - 1):
            tangent = interpolated[j+1] - interpolated[j-1]
            norm_t = np.linalg.norm(tangent)
            if norm_t < 1e-5:
                continue
            normal = np.array([-tangent[1], tangent[0]]) / norm_t

            # 3-octave waviness for organic look
            phase = t_new[j] * wave_freq * np.pi
            wave_val = (np.sin(phase) * 0.50
                        + np.sin(phase * 2.3) * 0.28
                        + np.sin(phase * 4.7) * 0.12
                        + np.random.normal(0, 0.10))
            wavy[j] += normal * wave_val * (wave_amp * face_sz)
        return wavy

    # ── Helper: draw a curve with tapered thickness and breaks ──
    def draw_tapered_and_broken_curve(canvas: np.ndarray, pts: np.ndarray, base_intensity: float, max_thickness: float, break_freq: float = 0.0, phase_offset: float = 0.0):
        num_segments = len(pts) - 1
        for j in range(num_segments):
            t0 = j / num_segments
            t1 = (j + 1) / num_segments
            t_mid = (t0 + t1) / 2.0

            # Tapering envelope (fades at ends)
            envelope = np.sin(np.pi * t_mid)

            # Break modulation (optional gaps)
            if break_freq > 0.0:
                break_val = np.sin(t_mid * break_freq + phase_offset)
                if break_val < -0.15:
                    envelope *= 0.1

            intensity_val = base_intensity * envelope
            thickness = max(1, int(round(max_thickness * ss_scale * envelope)))

            p0 = tuple((pts[j] * ss_scale).astype(np.int32))
            p1 = tuple((pts[j+1] * ss_scale).astype(np.int32))

            cv2.line(canvas, p0, p1, float(intensity_val), thickness, lineType=cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════
    # 1. FOREHEAD WRINKLES
    # ══════════════════════════════════════════════════════════════════════
    eyebrow_indices = [21, 71, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 301, 251]
    eyebrow_pts = landmarks[[i for i in eyebrow_indices if i < n_lm]]

    if 10 < n_lm and 9 < n_lm:
        v_forehead = landmarks[10] - landmarks[9]
        u_forehead = v_forehead / max(np.linalg.norm(v_forehead), 1e-5)

        forehead_shifts = [0.25, 0.45, 0.65] if fast_mode else [0.25, 0.45, 0.65]
        t_val = 1
        t_hil = 2

        for idx_f, shift in enumerate(forehead_shifts):
            pts_base = eyebrow_pts + shift * v_forehead
            # Center lines deeper than edge lines
            depth_factor = 1.0 - 0.12 * abs(idx_f - len(forehead_shifts) // 2)

            wavy_valley = generate_wavy_curve(pts_base, num_points=120, wave_amp=0.005, wave_freq=12.0)

            shift_dist = face_sz * 0.015
            wavy_hill = wavy_valley + u_forehead * shift_dist

            draw_tapered_and_broken_curve(W_valley, wavy_valley, 0.85 * depth_factor, t_val, break_freq=9.0, phase_offset=idx_f * 2.0)
            draw_tapered_and_broken_curve(W_hill, wavy_hill, 0.98 * depth_factor, t_hil, break_freq=9.0, phase_offset=idx_f * 2.0)

    # ══════════════════════════════════════════════════════════════════════
    # 2. GLABELLAR ("11") LINES — deep vertical furrows between brows
    # ══════════════════════════════════════════════════════════════════════
    if all(i < n_lm for i in [9, 168, 6, 107, 336]):
        glabella_center = landmarks[9]
        nose_bridge = landmarks[168]
        below_bridge = landmarks[6]

        left_brow_inner = landmarks[107]
        right_brow_inner = landmarks[336]

        brow_width = np.linalg.norm(left_brow_inner - right_brow_inner)
        v_down = below_bridge - glabella_center
        v_down_norm = v_down / max(np.linalg.norm(v_down), 1e-5)
        v_lateral = np.array([-v_down_norm[1], v_down_norm[0]], dtype=np.float32)

        for side_offset in [-0.18, 0.18]:
            lateral_shift = v_lateral * (brow_width * side_offset)

            top_pt = glabella_center + lateral_shift - v_down * 0.3
            mid_pt = nose_bridge + lateral_shift * 0.7
            bot_pt = below_bridge + lateral_shift * 0.4 + v_down * 0.1

            pts_glab = np.array([top_pt, mid_pt, bot_pt], dtype=np.float32)
            wavy_g = generate_wavy_curve(pts_glab, num_points=40, wave_amp=0.002, wave_freq=6.0)
            hill_g = wavy_g + v_lateral * (face_sz * 0.012) * np.sign(side_offset)

            # Glabellar lines are the deepest furrows
            draw_tapered_and_broken_curve(W_valley, wavy_g, 0.85, 2, break_freq=0.0)
            draw_tapered_and_broken_curve(W_hill, hill_g, 1.0, 3, break_freq=0.0)

    # ══════════════════════════════════════════════════════════════════════
    # 3. CROW'S FEET
    # ══════════════════════════════════════════════════════════════════════
    if 152 < n_lm and 9 < n_lm:
        v_face_down = landmarks[152] - landmarks[9]
        u_face_down = v_face_down / max(np.linalg.norm(v_face_down), 1e-5)
    else:
        u_face_down = np.array([0, 1], dtype=np.float32)

    for eye_type in ["left", "right"]:
        if eye_type == "left":
            outer_idx, inner_idx = 33, 133
            angle_offsets = [-0.38, 0.0, 0.38]
        else:
            outer_idx, inner_idx = 263, 362
            angle_offsets = [np.pi - 0.38, np.pi, np.pi + 0.38]

        if outer_idx >= n_lm or inner_idx >= n_lm:
            continue

        outer_pt = landmarks[outer_idx]
        inner_pt = landmarks[inner_idx]
        eye_w = np.linalg.norm(outer_pt - inner_pt)

        dir_vec = (outer_pt - inner_pt) / max(eye_w, 1.0)
        theta_0 = np.arctan2(dir_vec[1], dir_vec[0])

        for offset in angle_offsets:
            theta_target = theta_0 + (offset if eye_type == "left" else (offset - np.pi))
            u_dir = np.array([np.cos(theta_target), np.sin(theta_target)], dtype=np.float32)

            num_pts = 30
            t_vals = np.linspace(0, 1, num_pts)

            pts_main = []
            for t in t_vals:
                r = (0.15 + 0.85 * t) * (eye_w * 0.92)
                bend = (eye_w * 0.22) * (t ** 1.6)
                pt = outer_pt + r * u_dir + bend * u_face_down
                pts_main.append(pt)
            pts_main = np.array(pts_main)

            wavy_main = generate_wavy_curve(pts_main, num_points=40, wave_amp=0.003, wave_freq=8.0)
            shift_dist = face_sz * 0.010
            hill_main = wavy_main - u_face_down * shift_dist

            draw_tapered_and_broken_curve(W_valley, wavy_main, 0.85, 1)
            draw_tapered_and_broken_curve(W_hill, hill_main, 0.98, 2)

            # Sub-branch from middle crow's foot line
            if abs(offset) < 0.05 or abs(abs(offset) - np.pi) < 0.05:
                sub_start_idx = 12
                sub_pt_start = pts_main[sub_start_idx]

                theta_sub = theta_target + (0.38 if eye_type == "left" else -0.38)
                u_dir_sub = np.array([np.cos(theta_sub), np.sin(theta_sub)], dtype=np.float32)

                pts_sub = []
                for t in np.linspace(0, 1, 18):
                    r = t * (eye_w * 0.45)
                    bend = (eye_w * 0.12) * (t ** 1.6)
                    pt = sub_pt_start + r * u_dir_sub + bend * u_face_down
                    pts_sub.append(pt)
                pts_sub = np.array(pts_sub)

                wavy_sub = generate_wavy_curve(pts_sub, num_points=25, wave_amp=0.003, wave_freq=8.0)
                hill_sub = wavy_sub - u_face_down * shift_dist

                draw_tapered_and_broken_curve(W_valley, wavy_sub, 0.50, 1)
                draw_tapered_and_broken_curve(W_hill, hill_sub, 0.83, 2)

    # ══════════════════════════════════════════════════════════════════════
    # 4. NASOLABIAL FOLDS
    # ══════════════════════════════════════════════════════════════════════
    nose_tip = landmarks[1]
    for side in ["left", "right"]:
        if side == "left":
            p_start_idx, p_end_idx = 57, 61
        else:
            p_start_idx, p_end_idx = 287, 291

        if p_start_idx >= n_lm or p_end_idx >= n_lm:
            continue

        p_start = landmarks[p_start_idx]
        p_end = landmarks[p_end_idx]

        vec = p_end - p_start
        dist = np.linalg.norm(vec)
        p_mid = (p_start + p_end) / 2.0

        out_dir = p_mid - nose_tip
        out_dir /= max(np.linalg.norm(out_dir), 1e-6)

        p_ctrl = p_mid + out_dir * (dist * 0.22)
        ext_dir = vec / max(dist, 1e-6)
        p_ext = p_end + ext_dir * (dist * 0.28)

        t_values = np.linspace(0, 1, 40)
        curve_pts = []
        for t in t_values:
            pt = (1-t)**2 * p_start + 2*(1-t)*t * p_ctrl + t**2 * p_ext
            curve_pts.append(pt)
        curve_pts = np.array(curve_pts)

        wavy_nl = generate_wavy_curve(curve_pts, num_points=60, wave_amp=0.002, wave_freq=6.0)

        shift_dist_nl = face_sz * 0.018
        hill_nl = wavy_nl + out_dir * shift_dist_nl

        # Secondary shallow fold parallel to main
        shift_sec = out_dir * (dist * 0.16)
        wavy_nl_sec = wavy_nl + shift_sec
        hill_nl_sec = hill_nl + shift_sec

        draw_tapered_and_broken_curve(W_valley, wavy_nl, 1.15, 2)
        draw_tapered_and_broken_curve(W_hill, hill_nl, 0.98, 3)

        draw_tapered_and_broken_curve(W_valley, wavy_nl_sec, 0.30, 1)
        draw_tapered_and_broken_curve(W_hill, hill_nl_sec, 0.49, 2)

    # ══════════════════════════════════════════════════════════════════════
    # 5. MARIONETTE LINES — mouth corners → jaw
    # ══════════════════════════════════════════════════════════════════════
    for side in ["left", "right"]:
        mouth_idx = 61 if side == "left" else 291
        jaw_idx = 172 if side == "left" else 397
        chin_idx = 152

        if any(i >= n_lm for i in [mouth_idx, jaw_idx, chin_idx]):
            continue

        mouth_corner = landmarks[mouth_idx]
        jaw_pt = landmarks[jaw_idx]
        chin = landmarks[chin_idx]

        ctrl_pt = (mouth_corner + jaw_pt) / 2.0
        ctrl_pt[1] += face_sz * 0.05
        if side == "left":
            ctrl_pt[0] -= face_sz * 0.02
        else:
            ctrl_pt[0] += face_sz * 0.02

        end_pt = jaw_pt.copy() + (jaw_pt - chin) * 0.15
        end_pt[1] += face_sz * 0.03

        t_vals_mar = np.linspace(0, 1, 35)
        curve_pts_mar = []
        for t in t_vals_mar:
            pt = (1-t)**2 * mouth_corner + 2*(1-t)*t * ctrl_pt + t**2 * end_pt
            curve_pts_mar.append(pt)
        curve_pts_mar = np.array(curve_pts_mar)

        wavy_mar = generate_wavy_curve(curve_pts_mar, num_points=50, wave_amp=0.002, wave_freq=7.0)

        out_dir_mar = np.array([-1.0, 0.3] if side == "left" else [1.0, 0.3], dtype=np.float32)
        out_dir_mar /= np.linalg.norm(out_dir_mar)
        hill_mar = wavy_mar + out_dir_mar * (face_sz * 0.014)

        draw_tapered_and_broken_curve(W_valley, wavy_mar, 1.30, 2)
        draw_tapered_and_broken_curve(W_hill, hill_mar, 0.85, 3)

    # ══════════════════════════════════════════════════════════════════════
    # 6. UNDER-EYE WRINKLES — fine horizontal lines below the eyes
    # ══════════════════════════════════════════════════════════════════════
    for eye_side in ["left", "right"]:
        if eye_side == "left":
            # Lower left eye contour (inner→outer): 133,155,154,153,145,144,163,7,33
            contour_indices = [133, 155, 154, 153, 145, 144, 163, 7, 33]
        else:
            # Lower right eye contour (inner→outer): 362,382,381,380,374,373,390,249,263
            contour_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263]

        valid_indices = [i for i in contour_indices if i < n_lm]
        if len(valid_indices) < 4:
            continue

        contour_pts = landmarks[valid_indices]

        for line_offset in range(3):
            offset_vec = np.array([0.0, face_sz * 0.014 * (line_offset + 1)], dtype=np.float32)
            shifted_pts = contour_pts + offset_vec

            wavy_ue = generate_wavy_curve(shifted_pts, num_points=30, wave_amp=0.001, wave_freq=14.0)
            hill_ue = wavy_ue + np.array([0.0, -face_sz * 0.006], dtype=np.float32)

            line_intensity = 0.60 - line_offset * 0.15
            draw_tapered_and_broken_curve(W_valley, wavy_ue, line_intensity, 1, break_freq=12.0, phase_offset=line_offset * 3.0)
            draw_tapered_and_broken_curve(W_hill, hill_ue, line_intensity, 1, break_freq=12.0, phase_offset=line_offset * 3.0)

    # ══════════════════════════════════════════════════════════════════════
    # 7. LIP RADIAL (PERIORAL) WRINKLES — skipped in fast mode
    # ══════════════════════════════════════════════════════════════════════
    if not fast_mode and all(i < n_lm for i in [0, 17, 61, 291]):
        lip_top = landmarks[0]
        lip_bot = landmarks[17]
        lip_left = landmarks[61]
        lip_right = landmarks[291]
        lip_center = (lip_top + lip_bot) / 2.0
        lip_h = np.linalg.norm(lip_top - lip_bot)
        lip_w = np.linalg.norm(lip_left - lip_right)

        num_radial = 14
        for i_r in range(num_radial):
            angle = (i_r / num_radial) * 2.0 * np.pi
            u_rad = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

            start_r = max(lip_h, lip_w) * 0.42
            end_r = start_r + face_sz * 0.04

            pts_rad = []
            for t in np.linspace(0, 1, 12):
                r = start_r + t * (end_r - start_r)
                pt = lip_center + u_rad * r
                pts_rad.append(pt)
            pts_rad = np.array(pts_rad)

            wavy_lip = generate_wavy_curve(pts_rad, num_points=15, wave_amp=0.001, wave_freq=18.0)
            draw_tapered_and_broken_curve(W_valley, wavy_lip, 0.28, 1, break_freq=0.0)

    # ══════════════════════════════════════════════════════════════════════
    # 8. NECK HORIZONTAL LINES — skipped in fast mode
    # ══════════════════════════════════════════════════════════════════════
    if not fast_mode and 152 < n_lm:
        chin_pt = landmarks[152]

        for neck_line in range(3):
            y_offset = face_sz * (0.18 + 0.08 * neck_line)
            neck_center = chin_pt + np.array([0.0, y_offset], dtype=np.float32)

            if neck_center[1] >= h - 5:
                continue

            neck_left = neck_center + np.array([-face_sz * 0.55, 0.0], dtype=np.float32)
            neck_right = neck_center + np.array([face_sz * 0.55, 0.0], dtype=np.float32)
            neck_pts = np.array([neck_left, neck_center, neck_right], dtype=np.float32)

            wavy_neck = generate_wavy_curve(neck_pts, num_points=50, wave_amp=0.003, wave_freq=5.0)
            hill_neck = wavy_neck + np.array([0.0, -face_sz * 0.008], dtype=np.float32)

            neck_intensity = 0.55 - neck_line * 0.12
            draw_tapered_and_broken_curve(W_valley, wavy_neck, neck_intensity, 1, break_freq=6.0, phase_offset=neck_line)
            draw_tapered_and_broken_curve(W_hill, hill_neck, neck_intensity, 2, break_freq=6.0, phase_offset=neck_line)

    # ══════════════════════════════════════════════════════════════════════
    # 9. BLURRING & HARMONIZING VALLEY/HILL CANVASES
    # ══════════════════════════════════════════════════════════════════════
    sigma_v_sharp = max(0.5, face_sz * 0.003) * ss_scale
    sigma_v_soft = max(2.0, face_sz * 0.014) * ss_scale
    sigma_h = max(2.2, face_sz * 0.016) * ss_scale

    ksize_v_sharp = int(round(sigma_v_sharp * 3)) | 1
    ksize_v_soft = int(round(sigma_v_soft * 3)) | 1
    ksize_h = int(round(sigma_h * 3)) | 1

    W_valley_sharp = cv2.GaussianBlur(W_valley, (ksize_v_sharp, ksize_v_sharp), sigma_v_sharp)
    W_valley_soft = cv2.GaussianBlur(W_valley, (ksize_v_soft, ksize_v_soft), sigma_v_soft)
    W_hill_blurred = cv2.GaussianBlur(W_hill, (ksize_h, ksize_h), sigma_h)

    # Combined depth map: sharp crease + soft shading - wider hill overhang
    W_ss = 0.65 * W_valley_sharp + 0.35 * W_valley_soft - 0.45 * W_hill_blurred

    # Downsample back to original resolution
    W = cv2.resize(W_ss, (w, h), interpolation=cv2.INTER_AREA)

    # Crop to face oval to prevent wrinkles on ears/hair
    W *= _build_face_oval_mask(image, landmarks)

    # ══════════════════════════════════════════════════════════════════════
    # 10. MESH WARPING (PIXEL CONTRACTION/EXPANSION)
    # ══════════════════════════════════════════════════════════════════════
    dx = cv2.filter2D(W, cv2.CV_32F, np.array([[-0.5, 0, 0.5]], dtype=np.float32))
    dy = cv2.filter2D(W, cv2.CV_32F, np.array([[-0.5], [0], [0.5]], dtype=np.float32))

    dx = cv2.GaussianBlur(dx, (3, 3), 0.5)
    dy = cv2.GaussianBlur(dy, (3, 3), 0.5)

    warp_scale = face_sz * 0.22 * intensity
    map_x = x_grid - dx * warp_scale
    map_y = y_grid - dy * warp_scale

    warped_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # ══════════════════════════════════════════════════════════════════════
    # 11. DIRECTIONAL BUMP MAPPING — physically correct shading
    # ══════════════════════════════════════════════════════════════════════
    bump = dx * 0.707 + dy * 0.707

    # Deepen shading to make cheekbones pop, but not so muddy
    shading = -0.50 * W - 0.35 * bump
    shading *= intensity

    lab = cv2.cvtColor(warped_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + lab[:, :, 0] * shading, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def _apply_color_transfer(image: np.ndarray, mask: np.ndarray, intensity: float, landmarks: np.ndarray = None) -> np.ndarray:
    """Adaptive skin-tone color transfer for aging: sallowness, pallor, reduced redness."""
    if intensity < 0.01 or mask is None:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    if landmarks is not None:
        analysis_mask = _build_pure_skin_mask(image, landmarks)
    else:
        analysis_mask = mask

    mask_indices = analysis_mask > 0.1
    if not np.any(mask_indices):
        mask_indices = mask > 0.1
        if not np.any(mask_indices):
            return image

    mean_L = np.mean(L[mask_indices])
    std_L = np.std(L[mask_indices])
    mean_a = np.mean(a[mask_indices])
    std_a = np.std(a[mask_indices])
    mean_b = np.mean(b[mask_indices])
    std_b = np.std(b[mask_indices])

    # Aging color shifts (enhanced for photorealistic pallor/sallowness):
    # L: slight darkening + reduced dynamic range for aged skin
    target_mean_L = mean_L - 1.5 * intensity
    target_std_L = std_L * (1.0 - 0.15 * intensity)

    # a: reduced redness → natural pallor (prevent zombie green)
    target_mean_a = mean_a - 1.8 * intensity
    # b: very subtle yellowing (minimal — avoid jaundice)
    target_mean_b = mean_b + 1.2 * intensity

    scale_L = target_std_L / max(std_L, 1e-5)

    L_new = (L - mean_L) * scale_L + target_mean_L
    a_new = a + (target_mean_a - mean_a)
    b_new = b + (target_mean_b - mean_b)

    L_new = np.clip(L_new, 0, 255)
    a_new = np.clip(a_new, 0, 255)
    b_new = np.clip(b_new, 0, 255)

    lab_new = cv2.merge([L_new, a_new, b_new]).astype(np.uint8)
    result = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    mask_3 = mask[..., np.newaxis]
    blended = image.astype(np.float32) * (1.0 - mask_3) + result.astype(np.float32) * mask_3
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_aging_lut(image: np.ndarray, mask: np.ndarray, intensity: float) -> np.ndarray:
    """Apply an aging color LUT: subtle yellowing and desaturation."""
    if intensity < 0.01 or mask is None:
        return image

    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    for i in range(256):
        val = i / 255.0
        # Enhanced yellowing + slight blue-channel suppression
        b_val = val**1.03 - 0.008 * intensity
        g_val = val**1.01 - 0.003 * intensity
        r_val = val**0.995

        lut[0, i, 0] = np.clip(b_val * 255.0, 0, 255)
        lut[0, i, 1] = np.clip(g_val * 255.0, 0, 255)
        lut[0, i, 2] = np.clip(r_val * 255.0, 0, 255)

    graded = cv2.LUT(image, lut)

    mask_3 = mask[..., np.newaxis]
    blended = image.astype(np.float32) * (1.0 - mask_3) + graded.astype(np.float32) * mask_3
    return np.clip(blended, 0, 255).astype(np.uint8)


def _generate_solar_spots(image: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """
    Generate 28 photorealistic age/sun spots with color variation.

    Spot types:
      - Liver spots (dark brown, LAB shift)
      - Sun damage (light brown)
      - Broken capillaries (reddish)
    """
    if landmarks is None or intensity < 0.05:
        return image

    h, w = image.shape[:2]
    face_sz = _face_scale(landmarks)

    # Noise for irregular spot shapes
    noise = np.random.normal(0, 1.0, (h // 2, w // 2)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = cv2.GaussianBlur(noise, (11, 11), 3)

    # 28 spots: (anchor_idx, dx_rel, dy_rel, radius_rel, opacity, color_type)
    # color_type: 0=liver(dark brown), 1=sun(light brown), 2=capillary(reddish)
    spots_config = [
        # Left cheek (dense sun exposure area)
        (117,  0.05, -0.02, 0.022, 0.70, 0),
        (117, -0.06,  0.08, 0.018, 0.60, 1),
        (50,   0.02,  0.03, 0.025, 0.55, 1),
        (117,  0.10,  0.04, 0.014, 0.50, 2),
        (117, -0.03, -0.06, 0.010, 0.45, 1),
        # Right cheek
        (346, -0.05, -0.03, 0.020, 0.65, 0),
        (346,  0.06,  0.06, 0.024, 0.70, 1),
        (280, -0.02,  0.02, 0.016, 0.60, 1),
        (346, -0.10,  0.03, 0.012, 0.50, 2),
        (346,  0.03, -0.05, 0.011, 0.45, 1),
        # Forehead (high sun exposure)
        (109,  0.08, -0.04, 0.028, 0.50, 1),
        (338, -0.08, -0.05, 0.022, 0.55, 1),
        (67,   0.02, -0.06, 0.018, 0.45, 0),
        (10,   0.05,  0.04, 0.015, 0.40, 1),
        (10,  -0.06,  0.03, 0.020, 0.50, 1),
        (10,   0.12, -0.02, 0.012, 0.35, 2),
        # Temple / eye area
        (139, -0.02, -0.02, 0.020, 0.60, 2),
        (368,  0.02, -0.02, 0.022, 0.60, 2),
        (139,  0.04,  0.03, 0.010, 0.40, 1),
        (368, -0.04,  0.03, 0.010, 0.40, 1),
        # Nose bridge / sides
        (6,    0.04,  0.02, 0.012, 0.45, 1),
        (6,   -0.03,  0.04, 0.010, 0.40, 2),
        # Jawline
        (132,  0.03,  0.02, 0.018, 0.35, 1),
        (361, -0.03,  0.02, 0.016, 0.35, 1),
        # Small freckle-like spots
        (117,  0.12, -0.04, 0.007, 0.30, 1),
        (346, -0.11, -0.05, 0.008, 0.30, 1),
        (50,   0.06, -0.03, 0.006, 0.25, 1),
        (280, -0.06, -0.03, 0.006, 0.25, 1),
    ]

    spots_mask = np.zeros((h, w), dtype=np.float32)
    spots_color_a = np.zeros((h, w), dtype=np.float32)
    spots_color_b = np.zeros((h, w), dtype=np.float32)

    for idx, dx_r, dy_r, rad_r, opac, ctype in spots_config:
        if idx >= len(landmarks):
            continue
        cx, cy = landmarks[idx]
        x_spot = int(cx + dx_r * face_sz)
        y_spot = int(cy + dy_r * face_sz)
        radius = int(rad_r * face_sz)

        if radius < 1:
            continue

        x0 = max(0, x_spot - radius * 2)
        y0 = max(0, y_spot - radius * 2)
        x1 = min(w, x_spot + radius * 2)
        y1 = min(h, y_spot + radius * 2)

        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]

        noise_x = noise[y0:y1, x0:x1] * (radius * 0.32)
        noise_y = noise[y0:y1, x0:x1] * (radius * 0.32)
        dist2 = (xx - x_spot + noise_x)**2 + (yy - y_spot + noise_y)**2

        sigma_spot = radius * 0.55
        spot_patch = np.exp(-0.5 * dist2 / (sigma_spot**2)) * opac

        spots_mask[y0:y1, x0:x1] = np.maximum(spots_mask[y0:y1, x0:x1], spot_patch)

        # Color variation by type
        if ctype == 0:    # Liver spot (dark brown)
            spots_color_a[y0:y1, x0:x1] += spot_patch * 6.0
            spots_color_b[y0:y1, x0:x1] += spot_patch * 18.0
        elif ctype == 1:  # Sun damage (light brown)
            spots_color_a[y0:y1, x0:x1] += spot_patch * 3.0
            spots_color_b[y0:y1, x0:x1] += spot_patch * 12.0
        else:             # Broken capillary (reddish)
            spots_color_a[y0:y1, x0:x1] += spot_patch * 12.0
            spots_color_b[y0:y1, x0:x1] += spot_patch * 4.0

    spots_mask *= intensity * 0.85

    # Restrict spots strictly to face oval to prevent background contamination
    face_oval = _build_face_oval_mask(image, landmarks)
    spots_mask *= face_oval
    spots_color_a *= face_oval
    spots_color_b *= face_oval

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] - spots_mask * 42.0, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + spots_color_a * intensity * 0.85, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + spots_color_b * intensity * 0.85, 0, 255)

    spotted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Blend back using feathered face oval to avoid any hard edges
    m3 = face_oval[..., np.newaxis]
    blended = image.astype(np.float32) * (1.0 - m3) + spotted.astype(np.float32) * m3
    return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================================
# NEW HELPER FUNCTIONS — Sub-system 3, 5, 6
# ============================================================================

def _apply_skin_texture_aging(image: np.ndarray, landmarks: np.ndarray, face_oval_mask: np.ndarray, intensity: float) -> np.ndarray:
    """
    Skin texture degradation: pore coarsening, translucency loss, uneven pigmentation.

    Simulates the three main physical changes in aged skin:
      1. Pore coarsening — amplified high-frequency skin detail
      2. Translucency loss — increased local contrast (less subsurface scattering)
      3. Uneven pigmentation — Perlin-like noise in LAB a/b channels
    """
    if intensity < 0.05 or landmarks is None:
        return image

    h, w = image.shape[:2]
    face_sz = _face_scale(landmarks)
    skin_mask = _build_pure_skin_mask(image, landmarks)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ch = lab[:, :, 0].copy()

    # 1. Pore coarsening — amplify high-frequency skin detail
    sigma_pore = max(1.0, face_sz * 0.008)
    smooth_L = cv2.GaussianBlur(L_ch, (0, 0), sigma_pore)
    detail = L_ch - smooth_L
    pore_strength = 0.4 + 0.8 * intensity
    L_ch = L_ch + detail * pore_strength * skin_mask

    # 2. Translucency loss — increased local contrast (reverse SSS)
    clahe = cv2.createCLAHE(clipLimit=1.5 + 2.5 * intensity, tileGridSize=(12, 12))
    L_u8 = np.clip(L_ch, 0, 255).astype(np.uint8)
    clahe_L = clahe.apply(L_u8).astype(np.float32)
    trans_blend = 0.15 + 0.20 * intensity
    L_ch = L_ch * (1.0 - trans_blend * skin_mask) + clahe_L * (trans_blend * skin_mask)

    # 3. Uneven pigmentation — Perlin-like noise mapped to LAB a/b
    noise_scale = max(4, int(h * 0.08))
    noise_a = np.random.normal(0, 1.0, (noise_scale, noise_scale)).astype(np.float32)
    noise_b = np.random.normal(0, 1.0, (noise_scale, noise_scale)).astype(np.float32)
    noise_a = cv2.resize(noise_a, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_b = cv2.resize(noise_b, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_a = cv2.GaussianBlur(noise_a, (0, 0), max(1.0, face_sz * 0.05))
    noise_b = cv2.GaussianBlur(noise_b, (0, 0), max(1.0, face_sz * 0.05))

    pigment_strength = 1.5 * intensity

    lab[:, :, 0] = np.clip(L_ch, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + noise_a * pigment_strength * skin_mask, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + noise_b * pigment_strength * skin_mask, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Blend with original using face oval mask at controlled strength
    blend_factor = 0.2 + 0.3 * intensity
    mask_3 = face_oval_mask[..., np.newaxis]
    effective_mask = mask_3 * blend_factor
    blended = image.astype(np.float32) * (1.0 - effective_mask) + result.astype(np.float32) * effective_mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_eyebrow_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Build a dilated mask covering both eyebrows for whitening."""
    h, w = image.shape[:2]
    if landmarks is None:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    n_lm = landmarks.shape[0]

    left_brow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_brow = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

    for brow_indices in [left_brow, right_brow]:
        pts = landmarks[[i for i in brow_indices if i < n_lm]].astype(np.int32)
        if len(pts) < 3:
            continue
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)

    # Dilate to cover full brow area including fine hairs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=2)
    mask = mask_u8.astype(np.float32) / 255.0

    ksize = max(5, int(min(h, w) * 0.02) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    return np.clip(mask, 0.0, 1.0)


def _build_lower_face_hair_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Build a mask for the lower face region (chin/jawline) for facial hair whitening."""
    h, w = image.shape[:2]
    if landmarks is None:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    n_lm = landmarks.shape[0]

    # Broad lower face covering jaw, chin, mustache, and cheek beard
    lower_face_indices = [
        # Jawline & Jowls
        132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
        # Mustache/Upper lip area (broad)
        164, 2, 97, 326,
        # Cheek beard line
        205, 116, 123, 147, 213, 192, 214, 434, 416, 433, 376, 352, 345, 425
    ]

    pts = landmarks[[i for i in lower_face_indices if i < n_lm]].astype(np.int32)
    if len(pts) < 5:
        return mask

    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1.0)
    
    # Use a mild blur instead of aggressive dilation to keep boundaries tight but soft
    ksize = max(5, int(min(h, w) * 0.02) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # Exclude lips (we don't want to whiten lip skin)
    lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            308, 324, 318, 402, 317, 14, 87, 178, 95]
    lip_pts = landmarks[[i for i in lips if i < n_lm]].astype(np.int32)
    if len(lip_pts) > 3:
        cv2.fillPoly(mask, [lip_pts], 0.0)

    ksize = max(7, int(min(h, w) * 0.03) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    return np.clip(mask, 0.0, 1.0)


def _build_periorbital_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Build a mask covering the under-eye (periorbital) region for darkening."""
    h, w = image.shape[:2]
    if landmarks is None:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    n_lm = landmarks.shape[0]
    face_sz = _face_scale(landmarks)

    # Lower eye contour landmarks (well-documented MediaPipe indices)
    left_lower_eye = [133, 155, 154, 153, 145, 144, 163, 7, 33]
    right_lower_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263]

    for indices in [left_lower_eye, right_lower_eye]:
        pts = landmarks[[i for i in indices if i < n_lm]]
        if len(pts) < 4:
            continue
        # Shift downward to create under-eye bag region
        shifted_pts = pts.copy()
        shifted_pts[:, 1] += face_sz * 0.04

        both = np.vstack([pts, shifted_pts]).astype(np.int32)
        hull = cv2.convexHull(both)
        cv2.fillConvexPoly(mask, hull, 1.0)

    ksize = max(9, int(face_sz * 0.08) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.4)
    return np.clip(mask, 0.0, 1.0)


def _build_lip_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Build a mask covering the lip region for color desaturation."""
    h, w = image.shape[:2]
    if landmarks is None:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    n_lm = landmarks.shape[0]

    lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            308, 324, 318, 402, 317, 14, 87, 178, 95]
    pts = landmarks[[i for i in lips if i < n_lm]].astype(np.int32)
    if len(pts) < 5:
        return mask

    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1.0)

    ksize = max(5, int(min(h, w) * 0.015) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.3)
    return np.clip(mask, 0.0, 1.0)


def _apply_periorbital_darkening(image: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Darken the under-eye area to simulate periorbital aging (dark circles / bags)."""
    if landmarks is None or intensity < 0.05:
        return image

    periorbital = _build_periorbital_mask(image, landmarks)
    if np.max(periorbital) < 0.01:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Darken L channel + slight bluish tint (reduce b channel)
    dark_amount = 12.0 * intensity
    lab[:, :, 0] = np.clip(lab[:, :, 0] - periorbital * dark_amount, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + periorbital * 2.0 * intensity, 0, 255) # slight red
    lab[:, :, 2] = np.clip(lab[:, :, 2] - periorbital * 4.0 * intensity, 0, 255) # reduce yellow (more blue)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def _apply_lip_desaturation(image: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Reduce lip color saturation to simulate aging (lips lose color with age)."""
    if landmarks is None or intensity < 0.05:
        return image

    lip_mask = _build_lip_mask(image, landmarks)
    if np.max(lip_mask) < 0.01:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    desat = 0.25 * intensity
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - desat * lip_mask), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] - lip_mask * 8.0 * intensity, 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3 = lip_mask[..., np.newaxis]
    blended = image.astype(np.float32) * (1.0 - mask_3) + result.astype(np.float32) * mask_3
    return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR — apply_aging_filter
# ============================================================================

def apply_aging_filter(image: np.ndarray, intensity: float = 0.5, landmarks: np.ndarray = None) -> np.ndarray:
    """
    Photorealistic aging simulation restricted to the **face and hair** only.
    Background and clothing are left untouched.

    Pipeline (8 sub-systems):
      0. Facial Mesh Warp / Gravitational Sagging (30+ control points)
      1. Skin texture aging (pore coarsening, translucency loss, pigmentation)
      2. Micro wrinkle & texture enhancement (frequency + CLAHE)
      3. Structural wrinkles (forehead, glabellar, crow's feet, nasolabial,
         marionette, under-eye, perioral, neck)
      4. Color transfer, LUT, periorbital darkening, lip desaturation
      5. Solar/age spots (28 spots with color variation)
      6. Multi-region hair whitening (head + eyebrows + facial hair)
      7. Final compositing & tinting

    A MediaPipe face-mesh mask ensures effects are composited only onto
    the face + hair region.
    """
    if image is None:
        raise ValueError("Input image is None.")

    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]

    # Performance: auto fast-mode for small images (live webcam ≤640px)
    fast_mode = max(h, w) <= 640

    # Native resolution processing
    ds_factor = 1.0
    lo_image = image
    lh, lw = lo_image.shape[:2]

    # ── Build face + hair mask ────────────────────────────────────────
    if landmarks is not None and ds_factor > 1.0:
        lo_landmarks = landmarks / ds_factor
    else:
        lo_landmarks = landmarks

    if lo_landmarks is None:
        try:
            from modules.warping_module import detect_face_landmarks
        except ModuleNotFoundError:
            from backend.modules.warping_module import detect_face_landmarks
        lo_landmarks = detect_face_landmarks(lo_image)

    # ── 0. SAGGING & FACIAL MESH WARP ──────────────────────────────────
    lo_landmarks_warped = lo_landmarks
    if lo_landmarks is not None:
        lo_landmarks_warped = _warp_landmarks_for_sagging(lo_landmarks, lo_image.shape[:2], intensity)
        lo_image = _apply_sagging_warp(lo_image, lo_landmarks, intensity)

    face_mask = _build_face_hair_mask(lo_image, landmarks=lo_landmarks_warped)          # float32 [0..1]
    face_mask_3 = face_mask[..., np.newaxis]           # (H,W,1) for BGR ops
    face_oval_mask = _build_face_oval_mask(lo_image, lo_landmarks_warped)
    feathered_face_mask = _build_feathered_face_mask(lo_image, lo_landmarks_warped)
    feathered_face_mask_3 = feathered_face_mask[..., np.newaxis]

    # ── 1. SKIN TEXTURE AGING (pore coarsening, translucency, pigmentation) ──
    if lo_landmarks_warped is not None and not fast_mode:
        lo_image = _apply_skin_texture_aging(lo_image, lo_landmarks_warped, face_oval_mask, intensity)

    # ── 2. MICRO WRINKLE & TEXTURE ENHANCEMENT ──────────────────────────────
    lab = cv2.cvtColor(lo_image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # High-pass frequency filter → fine skin detail
    scale_factor = w / 480.0
    radius = int((14 + intensity * 40) * scale_factor)
    high_pass = apply_frequency_filter(l_ch, radius=radius, mode="high")
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    # Strip broad muddy texture, keep only crisp detail
    detail_blur = cv2.GaussianBlur(high_pass, (0, 0), 1.0)
    detail = high_pass.astype(np.float32) - detail_blur.astype(np.float32)

    # CLAHE for local contrast — makes existing creases pop
    clip_limit = 2.2 + 3.2 * intensity
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_ch)

    # Build wrinkled luminance
    l_float = l_ch.astype(np.float32)
    detail_strength = 1.0 + 1.5 * intensity
    aged_l = l_float + detail * detail_strength

    # Blend in CLAHE version
    clahe_blend = 0.12 + 0.18 * intensity
    aged_l = aged_l * (1.0 - clahe_blend) + clahe_l.astype(np.float32) * clahe_blend

    # Mild contrast push + slight darkening
    contrast = 1.02 + 0.10 * intensity
    darkness = 2.0 + 5.0 * intensity
    aged_l = aged_l * contrast - darkness

    # Micro wrinkle noise (restricted to crepiness regions)
    noise = np.random.normal(0, 4 * intensity, l_ch.shape).astype(np.float32)
    crepiness_mask = _build_crepiness_mask(lo_image, lo_landmarks_warped)
    aged_l = aged_l + noise * crepiness_mask
    aged_l = np.clip(aged_l, 0, 255).astype(np.uint8)

    # Merge back with original colour channels
    aged_lab = cv2.merge([aged_l, a_ch, b_ch])
    wrinkled = cv2.cvtColor(aged_lab, cv2.COLOR_LAB2BGR)

    # Subtle sharpening to crisp up fine lines
    sigma_sharp = max(1.0, 1.0 * scale_factor)
    blurred = cv2.GaussianBlur(wrinkled, (0, 0), sigma_sharp)
    sharp_s = 0.10 + 0.15 * intensity
    wrinkled = cv2.addWeighted(wrinkled, 1.0 + sharp_s, blurred, -sharp_s, 0)

    # Blend with original to keep it natural (reduced to preserve wrinkles)
    blend_ratio = 0.10 + 0.15 * intensity
    wrinkled = cv2.addWeighted(lo_image, 1.0 - blend_ratio, wrinkled, blend_ratio, 0)

    # ★ Composite wrinkle effect onto original using face mask
    result = (
        lo_image.astype(np.float32) * (1.0 - face_mask_3)
        + wrinkled.astype(np.float32) * face_mask_3
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── 3. STRUCTURAL LOCALIZED WRINKLES ─────────────────────────────
    if lo_landmarks_warped is not None:
        result = _apply_structural_wrinkles(result, lo_landmarks_warped, face_mask, intensity, fast_mode=fast_mode)

    # ── 4. COLOR TRANSFER, LUT, PERIORBITAL DARKENING, LIP DESAT ─────
    result = _apply_color_transfer(result, feathered_face_mask, intensity, landmarks=lo_landmarks_warped)
    result = _apply_aging_lut(result, feathered_face_mask, intensity)
    if lo_landmarks_warped is not None:
        result = _apply_periorbital_darkening(result, lo_landmarks_warped, intensity)
        result = _apply_lip_desaturation(result, lo_landmarks_warped, intensity)

    # ── 5. SOLAR/AGE SPOTS (28 spots with color variation) ───────────
    if lo_landmarks_warped is not None and not fast_mode:
        result = _generate_solar_spots(result, lo_landmarks_warped, intensity)

    # ── 6. MULTI-REGION HAIR WHITENING ───────────────────────────────
    # 6a. Head hair (ML segmenter or HSV fallback)
    try:
        from modules.hair_module import _get_hair_mask
    except ModuleNotFoundError:
        from backend.modules.hair_module import _get_hair_mask

    raw_hair_mask = None
    try:
        raw_hair_mask = _get_hair_mask(lo_image)
    except Exception as e:
        logger.warning("ML hair segmenter failed in aging filter: %s", e)

    if raw_hair_mask is not None and cv2.countNonZero(raw_hair_mask) > 0:
        hair_mask = cv2.GaussianBlur(raw_hair_mask, (15, 15), 0).astype(np.float32) / 255.0
    else:
        # Fallback: robust HSV-based thresholding hair mask
        hsv = cv2.cvtColor(lo_image, cv2.COLOR_BGR2HSV)
        v_ch_hsv = hsv[:, :, 2].astype(np.float32)
        s_ch_hsv = hsv[:, :, 1].astype(np.float32)
        h_ch_hsv = hsv[:, :, 0].astype(np.float32)

        y_coords = np.linspace(0.0, 1.0, lh, dtype=np.float32).reshape(-1, 1)
        pos_weight = np.clip(1.0 - y_coords * 1.1, 0.12, 1.0)
        pos_weight = np.broadcast_to(pos_weight, (lh, lw)).copy()

        skin_region = (
            (h_ch_hsv >= 0) & (h_ch_hsv <= 30)
            & (s_ch_hsv > 30) & (v_ch_hsv > 70)
        )
        not_skin = 1.0 - skin_region.astype(np.float32)

        dark_thresh = 145 + int(50 * intensity)
        dark_mask = np.clip((dark_thresh - v_ch_hsv) / max(dark_thresh, 1), 0, 1)
        fallback_mask = dark_mask * pos_weight * not_skin * face_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        hair_u8 = (np.clip(fallback_mask, 0, 1) * 255).astype(np.uint8)
        hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_CLOSE, kernel)
        hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_OPEN, kernel)
        hair_mask = hair_u8.astype(np.float32) / 255.0
        hair_mask = cv2.GaussianBlur(hair_mask, (25, 25), 10)

    # Apply head hair whitening using ADAPTIVE blend based on pixel luminance.
    # Dark hair → soft-light blend (lifts toward gray/white).
    # Light/blonde hair → screen blend (pushes to pure white without muddy gray).
    # Silver tint (slightly cool/blue rather than pure gray)
    b_val = (210.0 + 45.0 * intensity) / 255.0
    g_val = (200.0 + 55.0 * intensity) / 255.0
    r_val = (195.0 + 60.0 * intensity) / 255.0
    color_layer = np.full_like(result, (b_val, g_val, r_val), dtype=np.float32)
    base = result.astype(np.float32) / 255.0

    # Compute per-pixel luminance to decide blend mode
    lum = np.mean(base, axis=2, keepdims=True)  # (H,W,1)
    lum_broadcast = np.broadcast_to(lum, base.shape)  # (H,W,3)

    # Soft-light blend (good for dark hair)
    soft_light = np.where(
        color_layer <= 0.5,
        base - (1.0 - 2.0 * color_layer) * base * (1.0 - base),
        base + (2.0 * color_layer - 1.0) * (np.sqrt(np.clip(base, 1e-6, 1.0)) - base)
    )

    # Screen blend (good for light/blonde hair — always brightens toward white)
    screen = 1.0 - (1.0 - base) * (1.0 - color_layer)

    # Adaptive mix: use screen for light pixels (lum > 0.5), soft-light for dark
    blend_t = np.clip((lum_broadcast - 0.35) / 0.30, 0.0, 1.0)  # smooth ramp 0.35..0.65
    adaptive_result = soft_light * (1.0 - blend_t) + screen * blend_t

    colored_hair = np.clip(adaptive_result * 255.0, 0, 255).astype(np.uint8)

    # Reduce blur to prevent massive halo bleeding into background
    soft_hair_mask = cv2.GaussianBlur(hair_mask, (11, 11), 0).astype(np.float32)
    
    # Restrict soft mask from spilling outside the actual hair area too much
    soft_hair_mask = soft_hair_mask * np.clip(hair_mask * 1.5, 0.0, 1.0)
    
    soft_hair_mask_3ch = np.stack([soft_hair_mask] * 3, axis=-1)

    blend_strength = (0.45 + 0.55 * intensity) * soft_hair_mask_3ch
    blend_strength = np.clip(blend_strength, 0.0, 1.0)
    result = (
        colored_hair.astype(np.float32) * blend_strength
        + result.astype(np.float32) * (1.0 - blend_strength)
    ).astype(np.uint8)

    # 6b. Eyebrow whitening (landmark-based mask, 60% strength)
    if lo_landmarks_warped is not None:
        brow_mask = _build_eyebrow_mask(lo_image, lo_landmarks_warped)
        if np.max(brow_mask) > 0.01:
            # Silver tint for eyebrows
            brow_b = (215.0 + 40.0 * intensity) / 255.0
            brow_g = (205.0 + 50.0 * intensity) / 255.0
            brow_r = (200.0 + 55.0 * intensity) / 255.0
            brow_color = np.full_like(result, (brow_b, brow_g, brow_r), dtype=np.float32)
            base_b = result.astype(np.float32) / 255.0
            # Adaptive blend for eyebrows (screen for light brows, soft-light for dark)
            brow_lum = np.mean(base_b, axis=2, keepdims=True)
            brow_lum_bc = np.broadcast_to(brow_lum, base_b.shape)
            brow_soft = np.where(
                brow_color <= 0.5,
                base_b - (1.0 - 2.0 * brow_color) * base_b * (1.0 - base_b),
                base_b + (2.0 * brow_color - 1.0) * (np.sqrt(np.clip(base_b, 1e-6, 1.0)) - base_b)
            )
            brow_screen = 1.0 - (1.0 - base_b) * (1.0 - brow_color)
            brow_t = np.clip((brow_lum_bc - 0.35) / 0.30, 0.0, 1.0)
            brow_blend = brow_soft * (1.0 - brow_t) + brow_screen * brow_t
            brow_result = np.clip(brow_blend * 255.0, 0, 255).astype(np.uint8)

            brow_mask_3 = np.stack([brow_mask] * 3, axis=-1)
            brow_strength = 0.80 * intensity * brow_mask_3
            result = (
                brow_result.astype(np.float32) * brow_strength
                + result.astype(np.float32) * (1.0 - brow_strength)
            ).astype(np.uint8)

    # 6c. Facial hair / stubble whitening (salt-and-pepper effect on dark pixels)
    if lo_landmarks_warped is not None:
        lower_mask = _build_lower_face_hair_mask(lo_image, lo_landmarks_warped)
        if np.max(lower_mask) > 0.01:
            hsv_face = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            v_channel = hsv_face[:, :, 2].astype(np.float32)

            # Only affect dark pixels (likely facial hair / stubble)
            dark_pixel_mask = np.clip((120.0 - v_channel) / 80.0, 0, 1)
            # Restrict to face mask to strictly prevent ghost beard on dark background
            stubble_mask = lower_mask * dark_pixel_mask * face_mask

            # Salt-and-pepper: random per-pixel decision
            sp_noise = np.random.random((lh, lw)).astype(np.float32)
            sp_threshold = 0.3 + 0.4 * intensity
            sp_mask = (sp_noise < sp_threshold).astype(np.float32)

            final_stubble_mask = stubble_mask * sp_mask * intensity * 0.70
            final_stubble_mask_3 = np.stack([final_stubble_mask] * 3, axis=-1)

            white_layer = np.full_like(result, 210, dtype=np.float32)
            result = (
                white_layer * final_stubble_mask_3
                + result.astype(np.float32) * (1.0 - final_stubble_mask_3)
            ).astype(np.uint8)

    # ── 7. FINAL COMPOSITING & TINTING (face only) ──────────────────
    # Build tinted version
    lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Milder final tint to keep it pale and bright
    # Very subtle tint — avoid jaundice (reduced from 0.8 to 0.15)
    lab_out[:, :, 2] = np.clip(lab_out[:, :, 2] + 0.15 * intensity, 0, 255)
    lab_out[:, :, 0] = np.clip(lab_out[:, :, 0] - 0.1 * intensity, 0, 255)
    tinted = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ★ Composite tint onto result using feathered face mask (no tint on hair/clothes)
    result = (
        result.astype(np.float32) * (1.0 - feathered_face_mask_3)
        + tinted.astype(np.float32) * feathered_face_mask_3
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    if ds_factor > 1.0:
        # Upscale aged branch and mask, then blend back onto original hi-res.
        up_result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        up_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        up_mask = np.clip(up_mask, 0.0, 1.0)[..., None]
        merged = image.astype(np.float32) * (1.0 - up_mask) + up_result.astype(np.float32) * up_mask
        return np.clip(merged, 0, 255).astype(np.uint8)
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


def apply_aging(image: np.ndarray, intensity: float, landmarks: np.ndarray = None) -> np.ndarray:
    strength = normalize_strength(intensity)
    return apply_aging_filter(image, intensity=strength, landmarks=landmarks)


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


def _radial_grid(shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    rows, cols = shape
    cy, cx = rows / 2.0, cols / 2.0
    yy, xx = np.ogrid[:rows, :cols]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return radius, float(radius.max())


def _annular_bounds_from_band(band: str, max_radius: float) -> tuple[float, float]:
    band = (band or "mid").strip().lower()
    if band in {"low", "center", "low_frequency"}:
        return 0.0, max_radius * 0.18
    if band in {"mid", "middle", "medium", "mid_frequency"}:
        return max_radius * 0.18, max_radius * 0.42
    if band in {"high", "outer", "high_frequency"}:
        return max_radius * 0.42, max_radius * 0.98
    raise ValueError("fft_band must be 'low', 'mid', or 'high'.")


def _annular_bounds_from_coords(
    coords: dict,
    rows: int,
    cols: int,
    max_radius: float,
) -> tuple[float, float]:
    """
    Interpret a frontend spectrum selection as radial frequency distance.

    The old implementation treated x/y as a rectangular image patch. In shifted
    FFT space that is misleading: equal distances from the center represent the
    same frequency magnitude. This converts the selection center/size into an
    annulus and clamps very outer corner selections to avoid invalid components.
    """
    try:
        x = clamp(float(coords.get("x", 0.5)))
        y = clamp(float(coords.get("y", 0.5)))
        w = clamp(float(coords.get("w", coords.get("width", 0.08))))
        h = clamp(float(coords.get("h", coords.get("height", 0.08))))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("mask_coords must include numeric x, y, w, h values.") from exc

    cx = cols / 2.0
    cy = rows / 2.0
    sel_cx = (x + w / 2.0) * cols
    sel_cy = (y + h / 2.0) * rows
    selected_radius = float(np.hypot(sel_cx - cx, sel_cy - cy))

    if selected_radius > max_radius * 0.98:
        raise ValueError("Selection is outside the valid shifted FFT frequency disk.")

    thickness = max(max(rows, cols) * max(w, h) * 0.45, max_radius * 0.045)
    inner = max(0.0, selected_radius - thickness)
    outer = min(max_radius * 0.98, selected_radius + thickness)
    if outer - inner < max_radius * 0.025:
        raise ValueError("Selected FFT annulus is too small.")
    return inner, outer


def build_annular_fft_mask(
    shape: tuple[int, int],
    band: str = "mid",
    coords: dict | None = None,
    feather: float = 2.5,
) -> tuple[np.ndarray, tuple[float, float]]:
    radius, max_radius = _radial_grid(shape)
    if coords:
        inner, outer = _annular_bounds_from_coords(coords, shape[0], shape[1], max_radius)
    else:
        inner, outer = _annular_bounds_from_band(band, max_radius)

    mask = ((radius >= inner) & (radius <= outer)).astype(np.float32)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        max_value = float(mask.max())
        if max_value > 0:
            mask /= max_value
    return np.clip(mask, 0.0, 1.0), (inner, outer)


def overlay_fft_mask(spectrum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(spectrum.shape) == 2:
        base = cv2.cvtColor(spectrum, cv2.COLOR_GRAY2BGR)
    else:
        base = spectrum.copy()

    overlay = base.astype(np.float32)
    mask_f = np.clip(mask.astype(np.float32), 0.0, 1.0)
    tint = np.zeros_like(overlay)
    tint[:, :, 0] = 210.0
    tint[:, :, 1] = 240.0
    tint[:, :, 2] = 80.0
    overlay = overlay * (1.0 - mask_f[..., None] * 0.26) + tint * (mask_f[..., None] * 0.26)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_center_corner_fft_mask(
    shape: tuple[int, int],
    center_ratio: float = 0.13,
    corner_ratio: float = 0.16,
    feather: float = 2.0,
) -> np.ndarray:
    """
    Build a fixed shifted-FFT laboratory mask.

    The shifted FFT places low frequencies in the center. The four corners
    represent high-frequency regions, which correspond to sharp spatial changes.
    This mask intentionally avoids arbitrary magnitude picking: it keeps only
    the center region and all four corner regions.
    """
    rows, cols = shape
    yy, xx = np.ogrid[:rows, :cols]
    base = float(min(rows, cols))
    center_radius = max(3.0, base * center_ratio)
    corner_radius = max(3.0, base * corner_ratio)

    mask = np.zeros((rows, cols), dtype=np.float32)
    centers = [
        (cols / 2.0, rows / 2.0),
        (0.0, 0.0),
        (cols - 1.0, 0.0),
        (0.0, rows - 1.0),
        (cols - 1.0, rows - 1.0),
    ]

    for idx, (cx, cy) in enumerate(centers):
        radius = center_radius if idx == 0 else corner_radius
        region = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        mask[region] = 1.0

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        max_value = float(mask.max())
        if max_value > 0:
            mask /= max_value

    return np.clip(mask, 0.0, 1.0)


def apply_fft_center_corner_inverse(image: np.ndarray) -> dict:
    """
    Keep the center and four FFT corners, then reconstruct one image with IFFT.

    Returns the original FFT magnitude with the fixed mask overlay, the masked
    FFT magnitude, and the inverse FFT visualization.
    """
    if image is None:
        raise ValueError("Input image is None.")

    rows, cols = image.shape[:2]
    mask = build_center_corner_fft_mask((rows, cols))
    working = image.astype(np.float32)
    reconstructed = np.zeros_like(working, dtype=np.float32)

    selected_power = np.zeros((rows, cols), dtype=np.float32)
    for ch in range(3):
        fft_shifted = np.fft.fftshift(np.fft.fft2(working[:, :, ch]))
        selected_fft = fft_shifted * mask
        selected_power += np.log1p(np.abs(selected_fft)).astype(np.float32)
        restored = np.real(np.fft.ifft2(np.fft.ifftshift(selected_fft))).astype(np.float32)
        reconstructed[:, :, ch] = restored

    # IFFT components can be signed; normalize per channel for a clear visual.
    result = reconstructed - reconstructed.min(axis=(0, 1), keepdims=True)
    denom = np.maximum(result.max(axis=(0, 1), keepdims=True), 1e-6)
    result = np.clip((result / denom) * 255.0, 0, 255).astype(np.uint8)

    orig_spectrum = compute_magnitude_spectrum(compute_fft(image)[2])
    selected_spectrum = cv2.normalize(selected_power, None, 0, 255, cv2.NORM_MINMAX)
    selected_spectrum = np.clip(selected_spectrum, 0, 255).astype(np.uint8)

    return {
        "processed": result,
        "orig_spectrum": overlay_fft_mask(orig_spectrum, mask),
        "proc_spectrum": overlay_fft_mask(selected_spectrum, mask),
        "mask": mask,
        "band": "center_corners",
    }


def apply_fft_annular_filter(
    image: np.ndarray,
    intensity: float = 50,
    band: str = "mid",
    mask_coords: dict | None = None,
) -> dict:
    """
    Manipulate low/mid/high radial frequency bands with a centered annular mask.

    The selected band is attenuated in shifted FFT space and reconstructed with
    inverse FFT. This makes the lab demonstrate meaningful frequency bands:
    center = low frequencies, middle ring = mid frequencies, outer ring = high.
    """
    if image is None:
        raise ValueError("Input image is None.")

    strength = normalize_strength(intensity)
    rows, cols = image.shape[:2]
    mask, bounds = build_annular_fft_mask((rows, cols), band=band, coords=mask_coords)
    attenuation = 1.0 - (0.15 + 0.85 * strength) * mask

    working = image.astype(np.float32)
    processed = np.zeros_like(working, dtype=np.float32)

    for ch in range(3):
        fft_shifted = np.fft.fftshift(np.fft.fft2(working[:, :, ch]))
        manipulated = fft_shifted * attenuation
        restored = np.real(np.fft.ifft2(np.fft.ifftshift(manipulated)))
        processed[:, :, ch] = restored

    processed_u8 = np.clip(processed, 0, 255).astype(np.uint8)
    difference = cv2.absdiff(image, processed_u8)
    difference = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

    orig_fft_shifted = compute_fft(image)[2]
    proc_fft_shifted = compute_fft(processed_u8)[2]
    orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
    proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

    return {
        "processed": processed_u8,
        "difference": np.clip(difference, 0, 255).astype(np.uint8),
        "orig_spectrum": cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR),
        "proc_spectrum": overlay_fft_mask(proc_spectrum, mask),
        "mask": mask,
        "bounds": bounds,
        "band": band,
    }


def _coords_to_fft_rect(coords: dict, rows: int, cols: int) -> tuple[int, int, int, int]:
    """
    Convert normalized frontend selection coordinates into FFT pixel bounds.
    """
    try:
        x = float(coords.get("x", 0.0))
        y = float(coords.get("y", 0.0))
        width = float(coords.get("w", coords.get("width", 0.0)))
        height = float(coords.get("h", coords.get("height", 0.0)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("mask_coords must include numeric x, y, w, h values.") from exc

    x0 = int(round(clamp(x) * cols))
    y0 = int(round(clamp(y) * rows))
    x1 = int(round(clamp(x + width) * cols))
    y1 = int(round(clamp(y + height) * rows))

    x0, x1 = sorted((max(0, x0), min(cols, x1)))
    y0, y1 = sorted((max(0, y0), min(rows, y1)))

    if x1 - x0 < 3 or y1 - y0 < 3:
        raise ValueError("Selected FFT region is too small.")

    return x0, y0, x1, y1


def _build_symmetric_fft_patch_mask(
    shape: tuple[int, int],
    coords: dict,
    feather: float = 3.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Build a shifted-FFT mask for the selected patch plus its conjugate mirror.

    Keeping the mirrored region makes the inverse transform a stable real-valued
    artifact instead of a one-sided complex component.
    """
    rows, cols = shape
    x0, y0, x1, y1 = _coords_to_fft_rect(coords, rows, cols)

    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0

    mirror_x0 = max(0, cols - x1)
    mirror_x1 = min(cols, cols - x0)
    mirror_y0 = max(0, rows - y1)
    mirror_y1 = min(rows, rows - y0)
    if mirror_x1 > mirror_x0 and mirror_y1 > mirror_y0:
        mask[mirror_y0:mirror_y1, mirror_x0:mirror_x1] = 1.0

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        max_value = float(mask.max())
        if max_value > 0:
            mask /= max_value

    return np.clip(mask, 0.0, 1.0), (x0, y0, x1, y1)


def apply_fft_partial_region_artifact(
    image: np.ndarray,
    mask_coords: dict,
    intensity: float = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolate an off-center FFT magnitude patch, run IFFT only from that patch,
    and add the resulting spatial artifact back onto the original image.
    """
    if image is None:
        raise ValueError("Input image is None.")

    strength = normalize_strength(intensity)
    rows, cols = image.shape[:2]
    patch_mask, (x0, y0, x1, y1) = _build_symmetric_fft_patch_mask((rows, cols), mask_coords)

    working = image.astype(np.float32)
    artifact = np.zeros_like(working, dtype=np.float32)

    for ch in range(3):
        channel = working[:, :, ch]
        fft_shifted = np.fft.fftshift(np.fft.fft2(channel))
        isolated = fft_shifted * patch_mask
        component = np.real(np.fft.ifft2(np.fft.ifftshift(isolated))).astype(np.float32)

        scale = float(np.percentile(np.abs(component), 98))
        if scale < 1e-6:
            continue
        artifact[:, :, ch] = np.clip(component / scale, -1.0, 1.0)

    patch_cx = ((x0 + x1) * 0.5) - cols * 0.5
    patch_cy = ((y0 + y1) * 0.5) - rows * 0.5
    off_y = abs(((y0 + y1) * 0.5) - rows * 0.5) / max(rows * 0.5, 1.0)
    off_x = abs(((x0 + x1) * 0.5) - cols * 0.5) / max(cols * 0.5, 1.0)
    distance_boost = 0.75 + max(off_x, off_y) * 0.65
    artifact_gain = (34.0 + 116.0 * strength) * distance_boost

    processed = working + artifact * artifact_gain

    # A faint directional carrier makes different off-center patches easier to compare.
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    carrier_phase = (
        (patch_cx / max(cols, 1)) * xx
        + (patch_cy / max(rows, 1)) * yy
    ) * np.pi * 2.0
    carrier = np.sin(carrier_phase)
    carrier *= (2.0 + 5.0 * strength) * np.clip(patch_mask.max(), 0.0, 1.0)
    processed += carrier[..., None]

    processed = np.clip(processed, 0, 255).astype(np.uint8)

    artifact_vis = cv2.normalize(artifact, None, 0, 255, cv2.NORM_MINMAX)
    artifact_vis = np.clip(artifact_vis, 0, 255).astype(np.uint8)
    return processed, artifact_vis


def apply_fft_selected_region_inverse(
    image: np.ndarray,
    mask_coords: dict,
) -> dict:
    """
    Select FFT coefficients, keep their conjugate-symmetric counterpart, and
    reconstruct only that frequency component with inverse FFT.

    This is the FFT Laboratory path: it does not add an artificial effect to the
    source image. It shows what the selected frequency region contributes when
    transformed back into image space.
    """
    if image is None:
        raise ValueError("Input image is None.")

    rows, cols = image.shape[:2]
    patch_mask, rect = _build_symmetric_fft_patch_mask((rows, cols), mask_coords, feather=1.25)

    working = image.astype(np.float32)
    component = np.zeros_like(working, dtype=np.float32)

    for ch in range(3):
        fft_shifted = np.fft.fftshift(np.fft.fft2(working[:, :, ch]))
        isolated = fft_shifted * patch_mask
        restored = np.real(np.fft.ifft2(np.fft.ifftshift(isolated))).astype(np.float32)
        component[:, :, ch] = restored

    # Signed components are not displayable directly; normalize for visualization.
    component_vis = component - component.min(axis=(0, 1), keepdims=True)
    denom = component_vis.max(axis=(0, 1), keepdims=True)
    component_vis = np.divide(component_vis, np.maximum(denom, 1e-6)) * 255.0
    component_vis = np.clip(component_vis, 0, 255).astype(np.uint8)

    energy = np.linalg.norm(component, axis=2)
    energy_vis = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)
    energy_vis = cv2.cvtColor(np.clip(energy_vis, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    orig_spectrum = compute_magnitude_spectrum(compute_fft(image)[2])
    selected_spectrum = cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR).astype(np.float32)
    mask_3 = patch_mask[..., None]
    selected_spectrum *= (0.18 + 0.82 * mask_3)
    tint = np.zeros_like(selected_spectrum)
    tint[:, :, 1] = 230.0
    tint[:, :, 2] = 255.0
    selected_spectrum = selected_spectrum * (1.0 - mask_3 * 0.35) + tint * (mask_3 * 0.35)

    return {
        "processed": component_vis,
        "difference": energy_vis,
        "orig_spectrum": overlay_fft_mask(orig_spectrum, patch_mask),
        "proc_spectrum": np.clip(selected_spectrum, 0, 255).astype(np.uint8),
        "mask": patch_mask,
        "rect": rect,
        "band": "selection",
    }


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
        if idx >= len(landmarks):
            continue

        lm = landmarks[idx]

        if isinstance(lm, dict):
            x = int(float(lm["x"]) * width)
            y = int(float(lm["y"]) * height)
        else:
            x = int(float(lm[0]) * width)
            y = int(float(lm[1]) * height)

        points.append([
            int(np.clip(x, 0, max(width - 1, 0))),
            int(np.clip(y, 0, max(height - 1, 0))),
        ])

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

    # Hedef rengin (hue) orijinal ten rengiyle karışıp renk tekerleğinde kaymasını
    # (örneğin morun maviye dönmesini) önlemek için hue'yu doğrudan eşitliyoruz.
    # Yumuşak geçiş zaten aşağıdaki BGR alpha-blend (soft_mask) ile sağlanır.
    hsv[:, :, 0][mask_bool] = hue
    
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


def _add_eye_line_mask(
    mask: np.ndarray,
    points: np.ndarray,
    thickness: int,
    wing: bool = False,
) -> None:
    if len(points) < 2:
        return
    pts = points.astype(np.int32)
    cv2.polylines(mask, [pts], False, 1.0, thickness, cv2.LINE_AA)
    if wing and len(pts) >= 2:
        p0 = pts[0]
        p1 = pts[1]
        direction = p0 - p1
        norm = float(np.linalg.norm(direction))
        if norm > 1.0:
            wing_tip = p0 + (direction / norm * thickness * 2.4).astype(np.int32)
            cv2.line(mask, tuple(p0), tuple(wing_tip), 1.0, max(1, thickness - 1), cv2.LINE_AA)


def _apply_dark_mask(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float,
    blur_sigma: float,
    color_bgr: tuple[int, int, int] = (8, 8, 10),
) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    mask_float = mask.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0
    soft_mask = np.clip(mask_float, 0.0, 1.0)
    if blur_sigma > 0:
        soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), blur_sigma)
    max_value = float(soft_mask.max())
    if max_value > 0:
        soft_mask /= max_value
    soft_mask = soft_mask[..., None] * opacity
    dark = np.full_like(image, color_bgr, dtype=np.float32)
    result = image.astype(np.float32) * (1.0 - soft_mask) + dark * soft_mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _line_color_from_hue(hue: int) -> tuple[int, int, int]:
    hue = int(np.clip(hue, 0, 179))
    hsv = np.uint8([[[hue, 185, 68]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_tapered_lash(
    mask: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    thickness: int,
    curve: float,
) -> None:
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        return
    direction = direction / norm
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    p0 = start.astype(np.float32)
    p1 = p0 + direction * length * 0.55 + perp * curve
    p2 = p0 + direction * length + perp * curve * 1.35

    prev = p0
    steps = 8
    for i in range(1, steps + 1):
        t = i / steps
        pt = ((1.0 - t) ** 2 * p0) + (2.0 * (1.0 - t) * t * p1) + (t ** 2 * p2)
        local_thickness = max(1, int(round(thickness * (1.0 - t * 0.80))))
        alpha = max(0.18, 0.82 - t * 0.58)
        cv2.line(mask, tuple(prev.astype(int)), tuple(pt.astype(int)), alpha, local_thickness, cv2.LINE_AA)
        prev = pt


def _sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.float32)
    pts = points.astype(np.float32)
    segs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    total = float(seg_lens.sum())
    if total < 1.0:
        return pts

    samples = []
    distances = np.linspace(0.0, total, count)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    for dist in distances:
        seg_idx = int(np.searchsorted(cumulative, dist, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(segs) - 1)
        local = (dist - cumulative[seg_idx]) / max(seg_lens[seg_idx], 1e-6)
        samples.append(pts[seg_idx] + segs[seg_idx] * local)
    return np.array(samples, dtype=np.float32)


def apply_virtual_makeup(
    image: np.ndarray,
    landmarks: list = None,
    region: str = "lip",
    hue: int = 0,
    opacity: float = 0.5,
) -> np.ndarray:
    """
    Virtual makeup using landmark masks + HSV color manipulation + alpha blending.

    Realtime uyumluluğu: landmarks parametresi dışarıdan verilirse doğrudan
    kullanılır (PersistentFaceMesh + EMA), verilmezse detect_face_landmarks
    ile algılanır.

    region:
    - lip
    - blush
    - eyeshadow
    - eyeliner
    - mascara
    """
    if image is None:
        raise ValueError("Input image is None.")

    # ── Landmark çözümleme (tek seferlik) ─────────────────────────────
    if landmarks is None:
        try:
            from modules.warping_module import detect_face_landmarks
        except ModuleNotFoundError:
            from backend.modules.warping_module import detect_face_landmarks
        lm = detect_face_landmarks(image)
        if lm is None:
            raise ValueError("Landmarks could not be detected.")
        # detect_face_landmarks pixel coords (N,2) → normalize to [0,1]
        h_img, w_img = image.shape[:2]
        landmarks = [[float(pt[0]) / w_img, float(pt[1]) / h_img] for pt in lm]

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
        if len(outer_points) < 3:
            raise ValueError("Not enough lip landmarks were detected.")
        cv2.fillPoly(mask, [outer_points], 1.0)
        if len(inner_points) >= 3:
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

        saturation_multiplier = 1.65
        blur_sigma = max(3.0, min(h, w) * 0.008)
        opacity = min(opacity * 1.6, 0.95)
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

        saturation_multiplier = 1.35
        blur_sigma = max(5.0, min(h, w) * 0.012)
        opacity = min(opacity * 1.5, 0.65)
        normalize_mask = False

    elif region == "eyeliner":
        left_upper = [33, 246, 161, 160, 159, 158, 157, 173, 133]
        right_upper = [362, 398, 384, 385, 386, 387, 388, 466, 263]
        left_lower = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        right_lower = [362, 382, 381, 380, 374, 373, 390, 249, 263]

        thickness = max(1, int(min(h, w) * 0.006))
        for idxs, wing in ((left_upper, True), (right_upper, True), (left_lower, False), (right_lower, False)):
            pts = _normalized_landmarks_to_points(landmarks, idxs, w, h)
            _add_eye_line_mask(mask, pts, thickness, wing=wing)
        return _apply_dark_mask(
            image=image,
            mask=mask,
            opacity=min(opacity * 1.45, 0.90),
            blur_sigma=max(0.7, min(h, w) * 0.0022),
            color_bgr=_line_color_from_hue(hue),
        )

    elif region == "mascara":
        lash_groups = [
            [33, 246, 161, 160, 159, 158, 157, 173, 133],
            [362, 398, 384, 385, 386, 387, 388, 466, 263],
        ]
        thickness = 1
        base_lash_len = max(4.0, min(h, w) * 0.018)
        for eye_i, idxs in enumerate(lash_groups):
            pts = _normalized_landmarks_to_points(landmarks, idxs, w, h)
            if len(pts) < 3:
                continue
            center = pts.mean(axis=0)
            eye_width = max(float(np.ptp(pts[:, 0])), 1.0)
            samples = _sample_polyline(pts, max(21, int(eye_width / 3.2)))
            for lash_i, pt in enumerate(samples[1:-1], start=1):
                lateral = (pt[0] - center[0]) / eye_width
                up_direction = pt - center
                up_direction[1] -= max(2.0, min(h, w) * 0.018)
                outer_boost = 0.64 + 0.24 * min(abs(lateral) * 2.0, 1.0)
                natural_jitter = 0.90 + 0.08 * ((lash_i % 4) - 1.5)
                length = base_lash_len * outer_boost * natural_jitter
                curve = base_lash_len * lateral * (0.10 if eye_i == 0 else -0.10)
                _draw_tapered_lash(mask, pt, up_direction, length, thickness, curve)

            lid_mask = np.zeros((h, w), dtype=np.float32)
            _add_eye_line_mask(lid_mask, pts, max(1, thickness), wing=False)
            mask[:] = np.maximum(mask, lid_mask * 0.12)
        return _apply_dark_mask(
            image=image,
            mask=mask,
            opacity=min(opacity * 0.86, 0.58),
            blur_sigma=max(0.10, min(h, w) * 0.00035),
            color_bgr=_line_color_from_hue(hue),
        )

    else:
        raise ValueError("Region must be 'lip', 'blush', 'eyeshadow', 'eyeliner', or 'mascara'.")

    return _apply_color_with_mask(
        image=image,
        mask=mask,
        hue=hue,
        opacity=opacity,
        saturation_multiplier=saturation_multiplier,
        blur_sigma=blur_sigma,
        normalize_mask=normalize_mask,
    )


def _estimate_face_box(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(45, 45))
    if len(faces) > 0:
        x, y, fw, fh = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
        return int(x), int(y), int(fw), int(fh)

    fw = int(w * 0.46)
    fh = int(h * 0.58)
    x = (w - fw) // 2
    y = int(h * 0.18)
    return x, y, fw, fh


def apply_virtual_makeup_fallback(
    image: np.ndarray,
    region: str = "lip",
    hue: int = 0,
    opacity: float = 0.5,
) -> np.ndarray:
    """
    Approximate makeup masks for camera frames when landmark detection fails.
    """
    if image is None:
        raise ValueError("Input image is None.")

    h, w = image.shape[:2]
    x, y, fw, fh = _estimate_face_box(image)
    region = (region or "").strip().lower()
    mask = np.zeros((h, w), dtype=np.float32)

    if region in {"lip", "lips"}:
        center = (x + fw // 2, y + int(fh * 0.72))
        axes = (max(8, int(fw * 0.17)), max(4, int(fh * 0.035)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        saturation_multiplier = 1.45
        blur_sigma = max(1.5, min(h, w) * 0.005)
        opacity = min(opacity, 0.70)

    elif region == "eyeshadow":
        eye_y = y + int(fh * 0.39)
        eye_dx = int(fw * 0.18)
        eye_axes = (max(8, int(fw * 0.13)), max(5, int(fh * 0.045)))
        cv2.ellipse(mask, (x + fw // 2 - eye_dx, eye_y), eye_axes, -6, 0, 360, 1.0, -1)
        cv2.ellipse(mask, (x + fw // 2 + eye_dx, eye_y), eye_axes, 6, 0, 360, 1.0, -1)
        saturation_multiplier = 1.50
        blur_sigma = max(2.5, min(h, w) * 0.010)
        opacity = min(opacity, 0.65)

    elif region == "blush":
        cheek_y = y + int(fh * 0.57)
        cheek_dx = int(fw * 0.23)
        cheek_axes = (max(10, int(fw * 0.13)), max(8, int(fh * 0.07)))
        cv2.ellipse(mask, (x + fw // 2 - cheek_dx, cheek_y), cheek_axes, -12, 0, 360, 1.0, -1)
        cv2.ellipse(mask, (x + fw // 2 + cheek_dx, cheek_y), cheek_axes, 12, 0, 360, 1.0, -1)
        saturation_multiplier = 1.45
        blur_sigma = max(3.0, min(h, w) * 0.014)
        opacity = min(opacity, 0.58)

    elif region == "eyeliner":
        eye_y = y + int(fh * 0.39)
        eye_dx = int(fw * 0.18)
        eye_w = max(12, int(fw * 0.16))
        thickness = max(1, int(min(h, w) * 0.006))
        cv2.line(mask, (x + fw // 2 - eye_dx - eye_w, eye_y), (x + fw // 2 - eye_dx + eye_w, eye_y), 1.0, thickness, cv2.LINE_AA)
        cv2.line(mask, (x + fw // 2 + eye_dx - eye_w, eye_y), (x + fw // 2 + eye_dx + eye_w, eye_y), 1.0, thickness, cv2.LINE_AA)
        return _apply_dark_mask(image, mask, min(opacity * 1.35, 0.85), max(0.8, min(h, w) * 0.0025), _line_color_from_hue(hue))

    elif region == "mascara":
        eye_y = y + int(fh * 0.37)
        eye_dx = int(fw * 0.18)
        eye_w = max(10, int(fw * 0.14))
        lash_len = max(3, int(fh * 0.023))
        thickness = 1
        for cx in (x + fw // 2 - eye_dx, x + fw // 2 + eye_dx):
            for i, offset in enumerate(np.linspace(-eye_w, eye_w, 23)):
                px = int(cx + offset)
                curve = float(offset) * 0.025
                local_len = lash_len * (0.76 + 0.08 * (i % 4))
                _draw_tapered_lash(
                    mask,
                    np.array([px, eye_y], dtype=np.float32),
                    np.array([curve, -local_len], dtype=np.float32),
                    local_len,
                    thickness,
                    curve,
                )
        return _apply_dark_mask(image, mask, min(opacity * 0.86, 0.58), max(0.10, min(h, w) * 0.00035), _line_color_from_hue(hue))

    else:
        raise ValueError("Region must be 'lip', 'blush', 'eyeshadow', 'eyeliner', or 'mascara'.")

    return _apply_color_with_mask(
        image=image,
        mask=mask,
        hue=hue,
        opacity=opacity,
        saturation_multiplier=saturation_multiplier,
        blur_sigma=blur_sigma,
        normalize_mask=True,
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
