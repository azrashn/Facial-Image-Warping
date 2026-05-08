import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

try:
    from modules.frequency_module import (
        apply_aging,
        apply_deaging,
        apply_fft_filter,
        apply_cartoon_filter,
        apply_virtual_makeup,
        create_face_region_mask,
        blend_effect_with_mask,
        compute_energy_analysis,
        compute_magnitude_spectrum,
        compute_fft,
        encode_image_to_base64,
    )
    from modules.input_module import get_landmarks, preprocess_image
    from modules.metrics_module import compute_mse, compute_psnr, compute_ssim
    from modules.warping_module import (
        apply_eyebrow_raise,
        apply_eye_scaling,
        apply_face_slim,
        apply_lip_widen,
        apply_smile,
        detect_face_landmarks,
        geometric_warp,
        _corners,
        _gaussian_falloff,
        _face_scale,
        _prepare_warp,
    )
except ModuleNotFoundError:
    from backend.modules.frequency_module import (
        apply_aging,
        apply_deaging,
        apply_fft_filter,
        apply_cartoon_filter,
        apply_virtual_makeup,
        create_face_region_mask,
        blend_effect_with_mask,
        compute_energy_analysis,
        compute_magnitude_spectrum,
        compute_fft,
        encode_image_to_base64,
    )
    from backend.modules.input_module import get_landmarks, preprocess_image
    from backend.modules.metrics_module import compute_mse, compute_psnr, compute_ssim
    from backend.modules.warping_module import (
        apply_eyebrow_raise,
        apply_eye_scaling,
        apply_face_slim,
        apply_lip_widen,
        apply_smile,
        detect_face_landmarks,
        geometric_warp,
        _corners,
        _gaussian_falloff,
        _face_scale,
        _prepare_warp,
    )

router = APIRouter()
logger = logging.getLogger("facial_pipeline.process")

WARP_OPS = {"smile", "eyebrow", "lip", "slim"}
AGE_OPS = {"aging", "deaging", "age", "deage", "fft"}


def _hex_color_to_hue(color: str | None, fallback: int = 0) -> int:
    if not color:
        return fallback

    value = color.strip().lstrip("#")
    if len(value) != 6:
        return fallback

    try:
        rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback

    bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
    return int(hsv_pixel[0, 0, 0])


def _decode_upload(contents: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _data_url_from_image(image: np.ndarray) -> str:
    return f"data:image/png;base64,{encode_image_to_base64(image)}"


def _metrics_dict(original: np.ndarray, processed: np.ndarray) -> dict:
    return {
        "mse": float(compute_mse(original, processed)["mse"]),
        "psnr": float(compute_psnr(original, processed)["psnr"]),
        "ssim": float(compute_ssim(original, processed)["ssim"]),
    }


def _response_payload(
    image_b64: str,
    metrics: dict,
    orig_spectrum_b64: str | None = None,
    proc_spectrum_b64: str | None = None,
    orig_phase_b64: str | None = None,
    proc_phase_b64: str | None = None,
    energy: dict | None = None,
) -> dict:
    return {
        "image_b64": image_b64,
        "metrics": metrics,
        "orig_spectrum_b64": orig_spectrum_b64,
        "proc_spectrum_b64": proc_spectrum_b64,
        "orig_phase_b64": orig_phase_b64,
        "proc_phase_b64": proc_phase_b64,
        "energy": energy,
    }

def _compute_phase_b64(fft_shifted: np.ndarray) -> str:
    phase = np.angle(fft_shifted)
    phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return _data_url_from_image(cv2.cvtColor(phase_normalized, cv2.COLOR_GRAY2BGR))


@router.post("/process/warp")
async def process_warp(
    image: UploadFile = File(...),
    operation: str = Form(...),
    intensity: float = Form(50),
    smoothing: float = Form(30),
):
    op = (operation or "").strip().lower()
    if op not in WARP_OPS:
        raise HTTPException(status_code=400, detail="Invalid operation for /process/warp.")

    try:
        contents = await image.read()
        original = _decode_upload(contents)

        if op == "smile":
            processed = apply_smile(original, intensity)
        elif op == "eyebrow":
            processed = apply_eyebrow_raise(original, intensity)
        elif op == "lip":
            processed = apply_lip_widen(original, intensity)
        else:
            processed = apply_face_slim(original, intensity)

        smooth_strength = max(0.0, min(1.0, float(smoothing) / 100.0))
        if smooth_strength > 0:
            smoothed = cv2.GaussianBlur(processed, (0, 0), 0.5 + smooth_strength * 2.0)
            processed = cv2.addWeighted(
                processed,
                1.0 - smooth_strength * 0.4,
                smoothed,
                smooth_strength * 0.4,
                0,
            )

        metrics = _metrics_dict(original, processed)

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)
        proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        energy = compute_energy_analysis(
            processed,
            radius=int(10 + max(0.0, min(1.0, intensity / 100.0)) * 40),
        )

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
            energy=energy,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/age")
async def process_age(
    image: UploadFile = File(...),
    operation: str = Form(...),
    intensity: float = Form(50),
):
    op = (operation or "").strip().lower()

    if op not in AGE_OPS:
        raise HTTPException(status_code=400, detail="Invalid operation for /process/age.")

    try:
        contents = await image.read()
        original = _decode_upload(contents)

        if op in ["aging", "age", "deaging", "deage"]:
            # Original boyutu koruyoruz, 512x512 yapmıyoruz
            if op in ["aging", "age"]:
                processed = apply_aging(original, intensity)
            else:
                effected = apply_deaging(original, intensity)
                try:
                    rgb_for_landmarks = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                    landmarks = get_landmarks(rgb_for_landmarks)
                    face_mask = create_face_region_mask(original, landmarks)
                    processed = blend_effect_with_mask(
                        original=original,
                        effected=effected,
                        mask=face_mask,
                    )
                except Exception as exc:
                    logger.warning("Face mask unavailable for deaging; using full image: %s", exc)
                    processed = effected

            original_for_metrics = original

        elif op == "fft":
            processed, _ = apply_fft_filter(original, intensity)
            original_for_metrics = original

        else:
            raise HTTPException(status_code=400, detail="Invalid operation for /process/age.")

        _, _, processed_fft = compute_fft(processed)
        spectrum = compute_magnitude_spectrum(processed_fft)

        energy = compute_energy_analysis(
            processed,
            radius=int(10 + max(0.0, min(1.0, intensity / 100.0)) * 40),
        )

        metrics = _metrics_dict(original_for_metrics, processed)

        orig_fft_shifted = compute_fft(original_for_metrics)[2]
        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)

        if op == "fft":
            proc_spectrum = spectrum
            proc_phase_b64 = _compute_phase_b64(processed_fft)
        else:
            proc_fft_shifted = compute_fft(processed)[2]
            proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)
            proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(
            cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR)
        )

        proc_spectrum_b64 = _data_url_from_image(
            cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR)
        )

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
            energy=energy,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/cartoon")
async def process_cartoon(
    image: UploadFile = File(...),
):
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        processed = apply_cartoon_filter(original)
        metrics = _metrics_dict(original, processed)

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)
        proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        energy = compute_energy_analysis(processed, radius=30)

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
            energy=energy,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/makeup")
async def process_makeup(
    image: UploadFile = File(...),
    region: str = Form("lip"),
    hue: int = Form(0),
    color: str | None = Form(None),
    opacity: float = Form(0.5),
):
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        rgb_for_landmarks = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(rgb_for_landmarks)

        processed = apply_virtual_makeup(
            image=original,
            landmarks=landmarks,
            region=region,
            hue=_hex_color_to_hue(color, hue),
            opacity=opacity,
        )

        metrics = _metrics_dict(original, processed)

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)
        proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        energy = compute_energy_analysis(processed, radius=30)

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
            energy=energy,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/estimate_age")
async def process_estimate_age(
    image: UploadFile = File(...),
):
    try:
        contents = await image.read()
        img = _decode_upload(contents)

        try:
            from modules.ai_module import estimate_age
        except ModuleNotFoundError:
            from backend.modules.ai_module import estimate_age

        result_before = estimate_age(img)
        if result_before.get("status") != "success":
            raise HTTPException(
                status_code=422,
                detail=result_before.get("error", "Age estimation failed."),
            )

        before_age = result_before["estimated_age"]

        aged_img = apply_aging(img, intensity=50)
        result_after = estimate_age(aged_img)

        if result_after.get("status") != "success":
            after_age = before_age
        else:
            after_age = result_after["estimated_age"]

        diff = after_age - before_age
        if diff >= 0:
            age_diff_str = f"Görünen yaş değişimi: +{diff} yıl"
        else:
            age_diff_str = f"Görünen yaş değişimi: {diff} yıl"

        return {
            "status": "success",
            "estimated_age": before_age,
            "after_age": after_age,
            "age_bucket": result_before.get("age_bucket", ""),
            "confidence": result_before.get("confidence", 0),
            "age_diff_str": age_diff_str,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/eye-size")
async def process_eye_size(
    image: UploadFile = File(...),
    scale: float = Form(0),
):
    """
    Apply eye scaling (enlargement/shrinking) to the uploaded image.

    Parameters
    ----------
    scale : float
        Scaling intensity from -100 to 100. Positive enlarges, negative shrinks.
    """
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        try:
            from modules.warping_module import apply_eye_scaling
        except ModuleNotFoundError:
            from backend.modules.warping_module import apply_eye_scaling

        processed = apply_eye_scaling(original, int(scale))

        metrics = _metrics_dict(original, processed)

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)
        proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        energy = compute_energy_analysis(processed, radius=30)

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
            energy=energy,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/landmarks")
async def process_landmarks(
    image: UploadFile = File(...),
):
    """
    Extract 468 MediaPipe FaceMesh landmarks from the uploaded image.

    The image is resized to 512x512 and converted to RGB before landmark
    extraction. Returned coordinates are normalized between 0.0 and 1.0.
    """
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        preprocessed = preprocess_image(original)
        landmarks = get_landmarks(preprocessed)

        preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)
        preprocessed_b64 = f"data:image/png;base64,{encode_image_to_base64(preprocessed_bgr)}"

        return {
            "landmarks": landmarks,
            "count": len(landmarks),
            "image_b64": preprocessed_b64,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/glasses")
async def process_glasses(
    image: UploadFile = File(...),
    glasses_type: str = Form("aviator"),
):
    """
    Overlay procedural 3D-modeled glasses on the face.

    Parameters
    ----------
    glasses_type : str
        Model ID: ``"aviator"``, ``"wayfarer"``, ``"round"``
        (legacy ``"sunglasses"`` / ``"reading"`` still accepted).
    """
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        # Get landmarks
        rgb_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb_img)
        landmarks = get_landmarks(preprocessed)

        # Import glasses module
        try:
            from modules.glasses_module import apply_glasses
        except ModuleNotFoundError:
            from backend.modules.glasses_module import apply_glasses

        model_id = (glasses_type or "aviator").strip().lower()
        processed = apply_glasses(original, landmarks, model_id)

        metrics = _metrics_dict(original, processed)

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)
        proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        energy = compute_energy_analysis(processed, radius=30)

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
            energy=energy,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ──────────────────────────────────────────────────────────────────────────────
# EMOJI PRESET ENDPOINT – 6 modular preset functions
# ──────────────────────────────────────────────────────────────────────────────


class EmojiPresetRequest(BaseModel):
    """JSON body for the emoji-preset endpoint."""
    image_b64: str
    preset_name: str
    description: str | None = None


def _decode_base64_image(data_url: str) -> np.ndarray:
    """Decode a data-URL or raw base64 string into a BGR OpenCV image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 image.")
    return img


def _create_face_overlay_mask(image: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Create a simple convex-hull face mask from MediaPipe landmarks.
    Returns a single-channel float32 mask in [0, 1].
    """
    h, w = image.shape[:2]
    # Use face oval landmarks for the mask (MediaPipe face mesh silhouette)
    face_oval_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ]
    pts = []
    for idx in face_oval_idx:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append((int(lm["x"] * w), int(lm["y"] * h)))
    if len(pts) < 3:
        return np.zeros((h, w), dtype=np.float32)
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # Feather the edges
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    return mask.astype(np.float32) / 255.0


def _apply_color_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple,
    opacity: float = 0.35,
) -> np.ndarray:
    """
    Apply a color overlay on the image using the given mask and opacity.
    Preserves original image details through alpha blending.
    """
    overlay = np.full_like(image, color_bgr, dtype=np.uint8)
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    blended = image.astype(np.float32) * (1.0 - mask_3ch * opacity) + \
              overlay.astype(np.float32) * (mask_3ch * opacity)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _color_eye_landmarks(
    image: np.ndarray,
    landmarks: list,
    color_bgr: tuple,
    radius: int = 4,
) -> np.ndarray:
    """Draw filled circles on eye landmarks with the given color."""
    h, w = image.shape[:2]
    out = image.copy()
    # Left eye ring + right eye ring
    eye_indices = [33, 133, 160, 158, 153, 144, 159, 145,
                   362, 263, 387, 385, 380, 373, 386, 374]
    for idx in eye_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            cx, cy = int(lm["x"] * w), int(lm["y"] * h)
            cv2.circle(out, (cx, cy), radius, color_bgr, -1)
    return out


def _get_lip_landmarks_pts(landmarks: list, h: int, w: int) -> np.ndarray:
    """Extract FULL lip landmark pixel positions (upper + lower, outer contour)."""
    # MediaPipe full outer lip contour (clockwise loop covering both lips)
    lip_idx = [
        # Outer lip contour (upper lip top edge → right → lower lip bottom → left)
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        375, 321, 405, 314, 17, 84, 181, 91, 146,
    ]
    pts = []
    for idx in lip_idx:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append([int(lm["x"] * w), int(lm["y"] * h)])
    return np.array(pts, dtype=np.int32) if pts else np.array([], dtype=np.int32)


def _apply_lip_color(
    image: np.ndarray,
    landmarks: list,
    color_bgr: tuple,
    opacity: float = 0.45,
) -> np.ndarray:
    """Apply color to lips using landmark-based mask."""
    h, w = image.shape[:2]
    pts = _get_lip_landmarks_pts(landmarks, h, w)
    if len(pts) < 3:
        return image
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 3)
    mask_f = mask.astype(np.float32) / 255.0
    return _apply_color_overlay(image, mask_f, color_bgr, opacity)


# ── 1. ALIEN PRESET (v3 – single-pass warp, dense anchors, seamless blend) ──


def _generate_warp_anchors(
    w: int, h: int, face_lm: np.ndarray, spacing: int = 40
) -> np.ndarray:
    """
    Generate a dense grid of STATIC anchor points along the image edges
    and around the face bounding box.  These appear in both src and dst
    at the same location (zero displacement), producing small Delaunay
    triangles that interpolate smoothly into a completely static background.
    """
    pts = []

    # ── 1. Image-edge grid (all 4 borders) ──
    for x in range(0, w, spacing):
        xf = float(min(x, w - 1))
        pts.append([xf, 0.0])
        pts.append([xf, float(h - 1)])
    for y in range(spacing, h, spacing):
        yf = float(min(y, h - 1))
        pts.append([0.0, yf])
        pts.append([float(w - 1), yf])

    # ── 2. Explicit corners ──
    for c in [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]:
        pts.append([float(c[0]), float(c[1])])

    # ── 3. Face bounding-box ring (expanded 30 %) ──
    fx0 = float(np.min(face_lm[:, 0]))
    fx1 = float(np.max(face_lm[:, 0]))
    fy0 = float(np.min(face_lm[:, 1]))
    fy1 = float(np.max(face_lm[:, 1]))
    pad_x = (fx1 - fx0) * 0.30
    pad_y = (fy1 - fy0) * 0.30
    bx0 = max(0, fx0 - pad_x)
    bx1 = min(w - 1, fx1 + pad_x)
    by0 = max(0, fy0 - pad_y)
    by1 = min(h - 1, fy1 + pad_y)
    ring_sp = spacing // 2
    for x in np.arange(bx0, bx1, ring_sp):
        pts.append([float(x), float(by0)])
        pts.append([float(x), float(by1)])
    for y in np.arange(by0, by1, ring_sp):
        pts.append([float(bx0), float(y)])
        pts.append([float(bx1), float(y)])

    anchors = np.array(pts, dtype=np.float32)

    # ── 4. Remove anchors too close to any face landmark ──
    if len(anchors) > 0 and len(face_lm) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(face_lm)
        dists, _ = tree.query(anchors)
        keep = dists > 8.0          # keep only anchors > 8 px from any lm
        anchors = anchors[keep]

    return anchors


def _create_extended_face_mask(
    image: np.ndarray,
    landmarks: list,
    forehead_extend: float = 0.35,
) -> np.ndarray:
    """
    Face mask covering the FULL face up to the hairline.

    The top face-oval points are shifted upward by *forehead_extend*
    fraction of the face height so the mask covers the forehead that
    MediaPipe's silhouette normally leaves exposed.
    Multi-pass Gaussian blur creates soft feathered edges.
    """
    h, w = image.shape[:2]

    face_oval_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ]
    pts = []
    for idx in face_oval_idx:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append([int(lm["x"] * w), int(lm["y"] * h)])
    if len(pts) < 3:
        return np.zeros((h, w), dtype=np.float32)

    # ── Extend top points upward toward hairline ──
    ys = [p[1] for p in pts]
    top_y = min(ys)
    bot_y = max(ys)
    face_h = bot_y - top_y
    extend_px = face_h * forehead_extend

    for i, pt in enumerate(pts):
        # For points in the upper 30 % of the face oval, push upward
        if pt[1] < top_y + face_h * 0.30:
            t = max(0, (pt[1] - top_y) / (face_h * 0.30 + 1e-6))
            shift = extend_px * (1.0 - t * 0.7)
            pts[i] = [pt[0], max(0, int(pt[1] - shift))]

    poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # Heavy multi-pass feathering
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = cv2.GaussianBlur(mask, (51, 51), 20)

    return mask.astype(np.float32) / 255.0


def _apply_green_tint_hsv(
    image: np.ndarray,
    mask: np.ndarray,
    hue: int = 60,
    saturation_boost: float = 0.55,
    opacity: float = 0.50,
) -> np.ndarray:
    """
    Green tint via HSV hue-shift.  Preserves the V (brightness) channel
    so skin texture, shadows and highlights stay intact.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    tinted_hsv = hsv.copy()
    tinted_hsv[:, :, 0] = float(hue)
    tinted_hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * (1.0 - saturation_boost) + 255.0 * saturation_boost,
        0, 255,
    )

    tinted_bgr = cv2.cvtColor(
        np.clip(tinted_hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    mask_3ch = np.stack([mask, mask, mask], axis=2)
    blended = (
        image.astype(np.float32) * (1.0 - mask_3ch * opacity)
        + tinted_bgr.astype(np.float32) * (mask_3ch * opacity)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_alien(image: np.ndarray) -> np.ndarray:
    """
    👽 Alien v3 – single-pass warp, dense boundary anchors, seamless blend.

    Pipeline
    --------
    0. Detect 468 landmarks once
    1. Compute ALL deltas in one array (chin sculpt + cheek narrow + eyes)
    2. Warp ONCE with dense static boundary anchors → no tearing
    3. Build extended face mask (covers forehead to hairline)
    4. HSV green tint (luminance-preserving)
    5. cv2.seamlessClone composite onto original → no seam artifacts
    """
    out = image.copy()
    h, w = out.shape[:2]

    # ── Stage 0: detect landmarks ──
    lm = detect_face_landmarks(out)
    if lm is None:
        logger.warning("Alien: no face detected – returning original")
        return out

    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)
    face_center_x = (lm[133, 0] + lm[362, 0]) / 2.0
    nose_tip = lm[1].copy()

    # ── Stage 1a: Chin sculpting deltas ──
    chin_contour = [
        397, 365, 379, 378, 400, 377, 152,
        148, 176, 149, 150, 136, 172,
    ]
    mid_jaw = [361, 288, 58, 132]
    chin_pull = face_sz * 0.18

    for idx in chin_contour:
        pt = lm[idx]
        dx = face_center_x - pt[0]
        hw = min(1.0, abs(dx) / (face_sz * 0.6))
        deltas[idx, 0] += dx * 0.45 * hw
        vw = min(1.0, abs(pt[1] - nose_tip[1]) / (face_sz * 0.8))
        deltas[idx, 1] += chin_pull * 0.75 * vw

    for idx in mid_jaw:
        dx = face_center_x - lm[idx, 0]
        deltas[idx, 0] += dx * 0.50
        deltas[idx, 1] += face_sz * 0.14 * 0.25

    # Gaussian spread (chin)
    sigma_chin = face_sz * 0.20
    chin_set = set(chin_contour + mid_jaw)
    for a_idx in chin_contour + mid_jaw:
        if abs(deltas[a_idx, 0]) < 1e-6 and abs(deltas[a_idx, 1]) < 1e-6:
            continue
        wf = _gaussian_falloff(lm, a_idx, sigma_chin)
        for i in range(len(lm)):
            if i in chin_set:
                continue
            deltas[i, 0] += wf[i] * deltas[a_idx, 0] * 0.25
            deltas[i, 1] += wf[i] * deltas[a_idx, 1] * 0.25

    # ── Stage 1b: Cheek narrowing (replaces separate face_slim call) ──
    cheek_left = [234, 127, 162, 93]
    cheek_right = [454, 323, 389, 356]
    cheek_pull = face_sz * 0.06
    for idx in cheek_left:
        deltas[idx, 0] += cheek_pull
    for idx in cheek_right:
        deltas[idx, 0] -= cheek_pull

    # ── Stage 1c: Eye enlargement deltas ──
    left_eye_ring = [33, 133, 160, 158, 153, 144, 159, 145]
    right_eye_ring = [362, 263, 387, 385, 380, 373, 386, 374]
    center_l = np.mean(lm[left_eye_ring], axis=0)
    center_r = np.mean(lm[right_eye_ring], axis=0)
    eye_factor = 1.0
    eye_sigma = face_sz * 0.16

    dists_l = np.linalg.norm(lm - center_l, axis=1)
    dists_r = np.linalg.norm(lm - center_r, axis=1)
    wl = np.exp(-0.5 * (dists_l / max(eye_sigma, 1e-6)) ** 2)
    wr = np.exp(-0.5 * (dists_r / max(eye_sigma, 1e-6)) ** 2)

    for i in range(len(lm)):
        deltas[i] += (lm[i] - center_l) * eye_factor * wl[i]
        deltas[i] += (lm[i] - center_r) * eye_factor * wr[i]

    # ── Stage 1d: Anchor points (zero-delta) ──
    anchors_zero = [
        10, 338, 297, 332, 284, 251,                       # forehead
        70, 63, 105, 66, 107, 46, 53, 52, 65, 55,          # left brow
        300, 293, 334, 296, 336, 276, 283, 282, 295, 285,   # right brow
        168, 6, 197, 195, 5, 4,                             # nose bridge
    ]
    for idx in anchors_zero:
        deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    # ── Stage 2: Single-pass warp with dense boundary anchors ──
    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm, spacing=40)
    src_all = np.vstack([lm, boundary])
    dst_all = np.vstack([dst, boundary])      # boundary has zero delta
    warped = geometric_warp(out, src_all, dst_all)

    # ── Stage 3: Extended face mask (covers forehead to hairline) ──
    try:
        rgb_w = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        prep = preprocess_image(rgb_w)
        warp_lms = get_landmarks(prep)
        face_mask = _create_extended_face_mask(warped, warp_lms, 0.35)
    except Exception:
        face_mask = np.ones((h, w), dtype=np.float32) * 0.4

    # ── Stage 4: HSV green tint ──
    tinted = _apply_green_tint_hsv(
        warped, face_mask, hue=60, saturation_boost=0.55, opacity=0.50,
    )

    return tinted


# ── 2. ROBOT PRESET ──────────────────────────────────────────────────────────
def _apply_robot(image: np.ndarray) -> np.ndarray:
    """🤖 Robot: square jaw warp + silver overlay + yellow eyes + red antennas."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = detect_face_landmarks(out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Square jaw: push jaw outward horizontally
    jaw = [397,365,379,378,400,377,152,148,176,149,150,136,172,361,288,58,132]
    cx = (lm[133,0]+lm[362,0])/2.0
    for idx in jaw:
        dx = lm[idx,0] - cx
        deltas[idx,0] += np.sign(dx) * face_sz * 0.06
    # Flatten mouth
    mouth_top = [13,14,312,311,310,415,308,324,318,402,317]
    mouth_bot = [87,178,88,95,78,61,146,91,181,84,17]
    mouth_y = (lm[13,1]+lm[14,1]+lm[17,1])/3.0
    for idx in mouth_top:
        if idx < len(lm): deltas[idx,1] += (mouth_y - lm[idx,1]) * 0.3
    for idx in mouth_bot:
        if idx < len(lm): deltas[idx,1] += (mouth_y - lm[idx,1]) * 0.3

    anchors_zero = [10,338,297,332,284,251,70,63,105,66,107,300,293,334,296,336,168,6,197,195,5,4]
    for idx in anchors_zero: deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm,boundary]), np.vstack([dst,boundary]))

    # Extended mask + silver overlay
    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
        mask = _create_extended_face_mask(warped, wlms, 0.35)
    except Exception:
        mask = np.ones((h,w), np.float32)*0.4
        wlms = None
    tinted = _apply_color_overlay(warped, mask, (192,192,192), 0.30)

    # Yellow eyes
    if wlms:
        tinted = _color_eye_landmarks(tinted, wlms, (0,255,255), 5)

    # Red antennas from ears
    if wlms and len(wlms) > 454:
        for ear_idx in [234, 454]:
            ex, ey = int(wlms[ear_idx]["x"]*w), int(wlms[ear_idx]["y"]*h)
            sign = -1 if ear_idx == 234 else 1
            tip_x = ex + sign * int(face_sz * 0.3)
            tip_y = ey - int(face_sz * 0.5)
            cv2.line(tinted, (ex,ey), (tip_x,tip_y), (160,160,160), max(4,int(face_sz*0.04)))
            cv2.circle(tinted, (tip_x,tip_y), max(8,int(face_sz*0.07)), (0,0,255), -1)

    return tinted


# ── 3. CLOWN PRESET ──────────────────────────────────────────────────────────
def _apply_clown(image: np.ndarray) -> np.ndarray:
    """🤡 Clown: Joker face paint, panda eyes, messy red smile."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = detect_face_landmarks(out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Subtle smile warp
    deltas[61,0] -= face_sz * 0.05
    deltas[291,0] += face_sz * 0.05
    deltas[61,1] -= face_sz * 0.02
    deltas[291,1] -= face_sz * 0.02
    
    # Boundary Control
    anchors_zero = [10, 338, 297, 332, 284, 251, 152, 168, 6, 197, 195, 5]
    anchors_zero += [33, 133, 362, 263, 159, 145, 386, 374]
    
    for idx in anchors_zero:
        if idx < len(lm):
            deltas[idx] = 0.0

    deltas[np.abs(deltas) < 0.05] = 0.0

    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm,boundary]), np.vstack([dst,boundary]))

    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
    except Exception:
        wlms = None
        
    if wlms is not None:
        # 1. Clown Face Paint Base (Reduced Whiteness)
        try:
            base_mask = _create_extended_face_mask(warped, wlms, 0.05)
        except Exception:
            base_mask = np.zeros((h, w), dtype=np.float32)

        # Subtle white tint instead of heavy messy noise
        paint_alpha = base_mask * 0.25  # Lower opacity
        paint_color = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
        paint_alpha_3d = paint_alpha[..., np.newaxis]
        warped = (warped * (1 - paint_alpha_3d) + paint_color * paint_alpha_3d).astype(np.uint8)

        # 2. Clown "Emoji" Eyes (Blue Triangles)
        def draw_clown_triangle(img, pt, direction, size, color=(255, 50, 50)):
            # direction: -1 (pointing up), 1 (pointing down)
            pt_top = np.array([pt[0], pt[1] + direction * size])
            width = size * 0.5
            pt_left = np.array([pt[0] - width, pt[1] + direction * size * 0.2])
            pt_right = np.array([pt[0] + width, pt[1] + direction * size * 0.2])
            # To make it look like the emoji, the base of the triangle is near the eye, pointing outwards
            # Actually, standard way: triangle base near eye, pointing away.
            base_left = np.array([pt[0] - width*0.5, pt[1]])
            base_right = np.array([pt[0] + width*0.5, pt[1]])
            pts = np.array([pt_top, base_left, base_right], np.int32)
            cv2.fillPoly(img, [pts], color)

        if 159 < len(wlms) and 145 < len(wlms):
            pt_top_l = (int(wlms[159]["x"]*w), int(wlms[159]["y"]*h))
            pt_bot_l = (int(wlms[145]["x"]*w), int(wlms[145]["y"]*h))
            draw_clown_triangle(warped, pt_top_l, -1, int(face_sz * 0.25))
            draw_clown_triangle(warped, pt_bot_l, 1, int(face_sz * 0.25))
            
        if 386 < len(wlms) and 374 < len(wlms):
            pt_top_r = (int(wlms[386]["x"]*w), int(wlms[386]["y"]*h))
            pt_bot_r = (int(wlms[374]["x"]*w), int(wlms[374]["y"]*h))
            draw_clown_triangle(warped, pt_top_r, -1, int(face_sz * 0.25))
            draw_clown_triangle(warped, pt_bot_r, 1, int(face_sz * 0.25))

        # 3. Exaggerated Clown Smile (Thick Rounded Lips)
        outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        lip_pts = []
        for idx in outer_lip_indices:
            if idx < len(wlms):
                lip_pts.append([int(wlms[idx]["x"] * w), int(wlms[idx]["y"] * h)])
                
        if len(lip_pts) == len(outer_lip_indices):
            thickness = max(4, int(face_sz * 0.05))
            lip_pts = np.array(lip_pts, np.int32).reshape((-1, 1, 2))
            
            # Draw thick polylines
            cv2.polylines(warped, [lip_pts], isClosed=True, color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
            
            # Draw filled circles at every vertex to ensure perfectly rounded corners (dairesel dönüşler)
            for pt in lip_pts:
                cv2.circle(warped, (pt[0][0], pt[0][1]), thickness // 2, (0, 0, 255), -1, cv2.LINE_AA)

        # 4. Large Solid Red Nose
        if 4 < len(wlms):
            nx, ny = int(wlms[4]["x"]*w), int(wlms[4]["y"]*h)
            # Increase the diameter significantly
            cv2.circle(warped, (nx, ny), int(face_sz * 0.28), (0, 0, 255), -1, cv2.LINE_AA)

    return warped


# ── 4. STAR EYES PRESET ──────────────────────────────────────────────────────
def _draw_star(image: np.ndarray, cx: int, cy: int, size: int, color=(0, 255, 255)):
    """Draw a filled 5-pointed star at (cx, cy) with given size."""
    out = image.copy()
    pts = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        r = size if i % 2 == 0 else size * 0.4
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], color)
    return out


def _place_star_masks(image: np.ndarray, landmarks) -> np.ndarray:
    """Place yellow star shapes over both eyes."""
    if landmarks is None:
        return image
    h, w = image.shape[:2]
    out = image.copy()

    left_eye = [33, 133, 160, 158, 153, 144, 159, 145]
    right_eye = [362, 263, 387, 385, 380, 373, 386, 374]

    def eye_center(indices):
        xs = [int(landmarks[i]["x"]*w) for i in indices if i < len(landmarks)]
        ys = [int(landmarks[i]["y"]*h) for i in indices if i < len(landmarks)]
        return (sum(xs)//len(xs), sum(ys)//len(ys)) if xs else None

    cl = eye_center(left_eye)
    cr = eye_center(right_eye)
    if cl is None or cr is None:
        return image

    eye_dist = abs(cr[0] - cl[0])
    star_size = max(12, int(eye_dist * 0.45))

    for (cx, cy) in [cl, cr]:
        out = _draw_star(out, cx, cy, star_size, (0, 255, 255))
    return out


def _apply_star_eyes(image: np.ndarray) -> np.ndarray:
    """🤩 Star Eyes: subtle smile + warm tint + EAR-reactive glowing stars."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = detect_face_landmarks(out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Subtle wide smile
    deltas[61,0] -= face_sz * 0.04
    deltas[291,0] += face_sz * 0.04
    deltas[61,1] -= face_sz * 0.02
    deltas[291,1] -= face_sz * 0.02

    anchors_zero = [10,338,297,332,284,251,168,6,197,195,5,4,152]
    for idx in anchors_zero: deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm,boundary]), np.vstack([dst,boundary]))

    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
        mask = _create_extended_face_mask(warped, wlms, 0.35)
    except Exception:
        mask = np.ones((h,w), np.float32)*0.4
        wlms = None
        
    # Warm yellowish/golden color mask
    tinted = _apply_color_overlay(warped, mask, (50, 200, 255), 0.20)

    # Procedural drawing: EAR-tracked bloom + sharp stars over eyes
    if wlms:
        def _pt(idx):
            if idx >= len(wlms):
                return None
            return np.array([wlms[idx]["x"] * w, wlms[idx]["y"] * h], dtype=np.float32)

        def _eye_center(indices):
            pts = [_pt(i) for i in indices]
            pts = [p for p in pts if p is not None]
            if not pts:
                return None
            return np.mean(np.vstack(pts), axis=0)

        def _eye_ear(idxs):
            p1, p2, p3, p4, p5, p6 = [_pt(i) for i in idxs]
            if any(p is None for p in [p1, p2, p3, p4, p5, p6]):
                return 0.30
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            hdist = max(np.linalg.norm(p1 - p4), 1e-6)
            return float((v1 + v2) / (2.0 * hdist))

        def _star_poly(cx, cy, size, y_scale=1.0):
            pts = []
            for i in range(10):
                ang = i * np.pi / 5.0 - np.pi / 2.0
                r = size if i % 2 == 0 else size * 0.42
                x = cx + r * np.cos(ang)
                y = cy + r * np.sin(ang) * y_scale
                pts.append([int(round(x)), int(round(y))])
            return np.array(pts, dtype=np.int32)

        left_eye = [33, 133, 160, 158, 153, 144, 159, 145]
        right_eye = [362, 263, 387, 385, 380, 373, 386, 374]
        left_ear_idx = [33, 160, 158, 133, 153, 144]
        right_ear_idx = [362, 385, 387, 263, 373, 380]

        cl = _eye_center(left_eye)
        cr = _eye_center(right_eye)
        if cl is not None and cr is not None:
            eye_dist = max(float(abs(cr[0] - cl[0])), 1.0)
            base_size = max(12, int(eye_dist * 0.45))
            ear_l = _eye_ear(left_ear_idx)
            ear_r = _eye_ear(right_ear_idx)

            # More squint -> flatter stars (smaller Y-scale)
            def _ear_to_y_scale(ear_val):
                return float(np.clip(ear_val / 0.30, 0.35, 1.15))

            eyes = [
                (int(round(cl[0])), int(round(cl[1])), _ear_to_y_scale(ear_l)),
                (int(round(cr[0])), int(round(cr[1])), _ear_to_y_scale(ear_r)),
            ]

            glow_color = np.array([0, 150, 255], dtype=np.float32)    # warm orange in BGR
            core_color = (0, 235, 255)                                 # sharp yellow in BGR
            final = tinted.astype(np.float32)

            for cx, cy, y_scale in eyes:
                glow_size = int(base_size * 1.55)
                core_size = int(base_size * 0.92)

                # Bloom layer (large blurred semi-transparent star)
                glow_mask = np.zeros((h, w), dtype=np.float32)
                glow_pts = _star_poly(cx, cy, glow_size, y_scale)
                cv2.fillPoly(glow_mask, [glow_pts], 1.0, lineType=cv2.LINE_AA)
                k = max(7, int(glow_size * 0.9))
                if k % 2 == 0:
                    k += 1
                glow_mask = cv2.GaussianBlur(glow_mask, (k, k), sigmaX=0, sigmaY=0)
                glow_alpha = np.clip(glow_mask * 0.75, 0.0, 0.75)[..., None]
                final = final * (1.0 - glow_alpha) + glow_color * glow_alpha

                # Core star (sharp solid center)
                core_mask = np.zeros((h, w), dtype=np.float32)
                core_pts = _star_poly(cx, cy, core_size, y_scale)
                cv2.fillPoly(core_mask, [core_pts], 1.0, lineType=cv2.LINE_AA)
                core_alpha = np.clip(core_mask, 0.0, 1.0)[..., None]
                core_arr = np.zeros((h, w, 3), dtype=np.float32)
                core_arr[:] = np.array(core_color, dtype=np.float32)
                final = final * (1.0 - core_alpha) + core_arr * core_alpha

            tinted = np.clip(final, 0, 255).astype(np.uint8)

    return tinted


# ── 5. HEART-EYES PRESET ─────────────────────────────────────────────────────
def _draw_heart(image: np.ndarray, cx: int, cy: int, size: int, color=(0,0,255)):
    """Draw a filled heart shape at (cx, cy) with given size."""
    r = size // 2
    # Heart = two circles on top + triangle on bottom
    overlay = image.copy()
    cv2.circle(overlay, (cx - r//2, cy - r//3), r//2, color, -1)
    cv2.circle(overlay, (cx + r//2, cy - r//3), r//2, color, -1)
    tri = np.array([
        [cx - r, cy - r//4],
        [cx + r, cy - r//4],
        [cx, cy + r],
    ], dtype=np.int32)
    cv2.fillConvexPoly(overlay, tri, color)
    return overlay


def _place_heart_masks(image: np.ndarray, landmarks) -> np.ndarray:
    """Place red heart shapes over both eyes."""
    if landmarks is None:
        return image
    h, w = image.shape[:2]
    out = image.copy()

    left_eye = [33, 133, 160, 158, 153, 144, 159, 145]
    right_eye = [362, 263, 387, 385, 380, 373, 386, 374]

    def eye_center(indices):
        xs = [int(landmarks[i]["x"]*w) for i in indices if i < len(landmarks)]
        ys = [int(landmarks[i]["y"]*h) for i in indices if i < len(landmarks)]
        return (sum(xs)//len(xs), sum(ys)//len(ys)) if xs else None

    cl = eye_center(left_eye)
    cr = eye_center(right_eye)
    if cl is None or cr is None:
        return image

    eye_dist = abs(cr[0] - cl[0])
    heart_size = max(12, int(eye_dist * 0.45))

    for (cx, cy) in [cl, cr]:
        out = _draw_heart(out, cx, cy, heart_size, (0, 0, 255))
    return out


def _apply_heart_eyes(image: np.ndarray) -> np.ndarray:
    """😍 Heart-Eyes: brow raise + lip widen + red lips + heart overlays."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = detect_face_landmarks(out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Raise brows
    brow_all = [70,63,105,66,107,46,53,52,65,55,300,293,334,296,336,276,283,282,295,285]
    for idx in brow_all: deltas[idx,1] -= face_sz * 0.04
    # Widen lips
    deltas[61,0] -= face_sz * 0.03
    deltas[291,0] += face_sz * 0.03

    anchors_zero = [10,338,297,332,284,251,168,6,197,195,5,4,152]
    for idx in anchors_zero: deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm,boundary]), np.vstack([dst,boundary]))

    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
        mask = _create_extended_face_mask(warped, wlms, 0.35)
    except Exception:
        mask = np.ones((h,w), np.float32)*0.4
        wlms = None

    # Red lip color + hearts
    if wlms:
        warped = _apply_lip_color(warped, wlms, (0,0,255), 0.45)
        warped = _place_heart_masks(warped, wlms)

    return warped


# ── 6. CRYING PRESET ─────────────────────────────────────────────────────────
def _place_tear_masks(image: np.ndarray, landmarks) -> np.ndarray:
    """Draw light-blue teardrop shapes below each eye."""
    if landmarks is None:
        return image
    h, w = image.shape[:2]
    out = image.copy()

    # Lower-eyelid landmarks
    for lid_idx in [145, 374]:
        if lid_idx >= len(landmarks):
            continue
        lx = int(landmarks[lid_idx]["x"] * w)
        ly = int(landmarks[lid_idx]["y"] * h)

        # Teardrop: circle + elongated triangle below
        tear_r = max(3, int(h * 0.012))
        tear_len = max(8, int(h * 0.06))
        color = (255, 200, 100)  # light blue BGR

        # Draw on overlay for alpha blending
        overlay = out.copy()
        cv2.circle(overlay, (lx, ly + tear_r), tear_r, color, -1)
        tri = np.array([
            [lx - tear_r, ly + tear_r],
            [lx + tear_r, ly + tear_r],
            [lx, ly + tear_r + tear_len],
        ], dtype=np.int32)
        cv2.fillConvexPoly(overlay, tri, color)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    return out


def _apply_crying(image: np.ndarray) -> np.ndarray:
    """😢 Crying: brow frown + mouth corner droop + procedural tears."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = detect_face_landmarks(out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Sad/worried arched brows: inner UP, outer DOWN (strong)
    for idx in [107, 55, 336, 285]:
        deltas[idx,1] -= face_sz * 0.10
    for idx in [70, 46, 300, 276]:
        deltas[idx,1] += face_sz * 0.07

    # Mouth corners droop – strong downward pull on outer corners
    deltas[61,1] += face_sz * 0.08   # left corner down
    deltas[291,1] += face_sz * 0.08  # right corner down
    # Neighboring points for smooth curve
    for idx in [146, 91]:
        deltas[idx,1] += face_sz * 0.05
    for idx in [375, 321]:
        deltas[idx,1] += face_sz * 0.05

    anchors_zero = [10,338,297,332,284,251,168,6,197,195,5,4,152]
    for idx in anchors_zero: deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm,boundary]), np.vstack([dst,boundary]))

    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
        mask = _create_extended_face_mask(warped, wlms, 0.35)
    except Exception:
        mask = np.ones((h,w), np.float32)*0.4
        wlms = None

    # Draw tears
    if wlms:
        warped = _place_tear_masks(warped, wlms)

    return warped


# ── Preset dispatcher ────────────────────────────────────────────────────────
_EMOJI_PRESETS_MAP = {
    "alien": _apply_alien,
    "robot": _apply_robot,
    "clown": _apply_clown,
    "star_eyes": _apply_star_eyes,
    "heart_eyes": _apply_heart_eyes,
    "crying": _apply_crying,
}


@router.post("/process/emoji-preset")
async def process_emoji_preset(body: EmojiPresetRequest):
    """
    Apply one of the 6 emoji-themed facial presets to the uploaded image.

    Accepts a JSON body with ``image_b64`` (data-URL or raw base64) and
    ``preset_name`` (alien | robot | clown | star_eyes | heart_eyes | crying).
    """
    preset_key = (body.preset_name or "").strip().lower()
    preset_fn = _EMOJI_PRESETS_MAP.get(preset_key)
    if preset_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown emoji preset '{body.preset_name}'. "
                   f"Valid presets: {', '.join(_EMOJI_PRESETS_MAP.keys())}",
        )

    try:
        original = _decode_base64_image(body.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {exc}") from exc

    try:
        processed = preset_fn(original)

        metrics = _metrics_dict(original, processed)

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        orig_spectrum_b64 = _data_url_from_image(
            cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR)
        )
        proc_spectrum_b64 = _data_url_from_image(
            cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR)
        )

        orig_phase_b64 = _compute_phase_b64(orig_fft_shifted)
        proc_phase_b64 = _compute_phase_b64(proc_fft_shifted)

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            orig_phase_b64=orig_phase_b64,
            proc_phase_b64=proc_phase_b64,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Emoji preset '%s' failed: %s", preset_key, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
