import base64
import logging
import time
import threading

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket
from pydantic import BaseModel

try:
    from modules.frequency_module import (
        apply_aging,
        apply_deaging,
        apply_fft_filter,
        apply_fft_partial_region_artifact,
        apply_cartoon_filter,
        apply_virtual_makeup,
        apply_virtual_makeup_fallback,
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
        apply_fft_partial_region_artifact,
        apply_cartoon_filter,
        apply_virtual_makeup,
        apply_virtual_makeup_fallback,
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
_LIVE_STABLE_LOCK = threading.Lock()
_LIVE_STABLE_LANDMARKS: dict[str, np.ndarray] = {}

WARP_OPS = {"smile", "eyebrow", "lip", "slim"}
AGE_OPS = {"aging", "deaging", "age", "deage", "fft"}
MAKEUP_OPS = {"makeup_lips", "makeup_eyeshadow", "makeup_blush", "makeup_eyeliner", "makeup_mascara"}


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


# ══════════════════════════════════════════════════════════════════════════════
# GÖREV 2 — Downsample Performance Pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Heavy operations (landmark detection, Delaunay, geometric_warp) run on a
# LOW-resolution copy.  Final masks / displacement fields are scaled UP to
# the original resolution before compositing.  This keeps the geometry pass
# under ~30 ms on a standard laptop (targeting 30 FPS).
# ══════════════════════════════════════════════════════════════════════════════

LO_W, LO_H = 480, 360          # low-res processing canvas
_DS_INTERP_DOWN = cv2.INTER_AREA
_DS_INTERP_UP   = cv2.INTER_LINEAR


def _downsample(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (lo_image, sx, sy) – scale factors from lo → hi."""
    h, w = image.shape[:2]
    if w <= LO_W and h <= LO_H:
        return image, 1.0, 1.0
    lo = cv2.resize(image, (LO_W, LO_H), interpolation=_DS_INTERP_DOWN)
    return lo, w / LO_W, h / LO_H


def _upsample_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Scale a single-channel mask to target (h, w)."""
    return cv2.resize(mask, (target_hw[1], target_hw[0]),
                      interpolation=_DS_INTERP_UP)


def _process_lo_composite(
    hi_image: np.ndarray,
    apply_fn,
    *,
    needs_mask: bool = False,
) -> np.ndarray:
    """
    Run *apply_fn* on a low-resolution copy and composite back to hi-res.

    If apply_fn is purely pixel-level (color grading, FFT), the function is
    applied directly at full resolution (cheap).  For geometry-heavy functions
    that benefit from downsampling, this wrapper:

      1. Downsample the image
      2. Run apply_fn on the lo-res copy → get processed lo-res image
      3. Upscale the processed image to original resolution
      4. Blend with the original using a face-region mask (optional)
    """
    h, w = hi_image.shape[:2]
    if w * h <= LO_W * LO_H:
        # Already small — run directly
        return apply_fn(hi_image)

    lo, sx, sy = _downsample(hi_image)
    lo_result = apply_fn(lo)

    # Scale back
    hi_result = cv2.resize(lo_result, (w, h), interpolation=_DS_INTERP_UP)

    if not needs_mask:
        return hi_result

    # Blend: use the low-res result in the face region, original elsewhere
    # This avoids background blurring from upscale interpolation
    try:
        rgb_lo = cv2.cvtColor(lo, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(preprocess_image(rgb_lo))
        lo_mask = _create_face_overlay_mask(lo, landmarks)
        hi_mask = _upsample_mask(lo_mask, (h, w))
        hi_mask3 = hi_mask[..., np.newaxis]
        blended = (hi_result.astype(np.float32) * hi_mask3 +
                   hi_image.astype(np.float32) * (1.0 - hi_mask3))
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return hi_result


# ══════════════════════════════════════════════════════════════════════════════
# GÖREV 3 — Safe Aging/FFT Blend (float32 buffer, bounds checking)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_aging_blend(
    original: np.ndarray,
    aged: np.ndarray,
    face_mask: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Blend an aging/deaging result with the original using float32 arithmetic.

    Parameters
    ----------
    original : uint8 BGR image
    aged     : uint8 BGR image (same shape)
    face_mask: optional float32 single-channel mask [0, 1].
               If None, full-image blend is performed.
    strength : blending strength in [0, 1].

    Returns
    -------
    uint8 BGR blended image — guaranteed no overflow / NaN.
    """
    # Ensure same spatial dimensions
    if original.shape[:2] != aged.shape[:2]:
        aged = cv2.resize(aged, (original.shape[1], original.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    # Work in float32
    orig_f = original.astype(np.float32)
    aged_f = aged.astype(np.float32)

    # Sanitize NaN / Inf that might creep in from FFT operations
    aged_f = np.nan_to_num(aged_f, nan=0.0, posinf=255.0, neginf=0.0)

    strength = float(np.clip(strength, 0.0, 1.0))

    if face_mask is not None:
        # Ensure mask is float32 in [0, 1]
        if face_mask.dtype != np.float32:
            face_mask = face_mask.astype(np.float32)
        if face_mask.max() > 1.0:
            face_mask = face_mask / 255.0
        # Ensure spatial match
        if face_mask.shape[:2] != original.shape[:2]:
            face_mask = cv2.resize(face_mask,
                                   (original.shape[1], original.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        mask3 = face_mask[..., np.newaxis] * strength
        blended = orig_f * (1.0 - mask3) + aged_f * mask3
    else:
        blended = orig_f * (1.0 - strength) + aged_f * strength

    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# GÖREV 1 — Unified Dispatch Endpoint  /process/apply
# ══════════════════════════════════════════════════════════════════════════════
#
# Single entry-point for ALL filter categories:
#   - Geometric warps  (smile, eyebrow, lip, slim)
#   - Emoji presets     (alien, robot, clown, star_eyes, heart_eyes, crying)
#   - Aging / De-aging  (aging, deaging, fft)
#   - Hair dye          (hair_color)
#   - Cartoon           (cartoon)
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedApplyRequest(BaseModel):
    """JSON body for the unified /process/apply endpoint."""
    image_b64: str
    filter_name: str
    intensity: float = 50.0
    smoothing: float = 30.0
    # Hair dye specific
    hair_color: str | None = None       # "R,G,B"  e.g. "255,165,0"
    hair_intensity: float = 0.6
    makeup_hue: int = 0
    makeup_opacity: float = 0.5
    # Bypass FFT spectra computation for faster live mode
    skip_spectra: bool = False
    # Optional client-side timing probes for end-to-end profiling
    client_capture_ts: float | None = None
    client_send_ts: float | None = None


def _build_warp_fn(op: str, intensity: float, smoothing: float):
    """Return a callable(image) → image for the given warp operation."""
    def _fn(image):
        if op == "smile":
            result = apply_smile(image, intensity)
        elif op == "eyebrow":
            result = apply_eyebrow_raise(image, intensity)
        elif op == "lip":
            result = apply_lip_widen(image, intensity)
        elif op == "slim":
            result = apply_face_slim(image, intensity)
        elif op == "eye_scale":
            result = apply_eye_scaling(image, intensity)
        else:
            return image
        # Apply smoothing
        sm = max(0.0, min(1.0, smoothing / 100.0))
        if sm > 0:
            blurred = cv2.GaussianBlur(result, (0, 0), 0.5 + sm * 2.0)
            result = cv2.addWeighted(result, 1.0 - sm * 0.4,
                                     blurred, sm * 0.4, 0)
        return result
    return _fn


def _build_age_fn(op: str, intensity: float):
    """Return a callable(image) → image for aging/deaging."""
    def _fn(image):
        if op in ("aging", "age"):
            aged = apply_aging(image, intensity)
            try:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                landmarks = get_landmarks(rgb)
                face_mask = create_face_region_mask(image, landmarks)
                return _safe_aging_blend(image, aged, face_mask,
                                         strength=min(1.0, intensity / 100.0))
            except Exception:
                return _safe_aging_blend(image, aged,
                                         strength=min(1.0, intensity / 100.0))
        elif op in ("deaging", "deage"):
            deaged = apply_deaging(image, intensity)
            try:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                landmarks = get_landmarks(rgb)
                face_mask = create_face_region_mask(image, landmarks)
                return _safe_aging_blend(image, deaged, face_mask,
                                         strength=min(1.0, intensity / 100.0))
            except Exception:
                return _safe_aging_blend(image, deaged,
                                         strength=min(1.0, intensity / 100.0))
        elif op == "fft":
            processed, _ = apply_fft_filter(image, intensity)
            return processed
        return image
    return _fn


def _build_hair_fn(color_str: str, hair_intensity: float):
    """Return a callable(image) → image for hair-dye."""
    def _fn(image):
        try:
            from modules.hair_module import apply_hair_color
        except ModuleNotFoundError:
            from backend.modules.hair_module import apply_hair_color
        return apply_hair_color(image, color_str, hair_intensity)
    return _fn


def _build_makeup_fn(region: str, hue: int, opacity: float):
    """Return callable(image) -> image for landmark-based virtual makeup."""
    def _fn(image):
        rgb_for_landmarks = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(preprocess_image(rgb_for_landmarks))
        return apply_virtual_makeup(
            image=image,
            landmarks=landmarks,
            region=region,
            hue=int(hue),
            opacity=float(opacity),
        )
    return _fn


def _get_stable_landmarks(tag: str, image: np.ndarray) -> np.ndarray | None:
    """Landmark fetch with fallback to previous stable mesh."""
    lm = detect_face_landmarks(image)
    with _LIVE_STABLE_LOCK:
        if lm is not None:
            _LIVE_STABLE_LANDMARKS[tag] = lm.copy()
            return lm
        prev = _LIVE_STABLE_LANDMARKS.get(tag)
        return prev.copy() if prev is not None else None


# Categories that benefit from downsample processing
_GEOMETRY_HEAVY = {"smile", "eyebrow", "lip", "slim", "eye_scale",
                   "alien", "robot", "clown", "star_eyes",
                   "heart_eyes", "crying"}


@router.post("/process/apply")
async def process_unified_apply(body: UnifiedApplyRequest):
    """
    Unified dispatch endpoint for ALL filter categories.

    ``filter_name`` can be:
      - Warp:    ``smile``, ``eyebrow``, ``lip``, ``slim``
      - Emoji:   ``alien``, ``robot``, ``clown``, ``star_eyes``,
                 ``heart_eyes``, ``crying``
      - Aging:   ``aging``, ``deaging``, ``fft``
      - Hair:    ``hair_color``
      - Cartoon: ``cartoon``
    """
    fname = (body.filter_name or "").strip().lower()
    logger.info("process_unified_apply: filter=%s intensity=%.1f", fname, body.intensity)

    t_start = time.perf_counter()
    t_decode_done = t_start
    t_process_done = t_start

    # Decode image
    try:
        original = _decode_base64_image(body.image_b64)
        t_decode_done = time.perf_counter()
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Image decode failed: {exc}") from exc

    try:
        # ── Route to the correct filter function ─────────────────────────
        if fname in WARP_OPS or fname == "eye_scale":
            apply_fn = _build_warp_fn(fname, body.intensity, body.smoothing)
        elif fname in _EMOJI_PRESETS_MAP:
            apply_fn = _EMOJI_PRESETS_MAP[fname]
        elif fname in AGE_OPS:
            apply_fn = _build_age_fn(fname, body.intensity)
        elif fname in MAKEUP_OPS:
            region = fname.split("_", 1)[1]
            apply_fn = _build_makeup_fn(region, body.makeup_hue, body.makeup_opacity)
        elif fname == "hair_color" and body.hair_color:
            apply_fn = _build_hair_fn(body.hair_color, body.hair_intensity)
        elif fname == "cartoon":
            apply_fn = apply_cartoon_filter
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown filter '{fname}'. Valid: "
                       f"{', '.join(sorted(set(WARP_OPS) | {'eye_scale'} | set(_EMOJI_PRESETS_MAP) | AGE_OPS | MAKEUP_OPS | {'hair_color', 'cartoon'}))}"
            )

        # ── Apply with optional downsampling for geometry-heavy ops ──────
        if fname in _GEOMETRY_HEAVY:
            processed = _process_lo_composite(original, apply_fn,
                                               needs_mask=True)
        else:
            processed = apply_fn(original)
        t_process_done = time.perf_counter()

        # ── Build response ───────────────────────────────────────────────
        metrics = _metrics_dict(original, processed)

        profile = {
            "capture_ms": None,
            "transport_ms": None,
            "infer_ms": round((t_process_done - t_decode_done) * 1000.0, 2),
            "overlay_ms": 0.0,
            "render_ms": round((time.perf_counter() - t_process_done) * 1000.0, 2),
            "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        }
        if body.client_capture_ts is not None and body.client_send_ts is not None:
            try:
                profile["capture_ms"] = round(
                    max(0.0, (body.client_send_ts - body.client_capture_ts) * 1000.0), 2
                )
            except Exception:
                profile["capture_ms"] = None
        if body.client_send_ts is not None:
            try:
                profile["transport_ms"] = round(
                    max(0.0, (time.time() - float(body.client_send_ts)) * 1000.0), 2
                )
            except Exception:
                profile["transport_ms"] = None

        # Skip expensive FFT spectra in live mode
        if body.skip_spectra:
            return {
                "processed_image": _data_url_from_image(processed),
                "image_b64": _data_url_from_image(processed),
                "metrics": metrics,
                "profile": profile,
            }

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        payload = _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=_data_url_from_image(
                cv2.cvtColor(compute_magnitude_spectrum(orig_fft_shifted),
                             cv2.COLOR_GRAY2BGR)),
            proc_spectrum_b64=_data_url_from_image(
                cv2.cvtColor(compute_magnitude_spectrum(proc_fft_shifted),
                             cv2.COLOR_GRAY2BGR)),
            orig_phase_b64=_compute_phase_b64(orig_fft_shifted),
            proc_phase_b64=_compute_phase_b64(proc_fft_shifted),
            energy=None,
        )
        payload["profile"] = profile
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("process_unified_apply '%s' failed: %s", fname, exc,
                     exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
                # apply_deaging already includes internal face-mask-based blending
                # via _build_face_hair_mask — do NOT double-blend with blend_effect_with_mask
                processed = apply_deaging(original, intensity)

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

        if opacity > 1.0:
            opacity = opacity / 100.0

        target_hue = _hex_color_to_hue(color, hue)
        try:
            rgb_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            landmarks = get_landmarks(preprocess_image(rgb_img))
            if not landmarks:
                raise ValueError("No face detected.")
                
            processed = apply_virtual_makeup(
                image=original,
                landmarks=landmarks,
                region=region,
                hue=target_hue,
                opacity=opacity,
            )
        except Exception as exc:
            logger.warning("Makeup landmarks unavailable; using approximate mask: %s", exc)
            processed = apply_virtual_makeup_fallback(
                image=original,
                region=region,
                hue=target_hue,
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
        Model ID: ``"aviator"``, ``"wayfarer"``, ``"round"``, ``"square"``,
        ``"retro"``, ``"cat_eye"``, ``"sport"``, ``"futuristic"``
        (legacy ``"sunglasses"`` / ``"reading"`` / ``"cateye"`` still accepted).
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


def apply_alien_emoji(image_bgr: np.ndarray, intensity: int = 100) -> np.ndarray:
    """
    👽 Uzaylı filtresi:
    - Ters üçgen kafa (çene ince, alın geniş)
    - Büyük siyah oval gözler
    - Yeşil cilt tonu
    """
    try:
        h, w = image_bgr.shape[:2]

        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr.copy()

        face_sz = _face_scale(lm)
        deltas = np.zeros_like(lm)

        nose_tip = lm[1].copy()

        chin_indices = [
            152, 377, 400, 378, 379, 365, 397, 288,
            361, 323, 148, 176, 149, 150, 136, 172, 58, 132
        ]
        for idx in set(chin_indices):
            vec = lm[idx] - nose_tip
            dist = float(np.linalg.norm(vec))
            if dist < 1e-3:
                continue
            pull = face_sz * 0.18
            direction = -vec / dist
            deltas[idx, 0] += direction[0] * pull * 0.8
            deltas[idx, 1] += direction[1] * pull * 0.3

        temple_indices = [234, 454, 127, 356, 162, 389]
        for idx in temple_indices:
            cx = w / 2.0
            dx = lm[idx, 0] - cx
            deltas[idx, 0] += np.sign(dx) * face_sz * 0.08

        left_eye_pts  = [33, 133, 160, 159, 158, 157, 163, 144, 145, 153, 154, 155, 173, 246, 161]
        right_eye_pts = [362, 263, 387, 386, 385, 384, 390, 373, 374, 380, 381, 382, 398, 466, 388]

        c_left  = lm[left_eye_pts].mean(axis=0)
        c_right = lm[right_eye_pts].mean(axis=0)

        eye_scale = 0.8
        sigma = max(face_sz * 0.20, 1e-6)

        d_left = lm - c_left
        d_right = lm - c_right
        
        w_left = np.exp(-0.5 * (np.linalg.norm(d_left, axis=1) / sigma) ** 2)[:, np.newaxis]
        w_right = np.exp(-0.5 * (np.linalg.norm(d_right, axis=1) / sigma) ** 2)[:, np.newaxis]
        
        deltas += d_left * eye_scale * w_left
        deltas += d_right * eye_scale * w_right

        fixed = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                 1, 4, 5, 168, 6, 197, 195]
        for idx in fixed:
            deltas[idx] = 0.0
        deltas[np.abs(deltas) < 0.1] = 0.0

        base = _prepare_warp(image_bgr, lm, deltas)
        base = apply_eyebrow_raise(base, 40)

        lm2 = detect_face_landmarks(base)
        if lm2 is None:
            lm2 = lm

        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        jaw_pts = np.array([[int(lm2[i][0]), int(lm2[i][1])] for i in jaw_indices], dtype=np.int32)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        face_mask_blur = cv2.GaussianBlur(face_mask, (31, 31), 0).astype(np.float32) / 255.0
        face_mask_3ch = np.stack([face_mask_blur] * 3, axis=-1)

        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_green = hsv.copy()
        hsv_green[:, :, 0] = 75.0
        hsv_green[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2 + 25, 0, 255)
        hsv_green[:, :, 2] = np.clip(hsv[:, :, 2] * 0.88, 0, 255)
        green_img = cv2.cvtColor(hsv_green.astype(np.uint8), cv2.COLOR_HSV2BGR)

        result = (
            green_img.astype(np.float32) * face_mask_3ch * 0.60
            + base.astype(np.float32) * (1.0 - face_mask_3ch * 0.60)
        ).astype(np.uint8)

        lm3 = detect_face_landmarks(result)
        if lm3 is None:
            lm3 = lm2

        c_left2  = lm3[left_eye_pts].mean(axis=0)
        c_right2 = lm3[right_eye_pts].mean(axis=0)

        eye_rx = int(face_sz * 0.28)
        eye_ry = int(face_sz * 0.22)

        eye_layer = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(eye_layer, (int(c_left2[0]),  int(c_left2[1])),  (eye_rx, eye_ry), 0, 0, 360, (12, 12, 12), -1)
        cv2.ellipse(eye_layer, (int(c_right2[0]), int(c_right2[1])), (eye_rx, eye_ry), 0, 0, 360, (12, 12, 12), -1)

        ho = int(eye_rx * 0.28)
        vo = int(eye_ry * 0.28)
        cv2.circle(eye_layer, (int(c_left2[0])  - ho, int(c_left2[1])  - vo), int(eye_rx * 0.12), (70, 70, 70), -1)
        cv2.circle(eye_layer, (int(c_right2[0]) - ho, int(c_right2[1]) - vo), int(eye_rx * 0.12), (70, 70, 70), -1)

        eye_mask = (eye_layer.sum(axis=2) > 0).astype(np.float32)
        eye_mask = cv2.GaussianBlur(eye_mask, (5, 5), 0)
        eye_mask_3ch = np.stack([eye_mask] * 3, axis=-1)

        result = (
            eye_layer.astype(np.float32) * eye_mask_3ch
            + result.astype(np.float32) * (1.0 - eye_mask_3ch)
        ).astype(np.uint8)

        return result

    except Exception as exc:
        logger.error(f"apply_alien_emoji failed: {exc}")
        return image_bgr.copy()


# ── 2. ROBOT PRESET ──────────────────────────────────────────────────────────
def _pixel_landmarks_to_normalized(landmarks: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    return [
        {"x": float(pt[0]) / max(width, 1), "y": float(pt[1]) / max(height, 1)}
        for pt in landmarks
    ]


def _face_pose_axes(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    left_eye = lm[33] if len(lm) > 33 else lm[0]
    right_eye = lm[263] if len(lm) > 263 else lm[-1]
    chin = lm[152] if len(lm) > 152 else lm.mean(axis=0)
    forehead = lm[10] if len(lm) > 10 else lm.mean(axis=0)

    x_axis = right_eye - left_eye
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1.0:
        x_axis = np.array([1.0, 0.0], dtype=np.float32)
    else:
        x_axis = x_axis / x_norm

    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
    if float(np.dot(y_axis, chin - ((left_eye + right_eye) * 0.5))) < 0:
        y_axis = -y_axis

    center = (left_eye + right_eye + chin + forehead) * 0.25
    face_width = max(float(np.linalg.norm(lm[454] - lm[234])) if len(lm) > 454 else x_norm * 2.2, 1.0)
    face_height = max(float(np.linalg.norm(chin - forehead)), face_width * 1.15)
    return center.astype(np.float32), x_axis.astype(np.float32), y_axis.astype(np.float32), face_width, face_height


def _estimate_face_yaw(lm: np.ndarray) -> float:
    if len(lm) <= 454:
        return 0.0
    left_side = lm[234]
    right_side = lm[454]
    nose = lm[1] if len(lm) > 1 else lm[4]
    face_w = max(float(np.linalg.norm(right_side - left_side)), 1.0)
    face_mid_x = (left_side[0] + right_side[0]) * 0.5
    return float(np.clip((nose[0] - face_mid_x) / face_w, -0.6, 0.6))


def _pose_point(
    center: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    local_x: float,
    local_y: float,
) -> tuple[int, int]:
    pt = center + x_axis * local_x + y_axis * local_y
    return int(round(pt[0])), int(round(pt[1]))


def _apply_robot(image: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
    """🤖 Robot: square jaw warp + silver overlay + yellow eyes + red antennas."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = landmarks.astype(np.float32).copy() if landmarks is not None else detect_face_landmarks(out)
    if lm is None:
        cx, cy = w // 2, h // 2
        face_w = int(w * 0.42)
        face_h = int(h * 0.56)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(8.0, min(h, w) * 0.035))
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        metal = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) * 1.08 + 34.0
        out = (
            out.astype(np.float32) * (1.0 - mask[..., None] * 0.76)
            + metal * (mask[..., None] * 0.76)
        ).astype(np.uint8)
        visor_y = cy - int(face_h * 0.13)
        cv2.rectangle(
            out,
            (cx - int(face_w * 0.30), visor_y - int(face_h * 0.045)),
            (cx + int(face_w * 0.30), visor_y + int(face_h * 0.045)),
            (8, 12, 145),
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            out,
            (cx - int(face_w * 0.30), visor_y - int(face_h * 0.045)),
            (cx + int(face_w * 0.30), visor_y + int(face_h * 0.045)),
            (40, 40, 255),
            max(2, int(face_w * 0.015)),
            cv2.LINE_AA,
        )
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Square jaw: push jaw outward horizontally
    jaw = [397,365,379,378,400,377,152,148,176,149,150,136,172,361,288,58,132]
    cx = (lm[133,0]+lm[362,0])/2.0
    for idx in jaw:
        dx = lm[idx,0] - cx
        deltas[idx,0] += np.sign(dx) * face_sz * 0.12
        deltas[idx,1] += face_sz * 0.025

    temple = [234, 93, 132, 58, 172, 454, 323, 361, 288, 397]
    for idx in temple:
        if idx < len(lm):
            dx = lm[idx, 0] - cx
            deltas[idx, 0] += np.sign(dx) * face_sz * 0.045
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

    # Extended mask + metallic gray skin.
    try:
        wlms = _pixel_landmarks_to_normalized(dst, w, h)
        mask = _create_extended_face_mask(warped, wlms, 0.35)
    except Exception:
        mask = np.ones((h,w), np.float32)*0.4
        wlms = None

    mask = np.clip(mask, 0.0, 1.0)
    mask3 = mask[..., None]

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    metallic = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    metallic = metallic * 1.10 + np.array([24.0, 25.0, 27.0], dtype=np.float32)

    xx = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    shine = np.clip(1.0 - np.abs((xx * 0.75 + yy * 0.25) - 0.46) * 3.4, 0.0, 1.0)
    metallic += shine[..., None] * np.array([28.0, 30.0, 34.0], dtype=np.float32)

    tinted = (
        warped.astype(np.float32) * (1.0 - mask3 * 0.78)
        + metallic * (mask3 * 0.78)
    )

    stripe_period = max(5, int(face_sz * 0.035))
    stripe_height = max(1, stripe_period // 3)
    scan_mask = np.zeros((h, w), dtype=np.float32)
    y_indices = np.arange(h)
    mask_rows = (y_indices % stripe_period) < stripe_height
    scan_mask[mask_rows, :] = 1.0
    scan_mask = cv2.GaussianBlur(scan_mask * mask, (0, 0), 0.65)
    tinted = tinted - scan_mask[..., None] * 34.0
    tinted += scan_mask[..., None] * np.array([12.0, 7.0, 2.0], dtype=np.float32)
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    seam_color = (38, 42, 48)
    pose_center, x_axis, y_axis, pose_w, pose_h = _face_pose_axes(dst)
    panel_w = pose_w * 0.84
    panel_top = -pose_h * 0.56
    panel_bottom = pose_h * 0.58
    forehead_y = panel_top + (panel_bottom - panel_top) * 0.22
    cv2.line(
        tinted,
        _pose_point(pose_center, x_axis, y_axis, -panel_w * 0.38, forehead_y),
        _pose_point(pose_center, x_axis, y_axis, panel_w * 0.38, forehead_y),
        seam_color,
        max(1, int(face_sz * 0.010)),
        cv2.LINE_AA,
    )
    for x_plate in (-panel_w * 0.30, panel_w * 0.30):
        cv2.line(
            tinted,
            _pose_point(pose_center, x_axis, y_axis, x_plate, forehead_y),
            _pose_point(pose_center, x_axis, y_axis, x_plate, panel_top + pose_h * 0.08),
            seam_color,
            max(1, int(face_sz * 0.008)),
            cv2.LINE_AA,
        )

    face_outline_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ]
    outline = np.array(
        [dst[idx] for idx in face_outline_idx if idx < len(dst)],
        dtype=np.int32,
    )
    if len(outline) >= 3:
        cv2.polylines(
            tinted,
            [outline.reshape((-1, 1, 2))],
            True,
            (55, 60, 66),
            max(2, int(face_sz * 0.018)),
            cv2.LINE_AA,
        )

    yaw = _estimate_face_yaw(dst)
    antenna_specs = [(234, -0.35), (454, 0.35)]
    if yaw > 0.08:
        antenna_specs = [(234, -0.35)]
    elif yaw < -0.08:
        antenna_specs = [(454, 0.35)]

    for ear_idx, tilt in antenna_specs:
        if ear_idx < len(dst):
            ex, ey = dst[ear_idx].astype(int)
            antenna_len = int(face_sz * 0.42)
            tip = dst[ear_idx] - y_axis * antenna_len + x_axis * (tilt * face_sz)
            tip_x = int(np.clip(tip[0], 0, w - 1))
            tip_y = int(np.clip(tip[1], 0, h - 1))
            cv2.line(
                tinted,
                (int(ex), int(ey)),
                (tip_x, tip_y),
                (48, 52, 58),
                max(3, int(face_sz * 0.018)),
                cv2.LINE_AA,
            )
            cv2.circle(
                tinted,
                (tip_x, tip_y),
                max(5, int(face_sz * 0.034)),
                (58, 64, 72),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                tinted,
                (tip_x, tip_y),
                max(5, int(face_sz * 0.034)),
                (165, 170, 176),
                1,
                cv2.LINE_AA,
            )

    if wlms:
        def _pt(idx: int) -> tuple[int, int]:
            return int(wlms[idx]["x"] * w), int(wlms[idx]["y"] * h)

        eye_rings = [
            [33, 133, 160, 158, 153, 144, 159, 145],
            [362, 263, 387, 385, 380, 373, 386, 374],
        ]
        valid_eye_pts = []

        for eye_idx in eye_rings:
            if not all(idx < len(wlms) for idx in eye_idx):
                continue
            pts = np.array([_pt(idx) for idx in eye_idx], dtype=np.int32)
            valid_eye_pts.append(pts)

            eye_mask = np.zeros((h, w), dtype=np.float32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(eye_mask, hull, 1.0)
            eye_mask = cv2.GaussianBlur(eye_mask, (0, 0), max(1.2, face_sz * 0.008))

            glow_mask = np.zeros((h, w), dtype=np.float32)
            cx_eye, cy_eye = np.mean(pts, axis=0).astype(int)
            glow_r = max(9, int(face_sz * 0.085))
            cv2.circle(glow_mask, (cx_eye, cy_eye), glow_r, 1.0, -1, cv2.LINE_AA)
            glow_mask = cv2.GaussianBlur(glow_mask, (0, 0), glow_r * 0.65)

            red = np.full_like(tinted, (8, 18, 255), dtype=np.uint8)
            glow3 = glow_mask[..., None] * 0.35
            eye3 = eye_mask[..., None] * 0.92
            tinted = (
                tinted.astype(np.float32) * (1.0 - glow3)
                + red.astype(np.float32) * glow3
            )
            tinted = (
                tinted * (1.0 - eye3)
                + red.astype(np.float32) * eye3
            ).astype(np.uint8)

            x, y, ew, eh = cv2.boundingRect(hull)
            cv2.line(
                tinted,
                (x, cy_eye),
                (x + ew, cy_eye),
                (40, 40, 255),
                max(2, int(face_sz * 0.018)),
                cv2.LINE_AA,
            )

        if len(valid_eye_pts) == 2:
            left_center = np.mean(valid_eye_pts[0], axis=0).astype(int)
            right_center = np.mean(valid_eye_pts[1], axis=0).astype(int)
            bridge_y = int((left_center[1] + right_center[1]) / 2)
            cv2.line(
                tinted,
                (left_center[0], bridge_y),
                (right_center[0], bridge_y),
                (25, 25, 170),
                max(1, int(face_sz * 0.012)),
                cv2.LINE_AA,
            )

        for bolt_idx in [123, 352]:
            if bolt_idx < len(wlms):
                bx, by = _pt(bolt_idx)
                bolt_r = max(3, int(face_sz * 0.024))
                cv2.circle(tinted, (bx, by), bolt_r, (48, 52, 58), -1, cv2.LINE_AA)
                cv2.circle(tinted, (bx, by), bolt_r, (150, 155, 160), 1, cv2.LINE_AA)
                cv2.line(
                    tinted,
                    (bx - bolt_r + 1, by),
                    (bx + bolt_r - 1, by),
                    (115, 120, 125),
                    1,
                    cv2.LINE_AA,
                )

    return tinted


# ── 3. CLOWN PRESET ──────────────────────────────────────────────────────────
def apply_clown_emoji(image_bgr: np.ndarray, intensity: int = 100) -> np.ndarray:
    """
    🤡 Joker tarzı palyaço:
    - Beyaz yüz boyası
    - Büyük mavi eşkenar dörtgen göz makyajı
    - Kırmızı kaşlar
    - Büyük kırmızı dudak boyası
    - Çok büyük kırmızı top burun
    - Geniş kırmızı gülüş çizgisi
    """
    try:
        h, w = image_bgr.shape[:2]

        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr.copy()

        result = image_bgr.copy()
        face_sz = _face_scale(lm)

        # Yüz maskesi
        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        jaw_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in jaw_indices], dtype=np.int32)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        face_mask_blur = cv2.GaussianBlur(face_mask, (25, 25), 0).astype(np.float32) / 255.0
        face_mask_3ch = np.stack([face_mask_blur] * 3, axis=-1)

        # 1. Beyaz yüz boyası %55
        white = np.ones_like(result, dtype=np.float32) * 255
        result = (
            white * face_mask_3ch * 0.55
            + result.astype(np.float32) * (1.0 - face_mask_3ch * 0.55)
        ).astype(np.uint8)

        paint = np.zeros((h, w, 3), dtype=np.float32)

        # 2. Büyük mavi eşkenar dörtgen göz makyajı
        le_cx = int((lm[33][0]  + lm[133][0]) / 2)
        le_cy = int((lm[33][1]  + lm[133][1]) / 2)
        re_cx = int((lm[362][0] + lm[263][0]) / 2)
        re_cy = int((lm[362][1] + lm[263][1]) / 2)

        # Eşkenar dörtgen: 4 köşesi eşit uzaklıkta
        e_r = int(face_sz * 0.22)  # tüm yönlerde eşit yarıçap

        def rhombus_pts(cx, cy, r):
            return np.array([
                [cx - r, cy],   # sol
                [cx,     cy - r],  # üst
                [cx + r, cy],   # sağ
                [cx,     cy + r],  # alt
            ], dtype=np.int32)

        cv2.fillPoly(paint, [rhombus_pts(le_cx, le_cy, e_r)], (210, 90, 10))  # mavi BGR
        cv2.fillPoly(paint, [rhombus_pts(re_cx, re_cy, e_r)], (210, 90, 10))

        # 3. Kırmızı kaşlar
        left_brow_pts  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_brow_pts = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        lb_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in left_brow_pts],  dtype=np.int32)
        rb_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in right_brow_pts], dtype=np.int32)
        brow_thick = max(int(face_sz * 0.06), 3)
        cv2.polylines(paint, [lb_pts], False, (0, 0, 220), brow_thick)
        cv2.polylines(paint, [rb_pts], False, (0, 0, 220), brow_thick)

        # 4. Büyük kırmızı top burun
        nose_pt = (int(lm[4][0]), int(lm[4][1]))
        nose_r  = int(face_sz * 0.20)  # büyük
        cv2.circle(paint, nose_pt, nose_r, (0, 0, 240), -1)
        cv2.circle(paint,
                   (nose_pt[0] - int(nose_r * 0.3), nose_pt[1] - int(nose_r * 0.35)),
                   int(nose_r * 0.22), (100, 100, 255), -1)

        # 5. Büyük kırmızı dudak boyası
        outer_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        om_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in outer_mouth], dtype=np.int32)
        om_center = om_pts.mean(axis=0).astype(int)
        om_big = ((om_pts - om_center) * 1.35 + om_center).astype(np.int32)
        cv2.fillPoly(paint, [om_big], (0, 0, 225))

        # 6. Geniş kırmızı gülüş çizgisi
        left_corner  = (int(lm[61][0]),  int(lm[61][1]))
        right_corner = (int(lm[291][0]), int(lm[291][1]))
        left_cheek   = (int(lm[205][0] - face_sz * 0.20), int(lm[205][1] + face_sz * 0.05))
        right_cheek  = (int(lm[425][0] + face_sz * 0.20), int(lm[425][1] + face_sz * 0.05))
        line_w = max(int(face_sz * 0.08), 4)
        cv2.line(paint, left_corner,  left_cheek,  (0, 0, 225), line_w)
        cv2.line(paint, right_corner, right_cheek, (0, 0, 225), line_w)

        # Makyajı blend et
        paint_blur  = cv2.GaussianBlur(paint, (9, 9), 0)
        paint_alpha = np.clip(paint_blur.sum(axis=2, keepdims=True) / 280.0, 0, 1)
        paint_alpha = np.repeat(paint_alpha, 3, axis=2)

        final = (
            paint_blur * paint_alpha * 0.85
            + result.astype(np.float32) * (1.0 - paint_alpha * 0.85)
        ).astype(np.uint8)

        return final

    except Exception as e:
        logger.error(f"apply_clown_emoji failed: {e}")
        return image_bgr.copy()


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
    cv2.circle(overlay, (cx - r//2, cy - r//3), r//2, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx + r//2, cy - r//3), r//2, color, -1, cv2.LINE_AA)
    tri = np.array([
        [cx - r, cy - r//4],
        [cx + r, cy - r//4],
        [cx, cy + r],
    ], dtype=np.int32)
    cv2.fillConvexPoly(overlay, tri, color, cv2.LINE_AA)
    return overlay


def _place_heart_masks(image: np.ndarray, landmarks) -> np.ndarray:
    """Place large neon red heart shapes over both eyes."""
    if landmarks is None:
        return image
    h, w = image.shape[:2]
    
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
    
    # Göz bebeğini merkeze alan ve kaşları tamamen serbest bırakan boyut (Hafif büyütüldü)
    heart_size = max(17, int(eye_dist * 0.55))

    glow_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    solid_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    neon_color = (40, 40, 255)  # Parlak Kırmızı/Neon glow rengi (BGR)
    solid_color = (20, 20, 235) # Opak Kırmızı

    for (cx, cy) in [cl, cr]:
        # Kalbi göz merkezinde tut
        cy_adj = cy
        
        # 1. Glow Layer (daha büyük çizilip bulanıklaştırılacak)
        glow_canvas = _draw_heart(glow_canvas, cx, cy_adj, int(heart_size * 1.3), neon_color)
        
        # 2. Solid Heart Layer
        solid_temp = np.zeros((h, w, 3), dtype=np.uint8)
        solid_temp = _draw_heart(solid_temp, cx, cy_adj, heart_size, solid_color)
        mask = np.any(solid_temp > 0, axis=-1)
        solid_overlay[mask] = [solid_color[0], solid_color[1], solid_color[2], 255]

    # Neon parlama efekti (cv2.GaussianBlur)
    k_size = int(heart_size * 0.7)
    if k_size % 2 == 0: k_size += 1
    glow_blur = cv2.GaussianBlur(glow_canvas, (k_size, k_size), 0)
    
    result = image.astype(np.float32)
    
    # Glow ekle (Additive Blending)
    result += glow_blur.astype(np.float32) * 1.2
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Opak Kalpleri bindir (Alpha Blending)
    alpha = solid_overlay[..., 3:] / 255.0
    result = (result.astype(np.float32) * (1.0 - alpha) + solid_overlay[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    
    return result


def _apply_heart_eyes(image: np.ndarray) -> np.ndarray:
    """😍 Heart-Eyes: brow raise/widen + blush effect + red lips + neon heart overlays."""
    out = image.copy()
    h, w = out.shape[:2]
    lm = detect_face_landmarks(out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # Ekstrem Kaş Warping - "Masum" Aşık Emoji (😍) kaşı: 
    # İçler çok yukarı, kavis aşağı bastırılmış (düzleşmiş), dışlar aşağı
    left_brow_zones = [
        ([107, 55], -0.18, 0.02), # En iç: çok yukarı, hafif içe
        ([66, 65], -0.10, 0.01),  # Orta-iç: yukarı
        ([105, 52], 0.02, 0.0),   # Tepe kavis (Arch): hafif aşağı bastırarak kavisi kır
        ([63, 53], 0.08, -0.01),  # Orta-dış: aşağı
        ([70, 46], 0.12, -0.02),  # En dış uç: çok aşağı, hafif dışa
    ]
    
    right_brow_zones = [
        ([336, 285], -0.18, -0.02), # En iç
        ([296, 295], -0.10, -0.01), # Orta-iç
        ([334, 282], 0.02, 0.0),    # Tepe kavis
        ([293, 283], 0.08, 0.01),   # Orta-dış
        ([300, 276], 0.12, 0.02),   # En dış uç
    ]
    
    for zone in left_brow_zones:
        for idx in zone[0]:
            deltas[idx, 1] += face_sz * zone[1] # y ekseni (eksi=yukarı, artı=aşağı)
            deltas[idx, 0] += face_sz * zone[2] # x ekseni
            
    for zone in right_brow_zones:
        for idx in zone[0]:
            deltas[idx, 1] += face_sz * zone[1]
            deltas[idx, 0] += face_sz * zone[2]

    # Kapalı Gülümseme (Mouth Warping)
    # Dudak köşelerini ve hem üst hem alt dudak hattını *beraber* bükerek ağzı kapalı tut ve güçlü bir gülümseme oluştur
    smile_left = [
        ([61], -0.10, -0.06),             # En sol köşe: Çok güçlü yukarı ve dışa
        ([40, 146], -0.06, -0.03),        # Köşenin yanı (Üst ve Alt dudak beraber): Yukarı ve dışa
        ([39, 91], -0.03, -0.01),         # İçeri doğru
        ([37, 181], -0.01, 0.0)           # Merkeze yaklaşırken etki azalır
    ]
    for pts, y_force, x_force in smile_left:
        for idx in pts:
            deltas[idx, 1] += face_sz * y_force  # y ekseni (eksi=yukarı)
            deltas[idx, 0] += face_sz * x_force  # x ekseni (eksi=sola)
            
    smile_right = [
        ([291], -0.10, 0.06),             # En sağ köşe: Çok güçlü yukarı ve dışa
        ([270, 375], -0.06, 0.03),        # Köşenin yanı (Üst ve Alt dudak beraber)
        ([269, 321], -0.03, 0.01),        # İçeri doğru
        ([267, 405], -0.01, 0.0)          # Merkeze yaklaşırken
    ]
    for pts, y_force, x_force in smile_right:
        for idx in pts:
            deltas[idx, 1] += face_sz * y_force  # y ekseni (eksi=yukarı)
            deltas[idx, 0] += face_sz * x_force  # x ekseni (artı=sağa)

    anchors_zero = [10, 338, 297, 332, 284, 251, 168, 6, 197, 195, 5, 4, 152]
    for idx in anchors_zero: deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm, boundary]), np.vstack([dst, boundary]))

    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
        mask = _create_extended_face_mask(warped, wlms, 0.35)
    except Exception:
        mask = np.ones((h, w), np.float32) * 0.4
        wlms = None

    if wlms:
        # 1. Yanak Işık Yansıması (Blush Effect)
        left_cheek_idx = 205
        right_cheek_idx = 425
        
        def _get_wlm_pt(idx):
            if idx < len(wlms):
                return int(wlms[idx]["x"]*w), int(wlms[idx]["y"]*h)
            return None
            
        lc = _get_wlm_pt(left_cheek_idx)
        rc = _get_wlm_pt(right_cheek_idx)
        
        blush_radius = int(face_sz * 0.20)
        blush_mask = np.zeros((h, w), dtype=np.float32)
        
        if lc: cv2.circle(blush_mask, lc, blush_radius, 1.0, -1)
        if rc: cv2.circle(blush_mask, rc, blush_radius, 1.0, -1)
        
        k_size = int(blush_radius * 1.5)
        if k_size % 2 == 0: k_size += 1
        blush_mask = cv2.GaussianBlur(blush_mask, (k_size, k_size), 0)
        
        blush_color = np.full_like(warped, (40, 40, 255), dtype=np.uint8) # Yanaklar için neon kırmızı yansıma
        blush_alpha = blush_mask[..., np.newaxis] * 0.45 # Yarı saydam harmanlama
        
        warped = (warped.astype(np.float32) * (1.0 - blush_alpha) + blush_color.astype(np.float32) * blush_alpha).astype(np.uint8)

        # 2. Kırmızı Dudaklar
        warped = _apply_lip_color(warped, wlms, (10, 10, 240), 0.50)
        
        # 3. Büyük Neon Kalpler
        warped = _place_heart_masks(warped, wlms)

    return warped


# ── 6. CRYING PRESET (v2 – DSP-Aware Refraction Tears) ───────────────────────


def _build_teardrop_mask(
    h: int, w: int,
    anchor_x: int, anchor_y: int,
    tear_radius: int, tear_length: int,
) -> np.ndarray:
    """Create a smooth float32 teardrop mask (0..1).

    Shape: rounded circle at top → elongated tapered tail below.
    All drawing uses cv2 primitives (no pixel loops).
    """
    mask = np.zeros((h, w), dtype=np.float32)

    # Upper bulb (circle)
    cv2.circle(mask, (anchor_x, anchor_y + tear_radius), tear_radius, 1.0, -1, cv2.LINE_AA)

    # Lower tapered tail (triangle)
    tri = np.array([
        [anchor_x - tear_radius, anchor_y + tear_radius],
        [anchor_x + tear_radius, anchor_y + tear_radius],
        [anchor_x, anchor_y + tear_radius + tear_length],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, tri, 1.0, cv2.LINE_AA)

    # Smooth edges for natural appearance
    k = max(3, (tear_radius // 2) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), tear_radius * 0.3)
    # Re-normalize to [0, 1]
    mx = mask.max()
    if mx > 0:
        mask /= mx
    return mask


def _apply_eye_redness(
    image: np.ndarray,
    landmarks: list,
    redness_color: tuple = (10, 10, 50),
    opacity: float = 0.35,
) -> np.ndarray:
    """Apply a faint Gaussian-blurred reddish tint around both eye ROIs.

    Simulates the soreness / irritation of crying.  Uses a smooth alpha
    gradient so the redness fades naturally into surrounding skin.
    """
    h, w = image.shape[:2]

    # Eye landmark indices (upper + lower eyelid contour)
    left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    result = image.astype(np.float32) / 255.0
    red_layer = np.array(redness_color, dtype=np.float32) / 255.0

    for eye_indices in [left_eye_idx, right_eye_idx]:
        pts = []
        for idx in eye_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append([int(lm["x"] * w), int(lm["y"] * h)])
        if len(pts) < 4:
            continue

        pts_arr = np.array(pts, dtype=np.int32)
        # Expand ROI for the redness halo
        cx = int(np.mean(pts_arr[:, 0]))
        cy = int(np.mean(pts_arr[:, 1]))
        rx = int(np.std(pts_arr[:, 0]) * 2.5)
        ry = int(np.std(pts_arr[:, 1]) * 3.0)

        # Create elliptical gradient mask
        eye_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(eye_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1, cv2.LINE_AA)

        # Heavy blur for soft gradient
        blur_k = max(15, (rx | 1))
        if blur_k % 2 == 0:
            blur_k += 1
        eye_mask = cv2.GaussianBlur(eye_mask, (blur_k, blur_k), rx * 0.4)
        # Normalize
        mx = eye_mask.max()
        if mx > 0:
            eye_mask /= mx

        # Alpha blend redness
        alpha = eye_mask[..., np.newaxis] * opacity
        result = result * (1.0 - alpha) + red_layer * alpha

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _apply_refraction_tears(
    image: np.ndarray,
    landmarks: list,
) -> np.ndarray:
    """Procedural refraction tears with pixel displacement + specular highlight.

    Academic DSP Trick
    ------------------
    Instead of painting flat blue colour, we DISPLACE the underlying pixels
    based on the tear's convex shape (simulated normal map).  This creates
    the visual illusion of clear liquid refracting the face behind it.

    Additionally, a small sharp white specular highlight is placed at the
    convex top of each teardrop for realistic light reflection.
    """
    h, w = image.shape[:2]
    result = image.astype(np.float32)

    # Tear anchor points: inner-eye lower lids + outer-eye corners
    # Using inner corner (133, 362) and lower lid mid-point (145, 374)
    tear_anchors = []
    for lid_idx, corner_idx in [(145, 133), (374, 362)]:
        if lid_idx >= len(landmarks) or corner_idx >= len(landmarks):
            continue
        # Primary tear from lower lid center
        lx = int(landmarks[lid_idx]["x"] * w)
        ly = int(landmarks[lid_idx]["y"] * h)
        tear_anchors.append((lx, ly, "primary"))
        # Secondary smaller tear from inner corner
        cx = int(landmarks[corner_idx]["x"] * w)
        cy = int(landmarks[corner_idx]["y"] * h)
        tear_anchors.append((cx, cy, "secondary"))

    for ax, ay, tear_type in tear_anchors:
        if tear_type == "primary":
            tear_r = max(4, int(h * 0.016))
            tear_len = max(12, int(h * 0.08))
            refract_strength = 4.0
        else:
            tear_r = max(3, int(h * 0.010))
            tear_len = max(8, int(h * 0.05))
            refract_strength = 2.5

        # Build teardrop mask
        tear_mask = _build_teardrop_mask(h, w, ax, ay, tear_r, tear_len)

        # ── Pixel Displacement (Refraction) ──────────────────────────────
        # Build displacement field based on tear shape gradient
        # The gradient of the mask approximates the "normal" of the
        # tear's convex surface — pixels are shifted along this gradient.
        grad_x = cv2.Sobel(tear_mask, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(tear_mask, cv2.CV_32F, 0, 1, ksize=3)

        # Create coordinate grids
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        # Displace source coordinates by the gradient (refraction)
        map_x = xx - grad_x * refract_strength
        map_y = yy - grad_y * refract_strength

        # Clamp to image bounds
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # Apply displacement via remap (vectorized, no pixel loops)
        refracted = cv2.remap(
            result.astype(np.uint8), map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)

        # Composite: where tear_mask > 0, use refracted pixels
        tear_alpha = tear_mask[..., np.newaxis]

        # Slight brightness boost inside the tear (wet/glossy look)
        brightness_boost = 1.08
        refracted_bright = np.clip(refracted * brightness_boost, 0, 255)

        # Very faint blue-ish tint to hint at water (subtle, not flat paint)
        water_tint = np.array([245.0, 230.0, 220.0], dtype=np.float32)  # very pale cyan/blue
        tinted_refracted = refracted_bright * 0.92 + water_tint * 0.08

        result = result * (1.0 - tear_alpha) + tinted_refracted * tear_alpha

        # ── Specular Highlight ───────────────────────────────────────────
        # Sharp white dot at the convex top of the teardrop
        spec_x = ax
        spec_y = ay + max(2, tear_r // 3)
        spec_r = max(2, tear_r // 3)

        spec_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(spec_mask, (spec_x, spec_y), spec_r, 1.0, -1, cv2.LINE_AA)
        # Slight blur for soft specular
        sk = max(3, spec_r | 1)
        spec_mask = cv2.GaussianBlur(spec_mask, (sk, sk), spec_r * 0.3)
        mx = spec_mask.max()
        if mx > 0:
            spec_mask /= mx

        spec_alpha = spec_mask[..., np.newaxis] * 0.85
        white = np.full_like(result, 255.0)
        result = result * (1.0 - spec_alpha) + white * spec_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_crying(image: np.ndarray) -> np.ndarray:
    """😢 Crying v2 — DSP-Aware Refraction Tears + Eye Redness.

    Pipeline
    --------
    1. Geometric warp: sad brows (inner UP, outer DOWN) + mouth droop
    2. Eye redness: faint Gaussian red tint around eye ROIs
    3. Refraction tears: pixel displacement via gradient-based refraction
    4. Specular highlights: sharp white at tear convex top
    """
    out = image.copy()
    h, w = out.shape[:2]
    lm = _get_stable_landmarks("crying", out)
    if lm is None:
        return out
    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # ── Sad/worried arched brows: inner UP, outer DOWN (strong) ──
    for idx in [107, 55, 336, 285]:
        deltas[idx, 1] -= face_sz * 0.10
    for idx in [70, 46, 300, 276]:
        deltas[idx, 1] += face_sz * 0.07

    # ── Mouth corners droop ──
    deltas[61, 1] += face_sz * 0.08
    deltas[291, 1] += face_sz * 0.08
    for idx in [146, 91]:
        deltas[idx, 1] += face_sz * 0.05
    for idx in [375, 321]:
        deltas[idx, 1] += face_sz * 0.05

    # ── Slightly squint eyes (narrow them vertically for crying look) ──
    upper_lid_l = [159, 160, 161]
    lower_lid_l = [144, 145, 153]
    upper_lid_r = [386, 385, 384]
    lower_lid_r = [373, 374, 380]
    squint = face_sz * 0.025
    for idx in upper_lid_l + upper_lid_r:
        deltas[idx, 1] += squint
    for idx in lower_lid_l + lower_lid_r:
        deltas[idx, 1] -= squint * 0.5

    # ── Lock boundary anchors ──
    anchors_zero = [10, 338, 297, 332, 284, 251, 168, 6, 197, 195, 5, 4, 152]
    for idx in anchors_zero:
        deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.05] = 0.0

    # ── Single-pass warp with dense boundary anchors ──
    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm)
    warped = geometric_warp(out, np.vstack([lm, boundary]), np.vstack([dst, boundary]))

    # ── Get fresh landmarks on warped result ──
    try:
        prep = preprocess_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        wlms = get_landmarks(prep)
    except Exception:
        wlms = None

    if wlms is not None:
        # Stage 2: Eye redness
        warped = _apply_eye_redness(warped, wlms, redness_color=(10, 10, 50), opacity=0.35)

        # Stage 3 & 4: Refraction tears + specular highlights
        warped = _apply_refraction_tears(warped, wlms)

    return warped


# ── Preset dispatcher ────────────────────────────────────────────────────────
_EMOJI_PRESETS_MAP = {
    "alien": apply_alien_emoji,
    "robot": _apply_robot,
    "clown": apply_clown_emoji,
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


@router.post("/process/fft")
async def process_fft(
    image: UploadFile = File(...),
    intensity: float = Form(50),
    mask_coords: str | None = Form(None),
):
    """
    Apply FFT-based frequency filter to the uploaded image.

    Parameters
    ----------
    intensity : float
        Filter strength (0-100).
    mask_coords : str | None
        Optional JSON string with normalized coordinates
        {"x": float, "y": float, "w": float, "h": float} (0-1 range).
        If provided, the filter is applied only to that region.
    """
    import json as _json

    try:
        contents = await image.read()
        original = _decode_upload(contents)

        # Parse optional mask coordinates
        parsed_mask = None
        if mask_coords:
            try:
                parsed_mask = _json.loads(mask_coords)
                logger.info("[FFT] mask_coords received: %s", parsed_mask)
            except (_json.JSONDecodeError, ValueError) as exc:
                logger.warning("[FFT] Could not parse mask_coords: %s", exc)

        filter_intensity = float(intensity)
        if parsed_mask:
            processed, _ = apply_fft_partial_region_artifact(
                original,
                mask_coords=parsed_mask,
                intensity=filter_intensity,
            )
        else:
            processed, _ = apply_fft_filter(original, intensity=filter_intensity)

        metrics = _metrics_dict(original, processed)

        # Compute spectra for both original and processed
        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        orig_spectrum_b64 = _data_url_from_image(
            cv2.cvtColor(compute_magnitude_spectrum(orig_fft_shifted), cv2.COLOR_GRAY2BGR)
        )
        proc_spectrum_b64 = _data_url_from_image(
            cv2.cvtColor(compute_magnitude_spectrum(proc_fft_shifted), cv2.COLOR_GRAY2BGR)
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
            energy=compute_energy_analysis(processed, radius=30),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[FFT] Processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# 🤡  CLOWN TRANSFORMATION  (High-Quality, Integrated Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def apply_clown_transformation(image: np.ndarray) -> np.ndarray:
    """
    🤡 Joker Clown Transformation (High-Quality, Integrated Pipeline)
    
    Stage 1 — Single-pass combined geometric warp
        • Smile warp (corners up+out)
        • Eye enlarge
    Stage 2 — Greasepaint-white face paint (texture preserving)
    Stage 3 — Joker clown details
        • Rhombus blue eyes
        • Red brows
        • Big red nose
        • Wide red smile lines + filled lips
    """
    result_img = image.copy()
    h, w = result_img.shape[:2]

    # ── Stage 0: detect landmarks ──
    lm = detect_face_landmarks(result_img)
    if lm is None:
        logger.warning("apply_clown_transformation: no face detected")
        return result_img

    face_sz = _face_scale(lm)
    deltas = np.zeros_like(lm)

    # ── Stage 1a: Smile delta ──
    smile_strength = 0.14
    sigma_smile = face_sz * 0.28
    w_left = np.exp(-0.5 * (np.linalg.norm(lm - lm[61], axis=1) / sigma_smile) ** 2)
    w_right = np.exp(-0.5 * (np.linalg.norm(lm - lm[291], axis=1) / sigma_smile) ** 2)
    center_x = (lm[61, 0] + lm[291, 0]) / 2.0
    half_w = max(abs(lm[291, 0] - lm[61, 0]) / 2.0, 1e-6)

    for i in range(len(lm)):
        deltas[i, 0] += w_left[i] * (-face_sz * smile_strength * 0.60)
        deltas[i, 0] += w_right[i] * (face_sz * smile_strength * 0.60)
        dy_damp = 1.0 - np.exp(-0.5 * ((lm[i, 0] - center_x) / (half_w * 0.6)) ** 2)
        deltas[i, 1] += (w_left[i] + w_right[i]) * (-face_sz * smile_strength * 0.90) * dy_damp

    # ── Stage 1b: Eye-enlarge delta ──
    eye_factor = 0.75
    sigma_eye = face_sz * 0.14
    left_ring = [33, 133, 160, 158, 153, 144, 159, 145]
    right_ring = [362, 263, 387, 385, 380, 373, 386, 374]
    cl = np.mean(lm[left_ring], axis=0)
    cr = np.mean(lm[right_ring], axis=0)

    wl = np.exp(-0.5 * (np.linalg.norm(lm - cl, axis=1) / max(sigma_eye, 1e-6)) ** 2)
    wr = np.exp(-0.5 * (np.linalg.norm(lm - cr, axis=1) / max(sigma_eye, 1e-6)) ** 2)

    for i in range(len(lm)):
        deltas[i] += (lm[i] - cl) * eye_factor * wl[i]
        deltas[i] += (lm[i] - cr) * eye_factor * wr[i]

    # ── Stage 1c: Lock boundary anchors ──
    anchors_lock = [
        10, 338, 297, 332, 284, 251,
        152, 377, 400, 378, 379, 365, 397,
        70, 63, 105, 66, 107, 46, 53, 52,
        300, 293, 334, 296, 336, 276, 283, 282,
        168, 6, 197, 195, 5, 4,
    ]
    for idx in anchors_lock:
        deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.08] = 0.0

    # ── Stage 1d: Single-pass warp ──
    dst = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm, spacing=38)
    src_all = np.vstack([lm, boundary])
    dst_all = np.vstack([dst, boundary])
    warped = geometric_warp(result_img, src_all, dst_all)

    # ── Stage 2: Greasepaint-white face paint ──
    lm2 = detect_face_landmarks(warped)
    if lm2 is None:
        return warped
    face_sz2 = _face_scale(lm2)

    face_oval_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ]
    face_pts = np.array([[int(lm2[i][0]), int(lm2[i][1])] for i in face_oval_idx if i < len(lm2)], dtype=np.int32)
    if len(face_pts) >= 3:
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(face_pts), 255)
        face_mask = cv2.GaussianBlur(face_mask, (51, 51), 18)
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 7)

        white = np.full_like(warped, (250, 250, 248))
        alpha = face_mask.astype(np.float32) / 255.0 * 0.62
        a3 = alpha[..., np.newaxis]
        warped = (warped.astype(np.float32) * (1.0 - a3) + white.astype(np.float32) * a3).astype(np.uint8)

    # ── Stage 3: Joker details ──
    paint = np.zeros((h, w, 3), dtype=np.float32)

    def _px(idx):
        return int(lm2[idx][0]), int(lm2[idx][1])

    # 3a. Rhombus Blue Eyes
    le_cx, le_cy = int((lm2[33][0] + lm2[133][0])/2), int((lm2[33][1] + lm2[133][1])/2)
    re_cx, re_cy = int((lm2[362][0] + lm2[263][0])/2), int((lm2[362][1] + lm2[263][1])/2)
    e_r = int(face_sz2 * 0.22)
    def rhombus_pts(cx, cy, r):
        return np.array([[cx - r, cy], [cx, cy - r], [cx + r, cy], [cx, cy + r]], dtype=np.int32)
    cv2.fillPoly(paint, [rhombus_pts(le_cx, le_cy, e_r)], (210, 90, 10))
    cv2.fillPoly(paint, [rhombus_pts(re_cx, re_cy, e_r)], (210, 90, 10))

    # 3b. Red Brows
    lb_pts = np.array([list(_px(i)) for i in [70, 63, 105, 66, 107, 55, 65, 52, 53, 46] if i < len(lm2)], dtype=np.int32)
    rb_pts = np.array([list(_px(i)) for i in [300, 293, 334, 296, 336, 285, 295, 282, 283, 276] if i < len(lm2)], dtype=np.int32)
    brow_thick = max(int(face_sz2 * 0.06), 3)
    if len(lb_pts) > 0: cv2.polylines(paint, [lb_pts], False, (0, 0, 220), brow_thick)
    if len(rb_pts) > 0: cv2.polylines(paint, [rb_pts], False, (0, 0, 220), brow_thick)

    # 3c. Big Red Nose
    if 4 < len(lm2):
        nx, ny = _px(4)
        nose_r = int(face_sz2 * 0.20)
        cv2.circle(paint, (nx, ny), nose_r, (0, 0, 240), -1)
        cv2.circle(paint, (nx - int(nose_r * 0.3), ny - int(nose_r * 0.35)), int(nose_r * 0.22), (100, 100, 255), -1)

    # 3d. Big Red Lips
    outer_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    om_pts = np.array([list(_px(i)) for i in outer_mouth if i < len(lm2)], dtype=np.int32)
    if len(om_pts) > 0:
        om_center = om_pts.mean(axis=0).astype(int)
        om_big = ((om_pts - om_center) * 1.35 + om_center).astype(np.int32)
        cv2.fillPoly(paint, [om_big], (0, 0, 225))

    # 3e. Wide Smile Lines
    if 61 < len(lm2) and 291 < len(lm2) and 205 < len(lm2) and 425 < len(lm2):
        left_corner = _px(61)
        right_corner = _px(291)
        left_cheek = (int(lm2[205][0] - face_sz2 * 0.20), int(lm2[205][1] + face_sz2 * 0.05))
        right_cheek = (int(lm2[425][0] + face_sz2 * 0.20), int(lm2[425][1] + face_sz2 * 0.05))
        line_w = max(int(face_sz2 * 0.08), 4)
        cv2.line(paint, left_corner, left_cheek, (0, 0, 225), line_w)
        cv2.line(paint, right_corner, right_cheek, (0, 0, 225), line_w)

    # Blend Joker Paint
    paint_blur = cv2.GaussianBlur(paint, (9, 9), 0)
    paint_alpha = np.clip(paint_blur.sum(axis=2, keepdims=True) / 280.0, 0, 1)
    paint_alpha = np.repeat(paint_alpha, 3, axis=2)

    final = (
        paint_blur * paint_alpha * 0.85
        + warped.astype(np.float32) * (1.0 - paint_alpha * 0.85)
    ).astype(np.uint8)

    return final

@router.post("/process/clown_transformation")
async def process_clown_transformation(image: UploadFile = File(...)):
    """
    High-quality clown face transformation using Joker logic.
    """
    try:
        contents = await image.read()
        file_bytes = np.frombuffer(contents, np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
        
        processed = apply_clown_transformation(original)
        
        metrics = _metrics_dict(original, processed)
        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]
        orig_spectrum = compute_magnitude_spectrum(orig_fft_shifted)
        proc_spectrum = compute_magnitude_spectrum(proc_fft_shifted)

        return {
            "proc_image_b64": _data_url_from_image(processed),
            "image_b64": _data_url_from_image(processed),
            "metrics": metrics,
            "orig_spectrum_b64": _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR)),
            "proc_spectrum_b64": _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR)),
        }
    except Exception as exc:
        logger.exception("process_clown_transformation.failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.websocket("/process/ws_live")
async def process_ws_live(websocket: WebSocket):
    """
    Overhauled Live WebSocket router using Layered Pipeline (Stacking) logic.
    Multiple effects run simultaneously on the same frame.
    """
    await websocket.accept()
    logger.info("process_ws_live connected.")
    
    import time
    import asyncio
    import json
    import base64
    
    try:
        from modules.warping_module import (
            apply_smile, apply_lip_widen, apply_face_slim, apply_eye_scaling,
            apply_beard, detect_face_landmarks
        )
        from modules.hair_module import apply_hair_color
        from modules.glasses_module import apply_glasses
        from modules.frequency_module import apply_virtual_makeup
        from modules.frequency_module import apply_cartoon_filter, apply_fft_filter
    except ModuleNotFoundError:
        from backend.modules.warping_module import (
            apply_smile, apply_lip_widen, apply_face_slim, apply_eye_scaling,
            apply_beard, detect_face_landmarks
        )
        from backend.modules.hair_module import apply_hair_color
        from backend.modules.glasses_module import apply_glasses
        from backend.modules.frequency_module import apply_virtual_makeup
        from backend.modules.frequency_module import apply_cartoon_filter, apply_fft_filter

    active_states = {
        "smile": 0.0,
        "lip_widen": 0.0,
        "face_slim": 0.0,
        "eye_scaling": 0.0,
        "hair_color": None,        # string like "255,0,0"
        "hair_intensity": 0.6,
        "makeup_lips": None,       # int hue
        "makeup_opacity": 0.5,
        "beard": None,             # string "beard" or "goatee"
        "beard_intensity": 50.0,
        "glasses": None,           # string "aviator"
        "emoji": None,             # string "clown", "alien"
        "show_landmarks": False,
        "cartoon": False,
        "fft": 0.0
    }

    def _draw_landmarks(img, lm_array):
        if lm_array is not None:
            for pt in lm_array:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        return img

    last_fps_time = time.perf_counter()
    fps_frame_count = 0
    fps = 0.0

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "config":
                # Handle single filter legacy format if sent
                f = msg.get("filter")
                if f:
                    if f in ["smile", "lip_widen", "face_slim", "eye_scaling"]:
                        active_states[f] = msg.get("intensity", 50.0)
                    elif f == "hair_color":
                        active_states["hair_color"] = msg.get("hair_color", "255,0,0")
                        active_states["hair_intensity"] = msg.get("hair_intensity", 0.6)
                    elif f.startswith("makeup_"):
                        active_states["makeup_lips"] = msg.get("makeup_hue", 0)
                        active_states["makeup_opacity"] = msg.get("makeup_opacity", 0.5)
                    elif f == "beard":
                        active_states["beard"] = msg.get("beard_type", "beard")
                        active_states["beard_intensity"] = msg.get("intensity", 50.0)
                    elif f == "glasses":
                        active_states["glasses"] = msg.get("glasses_type", "aviator")
                    elif f in ["alien", "robot", "clown", "star_eyes", "heart_eyes", "crying"]:
                        active_states["emoji"] = f
                    elif f == "cartoon":
                        active_states["cartoon"] = True

                # Handle explicit toggles in the payload (UI sliders mapping directly)
                for key in active_states:
                    if key in msg:
                        active_states[key] = msg[key]

                # Special case: if UI sends nested makeup
                if "makeup_hue" in msg: active_states["makeup_lips"] = msg["makeup_hue"]
                if "makeup_opacity" in msg: active_states["makeup_opacity"] = msg["makeup_opacity"]
                if "glasses_type" in msg: active_states["glasses"] = msg["glasses_type"]
                if "beard_type" in msg: active_states["beard"] = msg["beard_type"]

                logger.info(f"Live config updated active_states: {active_states}")
                await websocket.send_json({"type": "status", "message": "Config applied"})
                continue

            elif msg_type == "frame":
                start_t = time.perf_counter()
                frame_data = msg.get("data", "")
                
                # Decode base64 frame
                if "," in frame_data:
                    frame_data = frame_data.split(",", 1)[1]
                try:
                    raw = base64.b64decode(frame_data)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    frame = None

                if frame is None:
                    continue

                # Detect landmarks once for the baseline frame
                landmarks = detect_face_landmarks(frame)
                result = frame.copy()
                
                if landmarks is not None:
                    # Stage 1: Geometric Warps
                    if active_states.get("eye_scaling"):
                        result = apply_eye_scaling(result, float(active_states["eye_scaling"]), landmarks=landmarks)
                        landmarks = detect_face_landmarks(result) or landmarks
                    if active_states.get("smile"):
                        result = apply_smile(result, float(active_states["smile"]), landmarks=landmarks)
                        landmarks = detect_face_landmarks(result) or landmarks
                    if active_states.get("lip_widen"):
                        result = apply_lip_widen(result, float(active_states["lip_widen"]), landmarks=landmarks)
                        landmarks = detect_face_landmarks(result) or landmarks
                    if active_states.get("face_slim"):
                        result = apply_face_slim(result, float(active_states["face_slim"]), landmarks=landmarks)
                        landmarks = detect_face_landmarks(result) or landmarks

                    # Stage 2: Color/Texture Overlays
                    if active_states.get("hair_color"):
                        result = apply_hair_color(result, active_states["hair_color"], float(active_states["hair_intensity"]))
                    if active_states.get("makeup_lips") is not None:
                        h_f, w_f = result.shape[:2]
                        norm_lms = [[float(pt[0]) / w_f, float(pt[1]) / h_f] for pt in landmarks]
                        result = apply_virtual_makeup(
                            image=result, landmarks=norm_lms, region="lips",
                            hue=int(active_states["makeup_lips"]), opacity=float(active_states["makeup_opacity"])
                        )
                    if active_states.get("beard"):
                        result = apply_beard(result, float(active_states["beard_intensity"]), landmarks=landmarks)

                    # Stage 3: AR Assets
                    if active_states.get("glasses"):
                        h_f, w_f = result.shape[:2]
                        norm_lms = [[float(pt[0]) / w_f, float(pt[1]) / h_f] for pt in landmarks]
                        result = apply_glasses(result, norm_lms, active_states["glasses"])
                        
                    if active_states.get("emoji"):
                        emoji_name = active_states["emoji"]
                        if emoji_name in _EMOJI_PRESETS_MAP:
                            result = _EMOJI_PRESETS_MAP[emoji_name](result)

                    # Stage 4: Debug Overlays
                    if active_states.get("show_landmarks"):
                        result = _draw_landmarks(result, landmarks)

                if active_states.get("cartoon"):
                    result = apply_cartoon_filter(result)
                if active_states.get("fft", 0.0) > 0.0:
                    result, _ = apply_fft_filter(result, float(active_states["fft"]))

                # FPS tracking
                fps_frame_count += 1
                now = time.perf_counter()
                if now - last_fps_time >= 1.0:
                    fps = fps_frame_count / (now - last_fps_time)
                    fps_frame_count = 0
                    last_fps_time = now

                # Encode to send back
                _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64 = base64.b64encode(buf).decode("ascii")
                encoded = f"data:image/jpeg;base64,{b64}"

                await websocket.send_json({
                    "type": "frame",
                    "data": encoded,
                    "fps": round(fps, 1),
                    "face_detected": landmarks is not None
                })

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error(f"process_ws_live error: {exc}", exc_info=True)
    finally:
        logger.info("process_ws_live disconnected.")
