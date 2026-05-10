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
    # Bypass FFT spectra computation for faster live mode
    skip_spectra: bool = False


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


# Categories that benefit from downsample processing
_GEOMETRY_HEAVY = {"smile", "eyebrow", "lip", "slim",
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

    # Decode image
    try:
        original = _decode_base64_image(body.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Image decode failed: {exc}") from exc

    try:
        # ── Route to the correct filter function ─────────────────────────
        if fname in WARP_OPS:
            apply_fn = _build_warp_fn(fname, body.intensity, body.smoothing)
        elif fname in _EMOJI_PRESETS_MAP:
            apply_fn = _EMOJI_PRESETS_MAP[fname]
        elif fname in AGE_OPS:
            apply_fn = _build_age_fn(fname, body.intensity)
        elif fname == "hair_color" and body.hair_color:
            apply_fn = _build_hair_fn(body.hair_color, body.hair_intensity)
        elif fname == "cartoon":
            apply_fn = apply_cartoon_filter
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown filter '{fname}'. Valid: "
                       f"{', '.join(sorted(set(WARP_OPS) | set(_EMOJI_PRESETS_MAP) | AGE_OPS | {'hair_color', 'cartoon'}))}"
            )

        # ── Apply with optional downsampling for geometry-heavy ops ──────
        if fname in _GEOMETRY_HEAVY:
            processed = _process_lo_composite(original, apply_fn,
                                               needs_mask=True)
        else:
            processed = apply_fn(original)

        # ── Build response ───────────────────────────────────────────────
        metrics = _metrics_dict(original, processed)

        # Skip expensive FFT spectra in live mode
        if body.skip_spectra:
            return {
                "processed_image": _data_url_from_image(processed),
                "image_b64": _data_url_from_image(processed),
                "metrics": metrics,
            }

        orig_fft_shifted = compute_fft(original)[2]
        proc_fft_shifted = compute_fft(processed)[2]

        return _response_payload(
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
        )

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
    """
    🤡 Clown — 3-aşamalı sıkı pipeline:

    Aşama 1 — Geometrik Warp (Yüz şekli değişir)
        a) Abartılı gülümseme: apply_smile(intensity=60)
        b) Göz büyütme: apply_eye_scaling(intensity=55)

    Aşama 2 — Renk & Maske (Boya)
        Yüz konturunu al → beyaz maske → %65 beyaz / %35 orijinal doku

    Aşama 3 — Çizim (En Son)
        Yamultulmuş dudak hattını al → kırmızı polylines çiz
    """
    result_img = image.copy()
    h, w = result_img.shape[:2]

    # ── AŞAMA 1a: Gülümseme Warpi ──────────────────────────────────────────
    result_img = apply_smile(result_img, intensity=60)

    # ── AŞAMA 1b: Göz Büyütme ──────────────────────────────────────────────
    result_img = apply_eye_scaling(result_img, intensity=55)

    # ── AŞAMA 2: Beyaz Yüz Maskesi ─────────────────────────────────────────
    lm_warped = detect_face_landmarks(result_img)
    if lm_warped is not None:
        face_oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109,
        ]
        face_pts = np.array(
            [[int(lm_warped[i][0]), int(lm_warped[i][1])] for i in face_oval_idx
             if i < len(lm_warped)],
            dtype=np.int32,
        )

        if len(face_pts) >= 3:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 8)

            white_layer = np.full_like(result_img, 255)
            alpha = face_mask.astype(np.float32) / 255.0 * 0.65   # 65% beyaz
            alpha_3ch = alpha[..., np.newaxis]
            result_img = (
                result_img.astype(np.float32) * (1.0 - alpha_3ch)
                + white_layer.astype(np.float32) * alpha_3ch
            ).astype(np.uint8)

    # ── AŞAMA 3: Kırmızı Dudak Çerçevesi ───────────────────────────────────
    lm_final = detect_face_landmarks(result_img)
    if lm_final is not None:
        outer_lip_idx = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        ]
        lip_pts = []
        for idx in outer_lip_idx:
            if idx < len(lm_final):
                lip_pts.append([int(lm_final[idx][0]), int(lm_final[idx][1])])

        if len(lip_pts) >= 3:
            face_sz = _face_scale(lm_final)
            thickness = max(3, int(face_sz * 0.04))
            lip_arr = np.array(lip_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                result_img, [lip_arr],
                isClosed=True,
                color=(0, 0, 220),     # parlak kırmızı (BGR)
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            # Köşeleri yuvarlat
            for pt in lip_arr:
                cv2.circle(
                    result_img, (pt[0][0], pt[0][1]),
                    thickness // 2, (0, 0, 220), -1, cv2.LINE_AA,
                )

    return result_img


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
    lm = detect_face_landmarks(out)
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

        # Apply FFT filter (apply_fft_filter does not yet support mask;
        # coordinates are logged above and will be used when the module is extended)
        filter_intensity = float(intensity)
        processed = apply_fft_filter(original, intensity=filter_intensity)

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
    Integrated, convincing clown pipeline executed on a SINGLE result_img:

    Stage 1 — Single-pass combined geometric warp
        • Smile warp : mouth corners pulled up-and-out (Gaussian falloff)
        • Eye enlarge: radial zoom on both eye rings (Gaussian falloff)
        Both deltas computed on original landmarks and applied in ONE call to
        geometric_warp with dense boundary anchors → zero tearing.

    Stage 2 — Greasepaint-white face paint
        Skin texture is preserved via soft alpha blend (not opaque overlay).
        Face oval mask is feathered at the jaw/neck boundary for a natural fade.

    Stage 3 — Classic clown details (drawn on warped geometry)
        • Filled bright-red lip area   (fillPoly on outer lip contour)
        • Big solid red nose circle    (landmark 4)
        • Red cheek circles            (landmarks 50, 280)
        • Blue lower-eye liner lines   (under each eye – classic clown)
    """
    result_img = image.copy()
    h, w = result_img.shape[:2]

    # ── Stage 0: detect landmarks once on the original ──────────────────────
    lm = detect_face_landmarks(result_img)
    if lm is None:
        logger.warning("apply_clown_transformation: no face detected")
        return result_img

    face_sz = _face_scale(lm)
    deltas   = np.zeros_like(lm)

    # ── Stage 1a: Smile delta  (corners up + out) ───────────────────────────
    smile_strength = 0.14   # fraction of face_sz to move corners
    sigma_smile    = face_sz * 0.28

    w_left  = np.exp(-0.5 * (np.linalg.norm(lm - lm[61],  axis=1) / sigma_smile) ** 2)
    w_right = np.exp(-0.5 * (np.linalg.norm(lm - lm[291], axis=1) / sigma_smile) ** 2)

    center_x = (lm[61, 0] + lm[291, 0]) / 2.0
    half_w   = max(abs(lm[291, 0] - lm[61, 0]) / 2.0, 1e-6)

    for i in range(len(lm)):
        # horizontal spread
        deltas[i, 0] += w_left[i]  * (-face_sz * smile_strength * 0.60)
        deltas[i, 0] += w_right[i] * ( face_sz * smile_strength * 0.60)
        # vertical lift — damped toward center to avoid Joker effect
        dy_damp = 1.0 - np.exp(-0.5 * ((lm[i, 0] - center_x) / (half_w * 0.6)) ** 2)
        deltas[i, 1] += (w_left[i] + w_right[i]) * (-face_sz * smile_strength * 0.90) * dy_damp

    # ── Stage 1b: Eye-enlarge delta (radial zoom on both eye rings) ──────────
    eye_factor = 0.75
    sigma_eye  = face_sz * 0.14

    left_ring  = [33, 133, 160, 158, 153, 144, 159, 145]
    right_ring = [362, 263, 387, 385, 380, 373, 386, 374]
    cl = np.mean(lm[left_ring],  axis=0)
    cr = np.mean(lm[right_ring], axis=0)

    wl = np.exp(-0.5 * (np.linalg.norm(lm - cl, axis=1) / sigma_eye) ** 2)
    wr = np.exp(-0.5 * (np.linalg.norm(lm - cr, axis=1) / sigma_eye) ** 2)

    for i in range(len(lm)):
        deltas[i] += (lm[i] - cl) * eye_factor * wl[i]
        deltas[i] += (lm[i] - cr) * eye_factor * wr[i]

    # ── Stage 1c: Lock boundary anchors ─────────────────────────────────────
    anchors_lock = [
        10, 338, 297, 332, 284, 251,          # top forehead
        152, 377, 400, 378, 379, 365, 397,     # chin / jaw bottom
        70, 63, 105, 66, 107, 46, 53, 52,      # left brow
        300, 293, 334, 296, 336, 276, 283, 282, # right brow
        168, 6, 197, 195, 5, 4,                # nose bridge
    ]
    for idx in anchors_lock:
        deltas[idx] = 0.0
    deltas[np.abs(deltas) < 0.08] = 0.0

    # ── Stage 1d: Single-pass warp with dense boundary ───────────────────────
    dst      = lm + deltas
    boundary = _generate_warp_anchors(w, h, lm, spacing=38)
    src_all  = np.vstack([lm, boundary])
    dst_all  = np.vstack([dst, boundary])
    result_img = geometric_warp(result_img, src_all, dst_all)

    # ── Stage 2: Greasepaint-white face paint ────────────────────────────────
    lm2 = detect_face_landmarks(result_img)          # fresh landmarks after warp
    if lm2 is not None:
        face_oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109,
        ]
        face_pts = np.array(
            [[int(lm2[i][0]), int(lm2[i][1])]
             for i in face_oval_idx if i < len(lm2)],
            dtype=np.int32,
        )
        if len(face_pts) >= 3:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)
            # Heavy feathering for natural jaw/neck transition
            face_mask = cv2.GaussianBlur(face_mask, (51, 51), 18)
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 7)

            # Greasepaint: preserve skin texture — 62 % white tint
            white = np.full_like(result_img, (250, 250, 248))  # slightly warm white
            alpha = face_mask.astype(np.float32) / 255.0 * 0.62
            a3    = alpha[..., np.newaxis]
            result_img = (
                result_img.astype(np.float32) * (1.0 - a3)
                + white.astype(np.float32) * a3
            ).astype(np.uint8)
        lm_paint = lm2        # reuse for stage 3
    else:
        lm_paint = None

    # ── Stage 3: Classic clown details ───────────────────────────────────────
    lm3 = detect_face_landmarks(result_img)

    if lm3 is not None:
        face_sz3 = _face_scale(lm3)

        def _px(idx):
            return int(lm3[idx][0]), int(lm3[idx][1])

        # 3a — Filled bright-red lips  ────────────────────────────────────────
        outer_lip_idx = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        ]
        lip_pts = np.array(
            [list(_px(i)) for i in outer_lip_idx if i < len(lm3)],
            dtype=np.int32,
        )
        if len(lip_pts) >= 3:
            lip_mask  = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(lip_mask, [lip_pts], 255)
            lip_mask  = cv2.GaussianBlur(lip_mask, (5, 5), 2)
            lip_alpha = lip_mask.astype(np.float32) / 255.0 * 0.92
            lip_a3    = lip_alpha[..., np.newaxis]
            red_layer = np.full_like(result_img, (0, 10, 210))  # vivid red (BGR)
            result_img = (
                result_img.astype(np.float32) * (1.0 - lip_a3)
                + red_layer.astype(np.float32) * lip_a3
            ).astype(np.uint8)
            # thin outline for crispness
            thickness_lip = max(2, int(face_sz3 * 0.025))
            cv2.polylines(result_img, [lip_pts.reshape(-1, 1, 2)],
                          True, (0, 0, 180), thickness_lip, cv2.LINE_AA)

        # 3b — Big solid red nose ─────────────────────────────────────────────
        if 4 < len(lm3):
            nx, ny = _px(4)
            nose_r = int(face_sz3 * 0.10)
            # Glow ring
            cv2.circle(result_img, (nx, ny), nose_r + 4, (60, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(result_img, (nx, ny), nose_r,     (0,  0,  220), -1, cv2.LINE_AA)
            # Specular highlight
            cv2.circle(result_img,
                       (nx - nose_r // 4, ny - nose_r // 4),
                       max(2, nose_r // 4), (255, 255, 255), -1, cv2.LINE_AA)

        # 3c — Red cheek circles ──────────────────────────────────────────────
        cheek_r = int(face_sz3 * 0.13)
        for cheek_idx in [50, 280]:
            if cheek_idx < len(lm3):
                cx, cy = _px(cheek_idx)
                cheek_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(cheek_mask, (cx, cy), cheek_r, 255, -1)
                cheek_mask = cv2.GaussianBlur(cheek_mask, (cheek_r | 1, cheek_r | 1), cheek_r // 3)
                ca = cheek_mask.astype(np.float32) / 255.0 * 0.55
                ca3 = ca[..., np.newaxis]
                red_c = np.full_like(result_img, (30, 30, 220))
                result_img = (
                    result_img.astype(np.float32) * (1.0 - ca3)
                    + red_c.astype(np.float32) * ca3
                ).astype(np.uint8)

        # 3d — Classic blue lower-eye liner ───────────────────────────────────
        #   Draw a short vertical accent line below each eye (classic clown makeup)
        liner_color = (200, 80, 0)   # deep blue-teal in BGR
        liner_thick = max(2, int(face_sz3 * 0.025))
        liner_len   = int(face_sz3 * 0.12)

        for lower_lid_idx in [145, 374]:
            if lower_lid_idx < len(lm3):
                lx, ly = _px(lower_lid_idx)
                cv2.line(result_img,
                         (lx, ly),
                         (lx, ly + liner_len),
                         liner_color, liner_thick, cv2.LINE_AA)

    return result_img


@router.post("/process/clown_transformation")
async def process_clown_transformation(
    image: UploadFile = File(...),
):
    """
    High-quality clown face transformation.

    Single-pass integrated pipeline:
      1. Combined geometric warp (smile + eye enlargement)
      2. Greasepaint-white face paint (texture-preserving)
      3. Filled red lips, big red nose, red cheeks, blue eye liner
    """
    logger.info("process_clown_transformation.received")
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        processed = apply_clown_transformation(original)
        metrics   = _metrics_dict(original, processed)

        logger.info("process_clown_transformation.success")
        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("process_clown_transformation.failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
