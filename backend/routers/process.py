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
    glasses_type: str = Form("sunglasses"),
):
    """
    Overlay procedural glasses on the face using MediaPipe landmark positions.

    Parameters
    ----------
    glasses_type : str
        Either ``"sunglasses"`` or ``"reading"`` (numaralı gözlük).
    """
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        # Get landmarks for eye positioning
        rgb_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb_img)
        landmarks = get_landmarks(preprocessed)

        # Map landmark coords back to original image dimensions
        h, w = original.shape[:2]

        def lm_to_px(lm):
            return int(lm["x"] * w), int(lm["y"] * h)

        # Key landmark indices (MediaPipe FaceMesh):
        # Left eye outer: 33,  Left eye inner: 133
        # Right eye inner: 362, Right eye outer: 263
        # Nose bridge top: 6
        left_outer = lm_to_px(landmarks[33])
        left_inner = lm_to_px(landmarks[133])
        right_inner = lm_to_px(landmarks[362])
        right_outer = lm_to_px(landmarks[263])
        nose_bridge = lm_to_px(landmarks[6])

        # Compute eye centres & sizing
        left_eye_cx = (left_outer[0] + left_inner[0]) // 2
        left_eye_cy = (left_outer[1] + left_inner[1]) // 2
        right_eye_cx = (right_inner[0] + right_outer[0]) // 2
        right_eye_cy = (right_inner[1] + right_outer[1]) // 2

        eye_distance = abs(right_eye_cx - left_eye_cx)
        lens_w = int(eye_distance * 0.65)
        lens_h = int(lens_w * 0.7)

        # Compute tilt angle from eye positions
        angle_rad = np.arctan2(right_eye_cy - left_eye_cy, right_eye_cx - left_eye_cx)
        angle_deg = np.degrees(angle_rad)

        processed = original.copy()
        overlay = np.zeros_like(processed)
        mask = np.zeros(processed.shape[:2], dtype=np.uint8)

        gtype = (glasses_type or "").strip().lower()

        if gtype == "reading":
            # --- Reading glasses: thin wire frame + transparent lenses ---
            frame_color = (50, 50, 50)       # dark grey frame
            lens_tint = (200, 180, 160)       # very light blue-ish tint
            frame_thickness = max(2, int(eye_distance * 0.025))

            # Left lens ellipse
            cv2.ellipse(overlay, (left_eye_cx, left_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, lens_tint, -1)
            cv2.ellipse(mask, (left_eye_cx, left_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, 40, -1)
            cv2.ellipse(overlay, (left_eye_cx, left_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, frame_color, frame_thickness)
            cv2.ellipse(mask, (left_eye_cx, left_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, 255, frame_thickness)

            # Right lens ellipse
            cv2.ellipse(overlay, (right_eye_cx, right_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, lens_tint, -1)
            cv2.ellipse(mask, (right_eye_cx, right_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, 40, -1)
            cv2.ellipse(overlay, (right_eye_cx, right_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, frame_color, frame_thickness)
            cv2.ellipse(mask, (right_eye_cx, right_eye_cy), (lens_w // 2, lens_h // 2),
                        angle_deg, 0, 360, 255, frame_thickness)

            # Bridge between lenses
            cv2.line(overlay, (left_inner[0], left_eye_cy), (right_inner[0], right_eye_cy),
                     frame_color, frame_thickness)
            cv2.line(mask, (left_inner[0], left_eye_cy), (right_inner[0], right_eye_cy),
                     255, frame_thickness)

            # Temple arms (from outer edges outward)
            arm_len = int(eye_distance * 0.4)
            cv2.line(overlay,
                     (left_outer[0], left_eye_cy),
                     (left_outer[0] - arm_len, left_eye_cy - int(lens_h * 0.15)),
                     frame_color, frame_thickness)
            cv2.line(mask,
                     (left_outer[0], left_eye_cy),
                     (left_outer[0] - arm_len, left_eye_cy - int(lens_h * 0.15)),
                     255, frame_thickness)
            cv2.line(overlay,
                     (right_outer[0], right_eye_cy),
                     (right_outer[0] + arm_len, right_eye_cy - int(lens_h * 0.15)),
                     frame_color, frame_thickness)
            cv2.line(mask,
                     (right_outer[0], right_eye_cy),
                     (right_outer[0] + arm_len, right_eye_cy - int(lens_h * 0.15)),
                     255, frame_thickness)

        else:
            # --- Sunglasses: thick frames + dark tinted lenses ---
            frame_color = (20, 20, 20)    # near-black
            lens_color = (40, 35, 30)     # dark brown tint
            frame_thickness = max(3, int(eye_distance * 0.04))

            # Slightly larger lenses for sunglasses
            sg_lens_w = int(lens_w * 1.15)
            sg_lens_h = int(lens_h * 1.1)

            # Left lens
            cv2.ellipse(overlay, (left_eye_cx, left_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, lens_color, -1)
            cv2.ellipse(mask, (left_eye_cx, left_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, 200, -1)
            cv2.ellipse(overlay, (left_eye_cx, left_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, frame_color, frame_thickness)
            cv2.ellipse(mask, (left_eye_cx, left_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, 255, frame_thickness)

            # Right lens
            cv2.ellipse(overlay, (right_eye_cx, right_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, lens_color, -1)
            cv2.ellipse(mask, (right_eye_cx, right_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, 200, -1)
            cv2.ellipse(overlay, (right_eye_cx, right_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, frame_color, frame_thickness)
            cv2.ellipse(mask, (right_eye_cx, right_eye_cy), (sg_lens_w // 2, sg_lens_h // 2),
                        angle_deg, 0, 360, 255, frame_thickness)

            # Thick bridge
            bridge_thickness = max(4, int(eye_distance * 0.035))
            cv2.line(overlay, (left_inner[0], left_eye_cy), (right_inner[0], right_eye_cy),
                     frame_color, bridge_thickness)
            cv2.line(mask, (left_inner[0], left_eye_cy), (right_inner[0], right_eye_cy),
                     255, bridge_thickness)

            # Temple arms
            arm_len = int(eye_distance * 0.45)
            arm_thickness = max(3, int(eye_distance * 0.03))
            cv2.line(overlay,
                     (left_outer[0], left_eye_cy),
                     (left_outer[0] - arm_len, left_eye_cy - int(sg_lens_h * 0.1)),
                     frame_color, arm_thickness)
            cv2.line(mask,
                     (left_outer[0], left_eye_cy),
                     (left_outer[0] - arm_len, left_eye_cy - int(sg_lens_h * 0.1)),
                     255, arm_thickness)
            cv2.line(overlay,
                     (right_outer[0], right_eye_cy),
                     (right_outer[0] + arm_len, right_eye_cy - int(sg_lens_h * 0.1)),
                     frame_color, arm_thickness)
            cv2.line(mask,
                     (right_outer[0], right_eye_cy),
                     (right_outer[0] + arm_len, right_eye_cy - int(sg_lens_h * 0.1)),
                     255, arm_thickness)

        # Blend overlay onto the processed image using the mask
        mask_f = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_f, mask_f, mask_f], axis=2)
        processed = (processed.astype(np.float32) * (1.0 - mask_3ch)
                     + overlay.astype(np.float32) * mask_3ch).astype(np.uint8)

        metrics = _metrics_dict(original, processed)

        orig_spectrum = compute_magnitude_spectrum(compute_fft(original)[2])
        proc_spectrum = compute_magnitude_spectrum(compute_fft(processed)[2])

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        energy = compute_energy_analysis(processed, radius=30)

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
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
    """Extract outer lip landmark pixel positions."""
    # MediaPipe outer lip indices
    lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
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


# ── 1. ALIEN PRESET ──────────────────────────────────────────────────────────
def _apply_alien(image: np.ndarray) -> np.ndarray:
    """
    👽 Alien: Face slim (strong) + Eye enlarge (strong) + bright green overlay.
    """
    out = image.copy()
    # Warping: slim face
    out = apply_face_slim(out, 80)
    # Warping: enlarge eyes
    out = apply_eye_scaling(out, 70)

    # Get landmarks for face mask
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb)
        landmarks = get_landmarks(preprocessed)
        face_mask = _create_face_overlay_mask(out, landmarks)
    except Exception:
        # Fallback: use entire image with low opacity
        face_mask = np.ones(out.shape[:2], dtype=np.float32) * 0.5

    # Bright green overlay (BGR)
    out = _apply_color_overlay(out, face_mask, (0, 255, 0), opacity=0.30)
    return out


# ── 2. ROBOT PRESET ──────────────────────────────────────────────────────────
def _apply_robot(image: np.ndarray) -> np.ndarray:
    """
    🤖 Robot: Flatten mouth + metallic gray/silver overlay + yellow eyes.
    """
    out = image.copy()

    # Warping: straighten/flatten mouth by applying inverse lip widen + zero smile
    out = apply_lip_widen(out, -30)
    out = apply_smile(out, -20)

    # Get landmarks for face mask and eye coloring
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb)
        landmarks = get_landmarks(preprocessed)
        face_mask = _create_face_overlay_mask(out, landmarks)
    except Exception:
        face_mask = np.ones(out.shape[:2], dtype=np.float32) * 0.5
        landmarks = None

    # Metallic gray/silver overlay (BGR)
    out = _apply_color_overlay(out, face_mask, (192, 192, 192), opacity=0.30)

    # Color eye landmarks bright yellow
    if landmarks:
        out = _color_eye_landmarks(out, landmarks, (0, 255, 255), radius=5)

    return out


# ── 3. ANGRY PRESET ──────────────────────────────────────────────────────────
def _apply_angry(image: np.ndarray) -> np.ndarray:
    """
    😡 Angry: Eyebrows frown (downward) + bright red face overlay.
    """
    out = image.copy()

    # Warping: move eyebrows downward (negative raise = frown)
    out = apply_eyebrow_raise(out, -70)

    # Get landmarks for face mask
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb)
        landmarks = get_landmarks(preprocessed)
        face_mask = _create_face_overlay_mask(out, landmarks)
    except Exception:
        face_mask = np.ones(out.shape[:2], dtype=np.float32) * 0.5

    # Bright red overlay (BGR)
    out = _apply_color_overlay(out, face_mask, (0, 0, 255), opacity=0.30)
    return out


# ── 4. COLD PRESET ───────────────────────────────────────────────────────────
def _apply_cold(image: np.ndarray) -> np.ndarray:
    """
    🥶 Cold: Light blue face overlay + dark blue/purple lip color + eye squint.
    """
    out = image.copy()

    # Warping: squint eyes (inverse/negative scaling)
    out = apply_eye_scaling(out, -40)

    # Get landmarks for face mask and lip color
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb)
        landmarks = get_landmarks(preprocessed)
        face_mask = _create_face_overlay_mask(out, landmarks)
    except Exception:
        face_mask = np.ones(out.shape[:2], dtype=np.float32) * 0.5
        landmarks = None

    # Light blue face overlay (BGR)
    out = _apply_color_overlay(out, face_mask, (255, 200, 150), opacity=0.30)

    # Dark blue/purple lip color
    if landmarks:
        out = _apply_lip_color(out, landmarks, (180, 50, 100), opacity=0.50)

    return out


# ── 5. HEART-EYES PRESET ─────────────────────────────────────────────────────
def _apply_heart_eyes(image: np.ndarray) -> np.ndarray:
    """
    😍 Heart-Eyes: Eyebrow raise + lip widen + red lip makeup + heart masks.
    """
    out = image.copy()

    # Warping: raise eyebrows + widen lips
    out = apply_eyebrow_raise(out, 60)
    out = apply_lip_widen(out, 50)

    # Get landmarks for lip color
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb)
        landmarks = get_landmarks(preprocessed)
    except Exception:
        landmarks = None

    # Red lip makeup
    if landmarks:
        out = _apply_lip_color(out, landmarks, (0, 0, 255), opacity=0.45)

    # Place heart-shaped masks over the eyes
    out = _place_heart_masks(out, landmarks)
    return out


def _place_heart_masks(
    image: np.ndarray,
    landmarks: list | None,
) -> np.ndarray:
    """
    Place heart-shaped overlays over both eyes.

    ──────────────────────────────────────────────────────────────
    TODO: Implement complex heart-shape masking logic here.

    Steps to implement:
    1. Compute the centre of each eye from landmarks.
    2. Determine the heart size from the inter-eye distance.
    3. Draw or stamp a heart-shaped polygon/image on each eye.
    4. Alpha-blend the hearts onto the image.
    ──────────────────────────────────────────────────────────────
    """
    # Skeleton: return image safely without modification
    return image


# ── 6. CRYING PRESET ─────────────────────────────────────────────────────────
def _apply_crying(image: np.ndarray) -> np.ndarray:
    """
    😢 Crying: Eyebrows frown (downward) + mouth frown (curve down) + tears.
    """
    out = image.copy()

    # Warping: eyebrows downward (frown)
    out = apply_eyebrow_raise(out, -50)

    # Warping: mouth frown (negative smile = downward curve)
    out = apply_smile(out, -60)

    # Get landmarks for tear placement
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(rgb)
        landmarks = get_landmarks(preprocessed)
    except Exception:
        landmarks = None

    # Place tear-drop overlays below each eye
    out = _place_tear_masks(out, landmarks)
    return out


def _place_tear_masks(
    image: np.ndarray,
    landmarks: list | None,
) -> np.ndarray:
    """
    Place tear-drop overlays below each eye.

    ──────────────────────────────────────────────────────────────
    TODO: Implement complex tear-drop masking logic here.

    Steps to implement:
    1. Identify the lower-eye landmark positions (indices 145, 374).
    2. Compute a teardrop path (e.g. Bézier curve) below each eye.
    3. Draw semi-transparent blue/white teardrop shapes.
    4. Alpha-blend onto the image.
    ──────────────────────────────────────────────────────────────
    """
    # Skeleton: return image safely without modification
    return image


# ── Preset dispatcher ────────────────────────────────────────────────────────
_EMOJI_PRESETS_MAP = {
    "alien": _apply_alien,
    "robot": _apply_robot,
    "angry": _apply_angry,
    "cold": _apply_cold,
    "heart_eyes": _apply_heart_eyes,
    "crying": _apply_crying,
}


@router.post("/process/emoji-preset")
async def process_emoji_preset(body: EmojiPresetRequest):
    """
    Apply one of the 6 emoji-themed facial presets to the uploaded image.

    Accepts a JSON body with ``image_b64`` (data-URL or raw base64) and
    ``preset_name`` (alien | robot | angry | cold | heart_eyes | crying).
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
