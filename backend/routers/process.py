import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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
        apply_face_slim,
        apply_lip_widen,
        apply_smile,
    )

router = APIRouter()
logger = logging.getLogger("facial_pipeline.process")

WARP_OPS = {"smile", "eyebrow", "lip", "slim"}
AGE_OPS = {"aging", "deaging", "age", "deage", "fft"}


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
    energy: dict | None = None,
) -> dict:
    return {
        "image_b64": image_b64,
        "metrics": metrics,
        "orig_spectrum_b64": orig_spectrum_b64,
        "proc_spectrum_b64": proc_spectrum_b64,
        "energy": energy,
    }


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

        orig_spectrum = compute_magnitude_spectrum(compute_fft(original)[2])
        proc_spectrum = compute_magnitude_spectrum(compute_fft(processed)[2])

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        energy = compute_energy_analysis(
            processed,
            radius=int(10 + max(0.0, min(1.0, intensity / 100.0)) * 40),
        )

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
            rgb_for_landmarks = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            landmarks = get_landmarks(rgb_for_landmarks)
            face_mask = create_face_region_mask(original, landmarks)

            if op in ["aging", "age"]:
                effected = apply_aging(original, intensity)
            else:
                effected = apply_deaging(original, intensity)

            processed = blend_effect_with_mask(
                original=original,
                effected=effected,
                mask=face_mask,
            )

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

        orig_spectrum = compute_magnitude_spectrum(
            compute_fft(original_for_metrics)[2]
        )

        if op == "fft":
            proc_spectrum = spectrum
        else:
            proc_spectrum = compute_magnitude_spectrum(
                compute_fft(processed)[2]
            )

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


@router.post("/process/makeup")
async def process_makeup(
    image: UploadFile = File(...),
    region: str = Form("lip"),
    hue: int = Form(0),
    opacity: float = Form(0.5),
):
    try:
        contents = await image.read()
        original = _decode_upload(contents)

        preprocessed = preprocess_image(original)
        landmarks = get_landmarks(preprocessed)
        preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)

        processed = apply_virtual_makeup(
            image=preprocessed_bgr,
            landmarks=landmarks,
            region=region,
            hue=hue,
            opacity=opacity,
        )

        metrics = _metrics_dict(preprocessed_bgr, processed)

        orig_spectrum = compute_magnitude_spectrum(compute_fft(preprocessed_bgr)[2])
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