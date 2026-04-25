import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

try:
    from modules.frequency_module import (
        apply_aging,
        apply_deaging,
        apply_fft_filter,
        compute_energy_analysis,
        compute_magnitude_spectrum,
        compute_fft,
        encode_image_to_base64,
    )
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
        compute_energy_analysis,
        compute_magnitude_spectrum,
        compute_fft,
        encode_image_to_base64,
    )
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
AGE_OPS = {"aging", "deaging", "fft"}


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
    spectrum_b64: str | None = None,
    energy: dict | None = None,
) -> dict:
    return {
        "image_b64": image_b64,
        "metrics": metrics,
        "spectrum_b64": spectrum_b64,
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

    logger.info(
        "process_warp.received",
        extra={"operation": op, "intensity": intensity, "smoothing": smoothing},
    )
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
            processed = cv2.addWeighted(processed, 1.0 - smooth_strength * 0.4, smoothed, smooth_strength * 0.4, 0)

        metrics = _metrics_dict(original, processed)
        logger.info(
            "process_warp.success",
            extra={"operation": op, "mse": metrics["mse"], "psnr": metrics["psnr"], "ssim": metrics["ssim"]},
        )
        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            spectrum_b64=None,
            energy=None,
        )
    except HTTPException:
        logger.exception("process_warp.http_error", extra={"operation": op})
        raise
    except Exception as exc:
        logger.exception("process_warp.failed", extra={"operation": op})
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

    logger.info(
        "process_age.received",
        extra={"operation": op, "intensity": intensity},
    )
    try:
        contents = await image.read()
        original = _decode_upload(contents)
        spectrum = compute_magnitude_spectrum(compute_fft(original)[2])

        if op == "aging":
            processed = apply_aging(original, intensity)
        elif op == "deaging":
            processed = apply_deaging(original, intensity)
        else:
            processed, spectrum = apply_fft_filter(original, intensity)

        energy = compute_energy_analysis(original, radius=int(10 + max(0.0, min(1.0, intensity / 100.0)) * 40))
        metrics = _metrics_dict(original, processed)
        logger.info(
            "process_age.success",
            extra={"operation": op, "mse": metrics["mse"], "psnr": metrics["psnr"], "ssim": metrics["ssim"]},
        )
        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            spectrum_b64=_data_url_from_image(cv2.cvtColor(spectrum, cv2.COLOR_GRAY2BGR)),
            energy=energy,
        )
    except HTTPException:
        logger.exception("process_age.http_error", extra={"operation": op})
        raise
    except Exception as exc:
        logger.exception("process_age.failed", extra={"operation": op})
        raise HTTPException(status_code=500, detail=str(exc)) from exc
