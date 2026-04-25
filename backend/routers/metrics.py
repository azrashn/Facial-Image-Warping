"""Router for quality metric comparison endpoints."""

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from modules.metrics_module import compute_mse, compute_psnr, compute_ssim
except ModuleNotFoundError:
    from backend.modules.metrics_module import compute_mse, compute_psnr, compute_ssim

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/compare")
async def compare_metrics(
    original: UploadFile = File(...),
    processed: UploadFile = File(...),
) -> dict:
    original_bytes = await original.read()
    processed_bytes = await processed.read()
    if not original_bytes or not processed_bytes:
        raise HTTPException(status_code=400, detail="Both images are required.")

    original_img = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR)
    processed_img = cv2.imdecode(np.frombuffer(processed_bytes, np.uint8), cv2.IMREAD_COLOR)
    if original_img is None or processed_img is None:
        raise HTTPException(status_code=400, detail="Invalid input image.")
    if original_img.shape != processed_img.shape:
        processed_img = cv2.resize(processed_img, (original_img.shape[1], original_img.shape[0]))

    return {
        "mse": float(compute_mse(original_img, processed_img)["mse"]),
        "psnr": float(compute_psnr(original_img, processed_img)["psnr"]),
        "ssim": float(compute_ssim(original_img, processed_img)["ssim"]),
    }
