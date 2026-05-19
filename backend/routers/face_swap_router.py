"""
face_swap_router.py — FastAPI router for the static face swap endpoint.

Endpoint
--------
POST /api/face-swap
    Accepts two multipart images (source face, target face) and returns
    the swapped result as a Base64 PNG data-URL.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from modules.face_swap_module import (
        FaceSwapError,
        apply_face_swap,
        load_source_face,
    )
    from modules.frequency_module import encode_image_to_base64
except ModuleNotFoundError:
    from backend.modules.face_swap_module import (
        FaceSwapError,
        apply_face_swap,
        load_source_face,
    )
    from backend.modules.frequency_module import encode_image_to_base64

logger = logging.getLogger(__name__)

router = APIRouter(tags=["face-swap"])


def _decode_upload_bytes(raw: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR OpenCV image."""
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return img


@router.post("/api/face-swap")
async def face_swap_endpoint(
    source: UploadFile = File(..., description="Source face image"),
    target: UploadFile = File(..., description="Target face image"),
):
    """
    Swap the face from the **source** image onto the **target** image.

    Both images must contain exactly one clearly visible face.

    Parameters (multipart/form-data)
    --------------------------------
    source : UploadFile
        The image whose face will be extracted.
    target : UploadFile
        The image that will receive the swapped face.

    Returns
    -------
    dict
        ``swapped_image``
            ``data:image/png;base64,…`` encoded result.
        ``processing_time_ms``
            Wall-clock time for the swap (milliseconds).
    """
    t_start = time.perf_counter()

    # ── Validate & decode uploads ─────────────────────────────────────────
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Source file is not an image (got {source.content_type}).",
        )
    if target.content_type and not target.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Target file is not an image (got {target.content_type}).",
        )

    try:
        source_bytes = await source.read()
        target_bytes = await target.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded files: {exc}",
        ) from exc

    if not source_bytes:
        raise HTTPException(status_code=400, detail="Source image is empty.")
    if not target_bytes:
        raise HTTPException(status_code=400, detail="Target image is empty.")

    source_bgr = _decode_upload_bytes(source_bytes)
    target_bgr = _decode_upload_bytes(target_bytes)

    # ── Run face swap pipeline ────────────────────────────────────────────
    try:
        result = apply_face_swap(source_bgr, target_bgr)
    except FaceSwapError as exc:
        logger.warning("Face swap failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected face swap error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Face swap processing failed: {exc}",
        ) from exc

    # ── Encode result ─────────────────────────────────────────────────────
    result_b64 = encode_image_to_base64(result)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    logger.info(
        "Face swap complete: source=%s target=%s  %.1f ms",
        source.filename,
        target.filename,
        elapsed_ms,
    )

    return {
        "swapped_image": f"data:image/png;base64,{result_b64}",
        "processing_time_ms": elapsed_ms,
    }
