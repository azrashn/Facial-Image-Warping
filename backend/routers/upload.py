"""
upload.py — Upload Router
==========================
DSP Project — Facial Image Warping (Group 14)
Role 1: CV & Input Developer

Endpoint
--------
POST /upload
    Accepts a multipart/form-data image upload, runs the full
    validate → detect/crop → preprocess → landmark-extraction pipeline,
    and returns:
        - processed_image : Base64-encoded PNG of the preprocessed 512×512 face
        - landmarks       : list of 468 {x, y, z} landmark dicts
"""

from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from modules.input_module import (
    detect_and_crop_face,
    get_landmarks,
    preprocess_image,
    validate_image,
)

router = APIRouter()


@router.post("/upload", summary="Upload a face image for processing")
async def upload_image(
    file: UploadFile = File(..., description="Face image (JPG, JPEG, PNG, or WEBP)"),
) -> dict[str, Any]:
    """Process an uploaded face image through the full CV pipeline.

    **Pipeline stages:**
    1. **Validate** — check file type / extension
    2. **Detect & crop** — locate the face, crop with padding
    3. **Preprocess** — resize to 512×512, convert to RGB
    4. **Extract landmarks** — 468-point MediaPipe FaceMesh

    Returns
    -------
    dict
        ``processed_image`` : ``str``
            Base64-encoded PNG (data-URI) of the preprocessed face.
        ``landmarks`` : ``list[dict]``
            468 dicts, each with ``x``, ``y``, ``z`` keys.
    """

    # 1. Validate ---------------------------------------------------------------
    image_bytes: bytes = await validate_image(file)

    # Decode bytes → OpenCV BGR ndarray
    np_arr: np.ndarray = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image. File may be corrupted.",
        )

    # 2. Detect & crop ----------------------------------------------------------
    cropped: np.ndarray = detect_and_crop_face(image_bgr)

    # 3. Preprocess (512×512, RGB) ----------------------------------------------
    preprocessed: np.ndarray = preprocess_image(cropped)

    # 4. Extract landmarks ------------------------------------------------------
    landmarks: list[dict[str, float]] = get_landmarks(preprocessed)

    # 5. Encode result image as Base64 PNG --------------------------------------
    # preprocessed is RGB → convert back to BGR for cv2.imencode
    bgr_out: np.ndarray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr_out)
    if not success:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail="Failed to encode the processed image.",
        )

    b64_string: str = (
        "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
    )

    return {
        "processed_image": b64_string,
        "landmarks": landmarks,
    }
