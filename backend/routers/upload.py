"""
upload.py – FastAPI router for image upload and face‐processing pipeline.

Endpoints
---------
POST /upload
    Accepts a multipart/form-data image file, runs the full pipeline
    (validate → detect/crop → preprocess → landmarks), and returns the
    processed 512×512 face as a Base64 PNG string together with the 468
    MediaPipe FaceMesh landmarks.
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from modules.frequency_module import encode_image_to_base64
    from modules.input_module import (
        detect_and_crop_face,
        get_landmarks,
        preprocess_image,
        validate_image,
    )
except ModuleNotFoundError:
    from backend.modules.frequency_module import encode_image_to_base64
    from backend.modules.input_module import (
        detect_and_crop_face,
        get_landmarks,
        preprocess_image,
        validate_image,
    )

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_image(image: UploadFile = File(...)) -> dict:
    """
    Upload an image and run the full face‐processing pipeline.

    Pipeline steps
    ^^^^^^^^^^^^^^
    1. **Validate** – ensure file extension is JPG / JPEG / PNG / WEBP.
    2. **Decode** – convert raw bytes to an OpenCV BGR image.
    3. **Detect & Crop** – locate the dominant face and crop it.
    4. **Preprocess** – resize to 512×512, convert to RGB.
    5. **Landmarks** – extract 468 MediaPipe FaceMesh landmarks.

    Returns
    -------
    dict
        ``processed_image_b64``
            ``data:image/png;base64,…`` encoded processed face.
        ``landmarks``
            List of 468 dicts ``{"x": float, "y": float}``.
        ``width``
            Width of the processed image (512).
        ``height``
            Height of the processed image (512).
        ``filename``
            Original upload filename.
    """
    # 1. Validate file extension & read bytes
    contents: bytes = await validate_image(image)

    # 2. Decode bytes → OpenCV BGR image
    nparr = np.frombuffer(contents, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded.")

    # 3. Detect face and crop
    cropped = detect_and_crop_face(decoded)

    # 4. Preprocess (resize 512×512, BGR → RGB)
    preprocessed = preprocess_image(cropped)

    # 5. Extract 468 facial landmarks (expects RGB input)
    landmarks = get_landmarks(preprocessed)

    # Encode the processed image (convert RGB back to BGR for cv2.imencode)
    processed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)
    image_b64 = encode_image_to_base64(processed_bgr)

    height, width = preprocessed.shape[:2]

    logger.info(
        "Upload pipeline complete for '%s': %dx%d, %d landmarks.",
        image.filename,
        width,
        height,
        len(landmarks),
    )

    return {
        "processed_image_b64": f"data:image/png;base64,{image_b64}",
        "landmarks": landmarks,
        "width": width,
        "height": height,
        "filename": image.filename,
    }
