"""
input_module.py — CV & Input Processing Pipeline
==================================================
DSP Project — Facial Image Warping (Group 14)
Role 1: CV & Input Developer

Provides:
    - validate_image      : file-type gate (JPG / JPEG / PNG / WEBP)
    - detect_and_crop_face: DNN-based face detection + bounding-box crop
    - preprocess_image    : resize to 512×512, RGB conversion, normalisation
    - get_landmarks       : MediaPipe FaceMesh → 468 landmark coordinates
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import HTTPException, UploadFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "webp"}
_ALLOWED_CONTENT_TYPES: set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
_TARGET_SIZE: tuple[int, int] = (512, 512)

# ---------------------------------------------------------------------------
# OpenCV DNN face detector (Caffe model shipped with OpenCV)
# Falls back to Haar Cascade if no DNN model files are found.
# ---------------------------------------------------------------------------

_HAAR_CASCADE_PATH: str = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]


def _get_face_detector() -> cv2.CascadeClassifier:
    """Return a Haar-cascade face detector (always available with OpenCV)."""
    cascade = cv2.CascadeClassifier(_HAAR_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            "Haar cascade XML could not be loaded. "
            f"Checked path: {_HAAR_CASCADE_PATH}"
        )
    return cascade


# Singleton — loaded once at module import time
_face_cascade: cv2.CascadeClassifier = _get_face_detector()


# ---------------------------------------------------------------------------
# 1. validate_image
# ---------------------------------------------------------------------------

async def validate_image(file: UploadFile) -> bytes:
    """Validate that *file* is an allowed image type and return its bytes.

    Allowed formats: **JPG, JPEG, PNG, WEBP**.

    Parameters
    ----------
    file : fastapi.UploadFile
        The uploaded file from the request.

    Returns
    -------
    bytes
        Raw image bytes.

    Raises
    ------
    HTTPException (400)
        If the file extension or MIME type is not in the allow-list.
    """
    filename: str = file.filename or ""
    extension: str = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '.{extension}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}."
            ),
        )

    content_type: str | None = file.content_type
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported MIME type '{content_type}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}."
            ),
        )

    image_bytes: bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return image_bytes


# ---------------------------------------------------------------------------
# 2. detect_and_crop_face
# ---------------------------------------------------------------------------

def detect_and_crop_face(image: np.ndarray) -> np.ndarray:
    """Detect the most prominent face and return the cropped region.

    Uses an OpenCV **Haar Cascade** classifier.

    Parameters
    ----------
    image : np.ndarray
        BGR image (as returned by ``cv2.imdecode``).

    Returns
    -------
    np.ndarray
        Cropped face region (BGR).

    Raises
    ------
    HTTPException (400)
        If no face is detected.
    """
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces: np.ndarray = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the uploaded image.",
        )

    # Pick the largest face by area
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_sorted[0]

    # Add padding around the detected region (20 %) for better coverage
    pad_x: int = int(w * 0.20)
    pad_y: int = int(h * 0.20)
    img_h, img_w = image.shape[:2]

    x1: int = max(x - pad_x, 0)
    y1: int = max(y - pad_y, 0)
    x2: int = min(x + w + pad_x, img_w)
    y2: int = min(y + h + pad_y, img_h)

    cropped: np.ndarray = image[y1:y2, x1:x2]
    return cropped


# ---------------------------------------------------------------------------
# 3. preprocess_image
# ---------------------------------------------------------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize the face crop to 512×512 and convert to RGB.

    Parameters
    ----------
    image : np.ndarray
        Cropped face image (BGR).

    Returns
    -------
    np.ndarray
        Preprocessed image in **RGB** colour-space, shape ``(512, 512, 3)``,
        pixel values in ``uint8`` range ``[0, 255]``.
    """
    resized: np.ndarray = cv2.resize(
        image,
        _TARGET_SIZE,
        interpolation=cv2.INTER_LANCZOS4,
    )

    # BGR → RGB
    rgb: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


# ---------------------------------------------------------------------------
# 4. get_landmarks
# ---------------------------------------------------------------------------

def get_landmarks(image: np.ndarray) -> list[dict[str, float]]:
    """Extract 468 facial landmarks using **MediaPipe FaceMesh**.

    Parameters
    ----------
    image : np.ndarray
        RGB image, expected shape ``(512, 512, 3)``, ``uint8``.

    Returns
    -------
    list[dict[str, float]]
        A JSON-serialisable list of 468 dicts, each containing
        ``{"x": float, "y": float, "z": float}``.  The *x* and *y*
        values are in **pixel** coordinates; *z* is the raw depth estimate
        from MediaPipe.

    Raises
    ------
    HTTPException (400)
        If MediaPipe cannot detect a face mesh in the image.
    """
    h, w = image.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image)

    if not results or not results.multi_face_landmarks:
        raise HTTPException(
            status_code=400,
            detail="MediaPipe FaceMesh could not detect a face in the image.",
        )

    face = results.multi_face_landmarks[0]

    landmarks: list[dict[str, float]] = [
        {
            "x": round(float(lm.x * w), 4),
            "y": round(float(lm.y * h), 4),
            "z": round(float(lm.z), 6),
        }
        for lm in face.landmark
    ]

    # FaceMesh with refine_landmarks=True returns 478 points (incl. iris).
    # Trim to the canonical 468 if we got more.
    if len(landmarks) > 468:
        landmarks = landmarks[:468]

    return landmarks
