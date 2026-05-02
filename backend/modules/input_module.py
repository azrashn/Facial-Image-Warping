"""
input_module.py – Face detection, preprocessing, and landmark extraction pipeline.

This module provides four core functions that form a sequential image‐processing
pipeline for facial images:

    validate_image  →  detect_and_crop_face  →  preprocess_image  →  get_landmarks

All heavy CV work (face detection via OpenCV DNN, landmark extraction via
MediaPipe FaceMesh) is encapsulated here so that the FastAPI router remains thin.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
"""Accepted image file extensions (case‐insensitive)."""

FACE_TARGET_SIZE: tuple[int, int] = (512, 512)
"""Width × Height to which every cropped face is resized."""

FACEMESH_NUM_LANDMARKS: int = 468
"""Expected number of landmarks returned by MediaPipe FaceMesh."""

# ---------------------------------------------------------------------------
# OpenCV DNN face detector (Caffe model shipped with OpenCV)
# ---------------------------------------------------------------------------

_PROTO_PATH: str | None = None
_MODEL_PATH: str | None = None
_FACE_NET: cv2.dnn.Net | None = None


def _get_face_net() -> cv2.dnn.Net:
    """
    Lazily load the OpenCV DNN face‐detection model.

    Falls back to the Haar Cascade classifier shipped with OpenCV if the
    Caffe model files are unavailable.
    """
    global _FACE_NET, _PROTO_PATH, _MODEL_PATH  # noqa: PLW0603

    if _FACE_NET is not None:
        return _FACE_NET

    # Try to locate the Caffe model distributed with opencv‑contrib / extras
    opencv_data = Path(cv2.data.haarcascades).parent  # type: ignore[attr-defined]
    proto_candidates = list(opencv_data.rglob("deploy.prototxt"))
    model_candidates = list(opencv_data.rglob("res10_300x300_ssd_iter_140000.caffemodel"))

    if proto_candidates and model_candidates:
        _PROTO_PATH = str(proto_candidates[0])
        _MODEL_PATH = str(model_candidates[0])
        _FACE_NET = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _MODEL_PATH)
        logger.info("Loaded OpenCV DNN face detector (Caffe SSD).")
    else:
        # DNN model files not found – will fall back to Haar in detect_and_crop_face
        logger.warning(
            "OpenCV DNN Caffe model not found; "
            "will use Haar Cascade for face detection."
        )
        _FACE_NET = None  # explicitly keep as None

    return _FACE_NET  # type: ignore[return-value]


def _detect_face_dnn(
    image: np.ndarray, confidence_threshold: float = 0.5
) -> tuple[int, int, int, int] | None:
    """
    Detect the most‐confident face using OpenCV's DNN SSD detector.

    Returns ``(x, y, w, h)`` of the bounding box, or *None* if no face
    meets the confidence threshold.
    """
    net = _get_face_net()
    if net is None:
        return None

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_box: tuple[int, int, int, int] | None = None
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > confidence_threshold and conf > best_conf:
            best_conf = conf
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            best_box = (x1, y1, x2 - x1, y2 - y1)

    return best_box


def _detect_face_haar(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Fallback face detection using the Haar Cascade classifier bundled with
    OpenCV.

    Returns ``(x, y, w, h)`` or *None*.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None

    # Pick the largest detection (by area)
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_sorted[0]
    return (int(x), int(y), int(w), int(h))


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------


async def validate_image(file: UploadFile) -> bytes:
    """
    Validate that *file* is an accepted image format.

    Parameters
    ----------
    file:
        The ``UploadFile`` received from the FastAPI endpoint.

    Returns
    -------
    bytes
        The raw file contents if validation passes.

    Raises
    ------
    HTTPException (400)
        If the file extension is not in ``ALLOWED_EXTENSIONS`` or the
        uploaded file is empty.
    """
    filename: str = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    contents: bytes = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return contents


def detect_and_crop_face(image: np.ndarray) -> np.ndarray:
    """
    Detect the dominant face in *image* and return the cropped region.

    Uses the OpenCV DNN SSD detector by default; falls back to a Haar
    Cascade if the DNN model is unavailable.

    Parameters
    ----------
    image:
        BGR image as a NumPy array (as returned by ``cv2.imdecode``).

    Returns
    -------
    np.ndarray
        Cropped face region in BGR format.

    Raises
    ------
    HTTPException (400)
        If no face is detected by either method.
    """
    # Attempt DNN detection first
    bbox = _detect_face_dnn(image)

    # Fall back to Haar Cascade
    if bbox is None:
        bbox = _detect_face_haar(image)

    if bbox is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the uploaded image. Please upload a clear face photo.",
        )

    x, y, w, h = bbox

    # Add a small margin (15 %) so the crop is not too tight
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    img_h, img_w = image.shape[:2]

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Face bounding box resulted in an empty crop.",
        )

    logger.info("Face detected and cropped: bbox=(%d, %d, %d, %d)", x, y, w, h)
    return cropped


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Resize the cropped face to ``FACE_TARGET_SIZE`` and convert to RGB.

    Pixel values are kept in the ``[0, 255]`` uint8 range (no float
    normalisation) so that downstream modules (warping, frequency, etc.)
    can consume the image directly.  If float normalisation is needed at a
    later stage, callers can divide by 255.

    Parameters
    ----------
    image:
        Cropped face image in BGR format.

    Returns
    -------
    np.ndarray
        Preprocessed face image in **RGB** format with shape
        ``(512, 512, 3)`` and dtype ``uint8``.
    """
    resized = cv2.resize(image, FACE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    logger.info("Image preprocessed to %s (RGB, uint8).", FACE_TARGET_SIZE)
    return rgb


def _get_landmark_model_path() -> str:
    """Download / cache the FaceLandmarker .task model (same as warping_module)."""
    import os
    import tempfile
    import urllib.request

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    env_p = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    if env_p and os.path.isfile(env_p):
        return env_p
    cache_dir = os.path.join(tempfile.gettempdir(), "facial_image_warping_mp")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "face_landmarker.task")
    if not os.path.isfile(path) or os.path.getsize(path) < 1024 * 1024:
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


_TASK_LANDMARKER_INPUT: Any = None


def get_landmarks(image: np.ndarray) -> list[dict[str, float]]:
    """
    Extract 468 facial landmarks using MediaPipe FaceMesh.

    Uses the new MediaPipe Tasks API (``FaceLandmarker``) as primary,
    falling back to the legacy ``mp.solutions.face_mesh`` if available.

    Parameters
    ----------
    image:
        Face image in **RGB** format with shape ``(H, W, 3)``.

    Returns
    -------
    list[dict[str, float]]
        A list of 468 dictionaries, each containing ``"x"`` and ``"y"``
        keys with normalised coordinates in ``[0.0, 1.0]``.

    Raises
    ------
    HTTPException (400)
        If MediaPipe cannot detect a face mesh in the image.
    """
    global _TASK_LANDMARKER_INPUT  # noqa: PLW0603

    landmarks: list[dict[str, float]] | None = None

    # ── Strategy 1: legacy mp.solutions (if available) ──
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(image)

        if results and results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = [
                {"x": round(float(lm.x), 6), "y": round(float(lm.y), 6)}
                for lm in face.landmark[:FACEMESH_NUM_LANDMARKS]
            ]

    # ── Strategy 2: new Tasks API (FaceLandmarker) ──
    if landmarks is None:
        try:
            from mediapipe.tasks.python.vision import FaceLandmarker
            from mediapipe.tasks.python.vision.core import image as mp_image_module

            if _TASK_LANDMARKER_INPUT is None:
                _TASK_LANDMARKER_INPUT = FaceLandmarker.create_from_model_path(
                    _get_landmark_model_path()
                )

            mp_img = mp_image_module.Image(
                mp_image_module.ImageFormat.SRGB, image
            )
            result = _TASK_LANDMARKER_INPUT.detect(mp_img)

            if result.face_landmarks:
                lm_list = result.face_landmarks[0]
                landmarks = [
                    {"x": round(float(p.x), 6), "y": round(float(p.y), 6)}
                    for p in lm_list[:FACEMESH_NUM_LANDMARKS]
                ]
        except Exception as exc:
            logger.error("Tasks API landmark detection failed: %s", exc)

    if not landmarks:
        raise HTTPException(
            status_code=400,
            detail=(
                "MediaPipe FaceMesh could not detect landmarks in the "
                "preprocessed image. Please try a different photo."
            ),
        )

    # MediaPipe with refine_landmarks=True returns 478 points (468 base +
    # 10 iris).  We take exactly the first 468 base landmarks.
    if len(landmarks) != FACEMESH_NUM_LANDMARKS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {FACEMESH_NUM_LANDMARKS} landmarks but got "
                f"{len(landmarks)}. The face may be partially occluded."
            ),
        )

    logger.info("Extracted %d facial landmarks.", len(landmarks))
    return landmarks
