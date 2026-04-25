"""
MediaPipe face landmarks + piecewise affine geometric warping.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay


def _clamp_intensity(intensity: int) -> float:
    return max(0.0, min(100.0, float(intensity))) / 100.0


_TASK_LANDMARKER = None
_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _face_landmarker_model_path() -> str:
    env_p = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    if env_p and os.path.isfile(env_p):
        return env_p
    cache_dir = os.path.join(tempfile.gettempdir(), "facial_image_warping_mp")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "face_landmarker.task")
    if not os.path.isfile(path) or os.path.getsize(path) < 1024 * 1024:
        urllib.request.urlretrieve(_TASK_MODEL_URL, path)
    return path


def _get_tasks_face_landmarker():
    global _TASK_LANDMARKER
    if _TASK_LANDMARKER is None:
        from mediapipe.tasks.python.vision import FaceLandmarker

        _TASK_LANDMARKER = FaceLandmarker.create_from_model_path(
            _face_landmarker_model_path()
        )
    return _TASK_LANDMARKER


def _landmarks_via_tasks(image_bgr: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
    from mediapipe.tasks.python.vision.core import image as mp_image_module

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_image_module.Image(mp_image_module.ImageFormat.SRGB, rgb)
    result = _get_tasks_face_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    lm_list = result.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float32)
    if pts.shape[0] > 468:
        pts = pts[:468].copy()
    return pts


def detect_face_landmarks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    return _landmarks_via_tasks(image_bgr, h, w)


def _corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ],
        dtype=np.float32,
    )


def geometric_warp(
    image_bgr: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = np.zeros_like(image_bgr)
    tri = Delaunay(dst_pts)
    for ia, ib, ic in tri.simplices:
        dst_tri = np.float32([dst_pts[ia], dst_pts[ib], dst_pts[ic]])
        src_tri = np.float32([src_pts[ia], src_pts[ib], src_pts[ic]])
        r = cv2.boundingRect(dst_tri)
        x, y, rw, rh = r
        x = max(x, 0)
        y = max(y, 0)
        rw = min(rw, w - x)
        rh = min(rh, h - y)
        if rw <= 1 or rh <= 1:
            continue
        mask = np.zeros((rh, rw), dtype=np.uint8)
        dst_crop = dst_tri - [x, y]
        cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)
        affine_mat = cv2.getAffineTransform(dst_crop[:3], src_tri[:3])
        warped = cv2.warpAffine(
            image_bgr,
            affine_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        roi = out[y : y + rh, x : x + rw]
        blended = np.where(mask[..., None] == 255, warped[y : y + rh, x : x + rw], roi)
        out[y : y + rh, x : x + rw] = blended
    return out


def _prepare_warp(
    image_bgr: np.ndarray, src_lm: np.ndarray, deltas: np.ndarray
) -> np.ndarray:
    dst = src_lm + deltas
    height, width = image_bgr.shape[:2]
    corners = _corners(width, height)
    src_all = np.vstack([src_lm, corners])
    dst_all = np.vstack([dst, corners])
    return geometric_warp(image_bgr, src_all, dst_all)


def apply_smile(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    lm = detect_face_landmarks(image_bgr)
    if lm is None:
        return image_bgr
    strength = _clamp_intensity(intensity)
    deltas = np.zeros_like(lm)
    up = 5.0 * strength
    out_x = 3.0 * strength
    for idx, sx in ((61, -1), (291, 1)):
        deltas[idx, 1] -= up
        deltas[idx, 0] += sx * out_x
    for idx in (39, 269):
        deltas[idx, 1] -= 2.0 * strength
    return _prepare_warp(image_bgr, lm, deltas)


def apply_eyebrow_raise(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    lm = detect_face_landmarks(image_bgr)
    if lm is None:
        return image_bgr
    strength = _clamp_intensity(intensity)
    deltas = np.zeros_like(lm)
    up = 6.0 * strength
    left = [70, 63, 105, 66, 107]
    right = [300, 293, 334, 296, 336]
    for idx in left + right:
        deltas[idx, 1] -= up
    return _prepare_warp(image_bgr, lm, deltas)


def apply_lip_widen(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    lm = detect_face_landmarks(image_bgr)
    if lm is None:
        return image_bgr
    strength = _clamp_intensity(intensity)
    deltas = np.zeros_like(lm)
    wx = 4.0 * strength
    pairs = [(61, -1), (291, 1), (78, -1), (308, 1), (95, -1), (324, 1)]
    for idx, sign in pairs:
        deltas[idx, 0] += sign * wx
    return _prepare_warp(image_bgr, lm, deltas)


def apply_face_slim(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    lm = detect_face_landmarks(image_bgr)
    if lm is None:
        return image_bgr
    strength = _clamp_intensity(intensity)
    deltas = np.zeros_like(lm)
    center_x = float(np.mean(lm[:, 0]))
    pull = 5.0 * strength
    jaw = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
    for idx in jaw:
        dx = center_x - lm[idx, 0]
        if abs(dx) < 1e-3:
            continue
        deltas[idx, 0] += np.sign(dx) * pull * 0.15
    return _prepare_warp(image_bgr, lm, deltas)
