"""
MediaPipe face landmarks + piecewise affine geometric warping.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import logging
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)

EMOJI_PRESETS = {
    "happy": {"smile": 0.7, "eyebrow_raise": 0.3, "lip_widen": 0.0, "eye_enlarge": 0.0},
    "surprised": {"smile": 0.0, "eyebrow_raise": 0.8, "lip_widen": 0.4, "eye_enlarge": 0.0},
    "joyful": {"smile": 1.0, "eyebrow_raise": 0.0, "lip_widen": 0.0, "eye_enlarge": 0.5},
    "neutral": {"smile": 0.0, "eyebrow_raise": 0.0, "lip_widen": 0.0, "eye_enlarge": 0.0},
}


def _clamp_intensity(intensity: int) -> float:
    return max(0.0, min(100.0, float(intensity))) / 100.0


def triangle_area(tri: np.ndarray) -> float:
    """Signed area of a 2-D triangle given (3,2) vertices."""
    a, b, c = tri
    return abs(
        a[0] * (b[1] - c[1]) +
        b[0] * (c[1] - a[1]) +
        c[0] * (a[1] - b[1])
    ) / 2.0


def _has_duplicate_vertices(tri: np.ndarray, eps: float = 1e-4) -> bool:
    """Return True if any two vertices in the triangle are identical."""
    a, b, c = tri
    if np.linalg.norm(a - b) < eps:
        return True
    if np.linalg.norm(b - c) < eps:
        return True
    if np.linalg.norm(a - c) < eps:
        return True
    return False


_TASK_LANDMARKER = None
_LM_STATE_LOCK = threading.Lock()
_LM_PREV_POINTS: Optional[np.ndarray] = None
_LM_PREV_TIME = 0.0
_LM_PREV_SHAPE: tuple[int, int] | None = None


def _estimate_head_pose(lm: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Approximate (yaw, pitch, roll) in degrees using solvePnP."""
    if lm.shape[0] < 468:
        return 0.0, 0.0, 0.0
    # Stable canonical 3D-ish template for core points.
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],        # nose tip (1)
            [0.0, -63.0, -12.0],    # chin (152)
            [-45.0, 32.0, -24.0],   # left eye corner (33)
            [45.0, 32.0, -24.0],    # right eye corner (263)
            [-34.0, -28.0, -20.0],  # left mouth corner (61)
            [34.0, -28.0, -20.0],   # right mouth corner (291)
        ],
        dtype=np.float32,
    )
    image_points = np.array(
        [lm[1], lm[152], lm[33], lm[263], lm[61], lm[291]], dtype=np.float32
    )
    focal = float(max(w, h))
    cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    ok, rvec, _tvec = cv2.solvePnP(
        model_points, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)
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


class PersistentFaceMesh:
    """
    Single persistent MediaPipe face landmark detector for video streaming.

    Supports two backends:
      1. **Tasks API** (mediapipe ≥0.10.x) — ``FaceLandmarker`` in ``VIDEO`` mode.
         This is the primary path for modern mediapipe where ``mp.solutions``
         has been removed.
      2. **Solutions API** (legacy mediapipe <0.10) — ``FaceMesh`` with
         ``static_image_mode=False``.

    Do **not** create a new detector per frame — that destroys temporal
    tracking and tanks FPS.
    """

    def __init__(self) -> None:
        self._backend: str = "none"
        self._mesh = None           # Legacy solutions FaceMesh
        self._landmarker = None     # Tasks API FaceLandmarker
        self._frame_ts_ms: int = 0  # Monotonic timestamp for VIDEO mode

        # ── Try Tasks API first (mediapipe >=0.10.x) ──
        try:
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )

            model_path = _face_landmarker_model_path()
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            self._backend = "tasks_video"
            logger.info("PersistentFaceMesh: using Tasks API (VIDEO mode)")
            return
        except Exception as exc:
            logger.debug("Tasks API VIDEO init failed: %s — trying solutions", exc)

        # ── Fallback: legacy solutions API ──
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._backend = "solutions"
            logger.info("PersistentFaceMesh: using legacy solutions FaceMesh")
            return

        raise RuntimeError(
            "No MediaPipe face landmark backend available. "
            "Install mediapipe >= 0.10.0 (Tasks API) or a legacy version with solutions."
        )

    def detect(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return (N, 2) float32 pixel landmarks or *None* if no face."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]

        if self._backend == "tasks_video":
            return self._detect_tasks(image_bgr, h, w)
        elif self._backend == "solutions":
            return self._detect_solutions(image_bgr, h, w)
        return None

    def _detect_tasks(self, image_bgr: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        """Detect using Tasks API FaceLandmarker in VIDEO mode."""
        from mediapipe.tasks.python.vision.core import image as mp_image_module

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp_image_module.Image(mp_image_module.ImageFormat.SRGB, rgb)

        # VIDEO mode requires monotonically increasing timestamps
        self._frame_ts_ms += 33  # ~30 FPS cadence
        try:
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        except Exception as exc:
            logger.debug("Tasks detect_for_video failed: %s", exc)
            return None

        if not result.face_landmarks:
            return None
        lm_list = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float32)
        if pts.shape[0] > 468:
            pts = pts[:468].copy()
        return pts

    def _detect_solutions(self, image_bgr: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        """Detect using legacy solutions FaceMesh."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def close(self) -> None:
        """Release underlying detector resources."""
        try:
            if self._mesh is not None:
                self._mesh.close()
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:
            pass


def detect_face_landmarks_live(mesh: PersistentFaceMesh, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Landmark detection using a persistent :class:`PersistentFaceMesh` instance."""
    return mesh.detect(image_bgr)


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


def _stable_ema_landmarks(raw_pts: Optional[np.ndarray], h: int, w: int) -> Optional[np.ndarray]:
    """Temporal smoothing + confidence gating for live tracking stability."""
    global _LM_PREV_POINTS, _LM_PREV_TIME, _LM_PREV_SHAPE
    if raw_pts is None or raw_pts.shape[0] < 100:
        with _LM_STATE_LOCK:
            if _LM_PREV_POINTS is not None and _LM_PREV_SHAPE == (h, w):
                return _LM_PREV_POINTS.copy()
        return None

    now = time.perf_counter()
    face_scale = float(np.linalg.norm(raw_pts[133] - raw_pts[362])) if raw_pts.shape[0] > 362 else 0.0
    min_face_scale = max(min(h, w) * 0.06, 12.0)
    if face_scale < min_face_scale:
        # Confidence gating: likely unstable / no true face lock.
        with _LM_STATE_LOCK:
            if _LM_PREV_POINTS is not None and _LM_PREV_SHAPE == (h, w):
                return _LM_PREV_POINTS.copy()
        return None

    alpha = float(np.clip(float(os.environ.get("LIVE_LANDMARK_EMA_ALPHA", 0.72)), 0.65, 0.80))
    with _LM_STATE_LOCK:
        stale_state = (
            _LM_PREV_POINTS is None
            or _LM_PREV_SHAPE != (h, w)
            or (now - _LM_PREV_TIME) > 1.0
        )
        if stale_state:
            _LM_PREV_POINTS = raw_pts.copy()
            _LM_PREV_TIME = now
            _LM_PREV_SHAPE = (h, w)
            return _LM_PREV_POINTS.copy()

        prev = _LM_PREV_POINTS
        mean_motion = float(np.mean(np.linalg.norm(raw_pts - prev, axis=1)))
        max_allowed_jump = max(face_scale * 0.35, 8.0)
        if mean_motion > max_allowed_jump:
            # Reject unstable frame and keep previous stable landmarks.
            _LM_PREV_TIME = now
            return prev.copy()

        smoothed = alpha * raw_pts + (1.0 - alpha) * prev
        _LM_PREV_POINTS = smoothed
        _LM_PREV_TIME = now
        _LM_PREV_SHAPE = (h, w)
        return smoothed.copy()


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
        raw_pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return _stable_ema_landmarks(raw_pts, h, w)

    return _stable_ema_landmarks(_landmarks_via_tasks(image_bgr, h, w), h, w)


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
    """Piecewise affine warp with robust triangle validation."""
    h, w = image_bgr.shape[:2]
    out = np.zeros_like(image_bgr)

    # --- Failsafe: wrap everything so the endpoint never crashes ---
    try:
        tri = Delaunay(dst_pts)
    except Exception as exc:
        logger.error("Delaunay triangulation failed: %s – returning original image", exc)
        return image_bgr.copy()

    total = len(tri.simplices)
    skipped_degenerate = 0
    skipped_duplicate = 0
    skipped_rect = 0
    warped_ok = 0

    for ia, ib, ic in tri.simplices:
        # ------ Req 1: force exact (3,2) float32 shape ------
        src_tri = np.asarray(
            [src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32
        ).reshape(3, 2)
        dst_tri = np.asarray(
            [dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32
        ).reshape(3, 2)

        # ------ Req 3: skip triangles with duplicate vertices ------
        if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
            skipped_duplicate += 1
            continue

        # ------ Req 2: skip degenerate (near-zero-area) triangles ------
        if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
            skipped_degenerate += 1
            continue

        # ------ Req 4: bounding rect safety ------
        r = cv2.boundingRect(dst_tri)
        bx, by, bw, bh = r
        # Clamp to image bounds
        bx = max(bx, 0)
        by = max(by, 0)
        bw = min(bw, w - bx)
        bh = min(bh, h - by)
        if bw <= 0 or bh <= 0:
            skipped_rect += 1
            continue

        try:
            mask = np.zeros((bh, bw), dtype=np.uint8)
            dst_crop = np.asarray(dst_tri - [bx, by], dtype=np.float32).reshape(3, 2)
            src_crop = np.asarray(src_tri, dtype=np.float32).reshape(3, 2)
            cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)

            # ------ Req 5: correct affine order src_crop → dst_crop ------
            warp_mat = cv2.getAffineTransform(src_crop, dst_crop)
            warped_patch = cv2.warpAffine(
                image_bgr,
                warp_mat,
                (bw, bh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            roi = out[by : by + bh, bx : bx + bw]
            blended = np.where(
                mask[..., None] == 255,
                warped_patch,
                roi,
            )
            out[by : by + bh, bx : bx + bw] = blended
            warped_ok += 1
        except Exception as tri_exc:
            logger.warning(
                "Triangle (%d,%d,%d) warp failed: %s", ia, ib, ic, tri_exc
            )
            continue

    # ------ Req 6: debug logging ------
    logger.debug(
        "geometric_warp stats – total: %d, warped: %d, "
        "skipped_degenerate: %d, skipped_duplicate: %d, skipped_rect: %d",
        total,
        warped_ok,
        skipped_degenerate,
        skipped_duplicate,
        skipped_rect,
    )

    # ------ Req 7: failsafe – if nothing was warped, return original ------
    if warped_ok == 0:
        logger.warning("No triangles warped successfully – returning original image")
        return image_bgr.copy()

    return out


def _prepare_warp(
    image_bgr: np.ndarray, src_lm: np.ndarray, deltas: np.ndarray
) -> np.ndarray:
    try:
        dst = src_lm + deltas
        height, width = image_bgr.shape[:2]
        corners = _corners(width, height)
        src_all = np.vstack([src_lm, corners])
        dst_all = np.vstack([dst, corners])
        return geometric_warp(image_bgr, src_all, dst_all)
    except Exception as exc:
        logger.error("_prepare_warp failed: %s – returning original image", exc)
        return image_bgr.copy()


def _gaussian_falloff(lm: np.ndarray, anchor_idx: int, sigma: float) -> np.ndarray:
    """
    Compute a (N,) weight array where each landmark's weight is
    exp(-dist² / 2σ²) relative to the anchor landmark.
    σ is expressed in pixels.
    """
    anchor = lm[anchor_idx]
    dists = np.linalg.norm(lm - anchor, axis=1)
    return np.exp(-0.5 * (dists / max(sigma, 1e-6)) ** 2)


def _face_scale(lm: np.ndarray) -> float:
    """Estimate face size (inter-eye distance) for resolution-independent σ."""
    # Landmarks 33 = nose tip, 133 = left eye outer, 362 = right eye outer
    left_eye = lm[133]
    right_eye = lm[362]
    return float(np.linalg.norm(left_eye - right_eye))


def apply_smile(
    image_bgr: np.ndarray,
    intensity: int,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Görev 3 Düzeltmesi: Natural Smile (Diagonal Displacement & Wide Falloff).
    Dudak köşelerini sadece dikey değil, elmacık kemiklerine doğru çapraz (yukarı ve dışa) çeker.
    Yanak kaslarının da harekete katılması için etki alanı (sigma) genişletilmiştir.
    Burun ucu, çene altı ve göz altları gibi kilit noktalar sabitlenerek yırtılma engellenir.
    """
    try:
        lm = landmarks if landmarks is not None else detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
            
        px = float(intensity)
        deltas = np.zeros_like(lm)
        face_sz = _face_scale(lm)
        yaw, _pitch, _roll = _estimate_head_pose(lm, image_bgr.shape[1], image_bgr.shape[0])
        
        # Etki Alanını Genişlet: Yanak kasları ve çevresinin harekete dahil olması için sigma büyütüldü
        sigma = face_sz * 0.25

        # 61: Sol dudak köşesi, 291: Sağ dudak köşesi
        w_left = _gaussian_falloff(lm, 61, sigma)
        w_right = _gaussian_falloff(lm, 291, sigma)
        
        center_x = (lm[61, 0] + lm[291, 0]) / 2.0
        half_width = abs(lm[291, 0] - lm[61, 0]) / 2.0 + 1e-6
        
        # Çapraz hareket katsayıları (Elmacık kemiklerine doğru: X'te dışarı, Y'de yukarı)
        move_x = px * 0.6  # Dışa doğru genişleme (X ekseni)
        move_y = px * 1.0  # Yukarı kaldırma (Y ekseni)
        
        for i in range(len(lm)):
            # Sol köşe (61) için: dx negatif (sola), dy negatif (yukarı)
            dx_left = w_left[i] * (-move_x)
            dy_left = w_left[i] * (-move_y)
            
            # Sağ köşe (291) için: dx pozitif (sağa), dy negatif (yukarı)
            dx_right = w_right[i] * (move_x)
            dy_right = w_right[i] * (-move_y)
            
            # Merkezdeki (dudak ortası) piksellerin yukarı katlanmasını (Joker efekti) önle
            dist_to_center_x = abs(lm[i, 0] - center_x)
            dy_damp = 1.0 - np.exp(-0.5 * (dist_to_center_x / (half_width * 0.6)) ** 2)
            
            # Displacement uygulaması (Hem X hem Y ekseni)
            # Head-turn compensation: reduce far-side horizontal pull for large yaw.
            # yaw > 0 => face turned right, left side is farther and attenuated more.
            far_left = 1.0 - 0.35 * np.clip(max(0.0, yaw) / 35.0, 0.0, 1.0)
            far_right = 1.0 - 0.35 * np.clip(max(0.0, -yaw) / 35.0, 0.0, 1.0)
            deltas[i, 0] += (dx_left * far_left + dx_right * far_right)
            deltas[i, 1] += (dy_left + dy_right) * dy_damp
            
        # Sabitlenecek Sınır Noktaları (Anchor Points) - Keskin bükülmeleri engeller
        anchor_points = [
            # Burun Ucu ve Çevresi
            1, 4, 5, 19, 94,
            # Göz Altları
            111, 117, 118, 119, 340, 346, 347, 348,
            # Çene Altı ve Dış Hatlar
            152, 148, 176, 149, 150, 377, 400, 378, 379, 365
        ]
        
        for idx in anchor_points:
            deltas[idx] = 0.0
            
        # Anchor noktalarını korumak için, çok düşük (ihmal edilebilir) hareketleri tam 0'a çekiyoruz
        deltas[np.abs(deltas) < 0.1] = 0.0
            
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_smile failed: %s", exc)
        return image_bgr.copy()


def apply_eyebrow_raise(
    image_bgr: np.ndarray,
    intensity: int,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rigid-body eyebrow lift: BOTH the upper and lower boundary
    landmarks of each brow translate by the exact same delta,
    preserving original thickness.  Gaussian falloff into the
    forehead prevents mesh tearing above the brow.
    """
    try:
        lm = landmarks if landmarks is not None else detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
        strength = _clamp_intensity(intensity)
        deltas = np.zeros_like(lm)

        face_sz = _face_scale(lm)
        lift = face_sz * 0.10 * strength  # max vertical lift in px

        # --- LEFT BROW: upper + lower boundary (rigid body) ---
        left_upper = [70, 63, 105, 66, 107]      # top edge
        left_lower = [46, 53, 52, 65, 55]         # bottom edge
        # --- RIGHT BROW: upper + lower boundary (rigid body) ---
        right_upper = [300, 293, 334, 296, 336]
        right_lower = [276, 283, 282, 295, 285]

        # Move ALL brow points by the same delta → preserves thickness
        all_brow = left_upper + left_lower + right_upper + right_lower
        for idx in all_brow:
            deltas[idx, 1] -= lift

        # Gaussian falloff into forehead above each brow centroid
        # so the skin above stretches smoothly instead of tearing
        sigma_fg = face_sz * 0.25
        left_center_idx = 66    # mid-brow left
        right_center_idx = 296  # mid-brow right

        w_l = _gaussian_falloff(lm, left_center_idx, sigma_fg)
        w_r = _gaussian_falloff(lm, right_center_idx, sigma_fg)

        forehead_falloff = lift * 0.4
        for i in range(len(lm)):
            # Only apply to points ABOVE the brow (lower y value)
            if lm[i, 1] < lm[left_center_idx, 1]:
                deltas[i, 1] -= w_l[i] * forehead_falloff
            if lm[i, 1] < lm[right_center_idx, 1]:
                deltas[i, 1] -= w_r[i] * forehead_falloff

        # Zero-out the brow indices from falloff so they keep exact rigid delta
        for idx in all_brow:
            deltas[idx, 1] = -lift

        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_eyebrow_raise failed: %s – returning original image", exc)
        return image_bgr.copy()


def apply_lip_widen(
    image_bgr: np.ndarray,
    intensity: int,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Görev 1 Düzeltmesi: Lip Widen (Distance-Based Weighted Displacement).
    Dudak köşesi etrafındaki noktaları Gauss ağırlığıyla yatayda kaydırarak
    yırtılmayı (pixel tearing) engeller.
    """
    try:
        lm = landmarks if landmarks is not None else detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
            
        px = float(intensity)
        deltas = np.zeros_like(lm)
        face_sz = _face_scale(lm)
        
        # Yırtılmayı engellemek için etki yarıçapı artırıldı
        sigma = face_sz * 0.15
        
        w_left = _gaussian_falloff(lm, 61, sigma)
        w_right = _gaussian_falloff(lm, 291, sigma)
        
        for i in range(len(lm)):
            # Lip Widen: dy=0, dx yatayda genişler
            # Sol köşe (61) dışa (-x)
            deltas[i, 0] += w_left[i] * (-px)
            
            # Sağ köşe (291) dışa (+x)
            deltas[i, 0] += w_right[i] * (px)
            
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_lip_widen failed: %s", exc)
        return image_bgr.copy()


def apply_face_slim(
    image_bgr: np.ndarray,
    intensity: int,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Face slim with smooth radial contraction toward the nose tip.
    Each jaw/cheek landmark is pulled along the vector toward the
    nose tip with a radial falloff — strongest at the outer jaw,
    decaying smoothly toward inner cheeks.
    """
    try:
        lm = landmarks if landmarks is not None else detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
        strength = _clamp_intensity(intensity)
        deltas = np.zeros_like(lm)
        yaw, _pitch, _roll = _estimate_head_pose(lm, image_bgr.shape[1], image_bgr.shape[0])

        face_sz = _face_scale(lm)
        nose_tip = lm[1].copy()  # landmark 1 = nose tip

        # Jaw contour indices (MediaPipe face mesh silhouette)
        jaw_contour = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]

        # Compute max distance from nose tip among jaw points
        jaw_positions = lm[jaw_contour]
        jaw_vecs = jaw_positions - nose_tip
        jaw_dists = np.linalg.norm(jaw_vecs, axis=1)
        max_jaw_dist = float(np.max(jaw_dists)) if np.max(jaw_dists) > 1e-3 else 1.0

        max_pull = face_sz * 0.10 * strength

        for i, idx in enumerate(jaw_contour):
            vec = lm[idx] - nose_tip
            dist = float(np.linalg.norm(vec))
            if dist < 1e-3:
                continue

            # Radial falloff: strongest at outer jaw, zero at nose tip
            #   normalized_dist in [0, 1] where 1 = outermost jaw point
            normalized_dist = dist / max_jaw_dist

            # Smooth cubic falloff for natural contour
            weight = normalized_dist ** 2

            # Pull direction: unit vector from point toward nose tip
            direction = -vec / dist  # toward nose
            # Only take the horizontal component to avoid vertical squish
            pull_x = direction[0] * weight * max_pull
            pull_y = direction[1] * weight * max_pull * 0.3  # dampen vertical

            # Pose-aware attenuation: preserve perspective on far side.
            is_left = lm[idx, 0] < nose_tip[0]
            if yaw > 0 and is_left:
                pull_x *= (1.0 - 0.35 * np.clip(yaw / 35.0, 0.0, 1.0))
            if yaw < 0 and not is_left:
                pull_x *= (1.0 - 0.35 * np.clip((-yaw) / 35.0, 0.0, 1.0))
            deltas[idx, 0] += pull_x
            deltas[idx, 1] += pull_y

        # Gaussian falloff to neighboring non-jaw landmarks for mesh smoothness
        sigma_spread = face_sz * 0.15
        jaw_set = set(jaw_contour)
        for anchor_idx in jaw_contour:
            if abs(deltas[anchor_idx, 0]) < 1e-6 and abs(deltas[anchor_idx, 1]) < 1e-6:
                continue
            w = _gaussian_falloff(lm, anchor_idx, sigma_spread)
            for i in range(len(lm)):
                if i in jaw_set:
                    continue  # don't double-apply to jaw points
                deltas[i, 0] += w[i] * deltas[anchor_idx, 0] * 0.3
                deltas[i, 1] += w[i] * deltas[anchor_idx, 1] * 0.3

        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_face_slim failed: %s – returning original image", exc)
        return image_bgr.copy()


def apply_eye_scaling(
    image_bgr: np.ndarray,
    intensity: int,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Görev 2 Düzeltmesi: Radial Eye Scaling (Smooth Falloff & Anchors).
    Gözleri kendi merkezlerinden orantılı şekilde büyütür/küçültür.
    Geniş bir Gauss ağırlığı (falloff) kullanarak çevresel piksellerin yırtılmasını önler.
    Kaş, burun köprüsü ve yanak üstü gibi anchor noktalar KESİNLİKLE sabitlenir.
    """
    try:
        lm = landmarks if landmarks is not None else detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr

        factor = max(-1.0, min(1.0, float(intensity) / 100.0))
        deltas = np.zeros_like(lm)
        face_sz = _face_scale(lm)

        left_eye_ring = [33, 133, 160, 158, 153, 144, 159, 145]
        right_eye_ring = [362, 263, 387, 385, 380, 373, 386, 374]

        center_left = np.mean(lm[left_eye_ring], axis=0)
        center_right = np.mean(lm[right_eye_ring], axis=0)

        # Yumuşak geçiş için Gauss sigma (etki alanı)
        sigma = face_sz * 0.12

        def custom_falloff(points, center, sig):
            dists = np.linalg.norm(points - center, axis=1)
            return np.exp(-0.5 * (dists / max(sig, 1e-6)) ** 2)

        w_left = custom_falloff(lm, center_left, sigma)
        w_right = custom_falloff(lm, center_right, sigma)

        for i in range(len(lm)):
            disp_left = (lm[i] - center_left) * factor * w_left[i]
            disp_right = (lm[i] - center_right) * factor * w_right[i]
            deltas[i] += disp_left + disp_right

        # Sınırlandırma (Anchor Points) - kesinlikle hareket etmeyecek bölgeler
        anchor_points = [
            # Kaşlar (Eyebrows)
            70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
            300, 293, 334, 296, 336, 276, 283, 282, 295, 285,
            # Burun köprüsü (Nose bridge)
            168, 6, 197, 195, 5, 4,
            # Elmacık kemiği üstü (Upper cheekbones)
            116, 117, 118, 119, 100, 101, 345, 346, 347, 348, 329, 330,
            # Dış hatlar / Çene (Jaw / Outer contour)
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        for idx in anchor_points:
            deltas[idx] = 0.0

        # Anchor noktalarını korumak için, çok düşük (ihmal edilebilir) hareketleri tam 0'a çekiyoruz
        deltas[np.abs(deltas) < 0.1] = 0.0

        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_eye_scaling failed: %s – returning original image", exc)
        return image_bgr.copy()


def apply_emoji_preset(
    image_bgr: np.ndarray,
    emoji_name: str,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply predefined expression presets by chaining existing warps.
    """
    try:
        preset_key = (emoji_name or "neutral").strip().lower()
        preset = EMOJI_PRESETS.get(preset_key, EMOJI_PRESETS["neutral"])
        out = image_bgr.copy()

        if preset.get("smile", 0.0) > 0:
            out = apply_smile(
                out, int(round(preset["smile"] * 100)), landmarks=landmarks
            )
        if preset.get("eyebrow_raise", 0.0) > 0:
            out = apply_eyebrow_raise(
                out, int(round(preset["eyebrow_raise"] * 100)), landmarks=landmarks
            )
        if preset.get("lip_widen", 0.0) > 0:
            out = apply_lip_widen(
                out, int(round(preset["lip_widen"] * 100)), landmarks=landmarks
            )
        if preset.get("eye_enlarge", 0.0) != 0:
            out = apply_eye_scaling(
                out, int(round(preset["eye_enlarge"] * 100)), landmarks=landmarks
            )

        return out
    except Exception as exc:
        logger.error("apply_emoji_preset failed: %s – returning original image", exc)
        return image_bgr.copy()


def apply_beard(
    image_bgr: np.ndarray,
    intensity: int,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Görev 5 Düzeltmesi: Ultimate Facial Hair (Multiply Blend & Normalized Mask).
    Sakal/Bıyık dokusunu fiziksel olarak koyulaştırmak için Multiply Blend kullanır.
    Maskeyi cv2.normalize ile [0,1] aralığına tam yayarak şeffaflığı tamamen ortadan kaldırır.
    Alttaki deri rengini "yutmak" yerine direkt olarak cilt piksellerini karartır.
    """
    try:
        lm = landmarks if landmarks is not None else detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr

        alpha = _clamp_intensity(intensity)
        if alpha <= 0:
            return image_bgr.copy()

        h, w = image_bgr.shape[:2]
        beard_poly_idx = [17, 18, 200, 199, 175, 15, 12, 0]
        beard_poly = np.array([lm[i] for i in beard_poly_idx], dtype=np.int32).reshape((-1, 1, 2))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [beard_poly], 255)

        # Gürültü matrisi (Noise pattern) oluştur
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randu(noise, 0, 255)
        
        # Daha gür ve kalın kıl kökleri için noise'u hafif büyütüp tekrar orijinal boyuta al
        noise_small = cv2.resize(noise, (w // 2, h // 2))
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 1. Maskeyi Keskinleştir (Thresholding & Normalization)
        # Sıkı bir threshold ile kıl köklerini seç
        _, hair = cv2.threshold(noise, 170, 255, cv2.THRESH_BINARY)
        hair = cv2.bitwise_and(hair, mask)
        
        # Kıl sınırlarını hafif yumuşat (aliasing olmasın diye)
        hair = cv2.GaussianBlur(hair, (0, 0), 0.8)
        
        # Maskeyi [0, 1] aralığına tam olarak yay (cv2.normalize)
        hair_float = np.zeros((h, w), dtype=np.float32)
        cv2.normalize(hair.astype(np.float32), hair_float, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # Alpha (Intensity) değerini direkt olarak harmanlama gücü olarak kullan
        # %100 seçildiğinde maske içindeki pikseli maksimum oranda etkileyecek
        blend_power = np.clip(alpha * 1.5, 0.0, 1.0)
        
        # Etkili Maske (Effective Mask)
        effective_mask = hair_float * blend_power
        effective_mask = effective_mask[..., None] # 3 kanala uygulamak için genişlet
        
        image_float = image_bgr.astype(np.float32)
        
        # 1 & 2. Çok Koyu Sabit Renk ve Gerçek Multiply Blend Modu
        # Sakal eklenecek bölgeyi gri bir sis yapmak yerine, maskenin gücüne göre
        # cildi RGB(20, 20, 20) rengi ile matematiksel olarak çarpıyoruz.
        beard_color = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        
        # Multiply mantığı: (Cilt * Sakal Rengi) / 255. 
        # Sakal rengi çok düşük olduğu için cilt pikselleri kapkara ama dokulu kalır.
        multiply_blend = image_float * (beard_color / 255.0)
        
        # Maskenin olduğu (kıl kökleri) yerde karanlık multiply_blend, olmadığı yerde orijinal cilt
        blended = image_float * (1.0 - effective_mask) + multiply_blend * effective_mask
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception as exc:
        logger.error("apply_beard failed: %s – returning original image", exc)
        return image_bgr.copy()

