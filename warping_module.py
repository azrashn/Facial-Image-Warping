"""
Rol 2: MediaPipe yüz işaretçileri + SciPy Delaunay parça-affine geometrik çarpıtma.
Bağımlılıklar: mediapipe, opencv-python, numpy, scipy (requirements.txt ile uyumlu).
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
    # 478 noktalı modelde ilk 468, klasik Face Mesh indeksleriyle uyumludur.
    if pts.shape[0] > 468:
        pts = pts[:468].copy()
    return pts


def detect_face_landmarks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    FR-08: MediaPipe Face Mesh (legacy) veya Tasks Face Landmarker ile
    yüz landmark'ları (piksel, Nx2 float32). Yüz yoksa None.
    """
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
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]],
        dtype=np.float32,
    )


def geometric_warp(
    image_bgr: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    FR-18/FR-19: Hedef noktalara göre Delaunay üçgenleri + parça-affine ters çarpıtma.
    src_pts, dst_pts: aynı uzunlukta (Mx2); köşe noktaları çağıran taraf eklemeli.
    """
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
        M = cv2.getAffineTransform(dst_crop[:3], src_tri[:3])
        warped = cv2.warpAffine(
            image_bgr,
            M,
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
    """deltas: Nx2, sadece ilgili indeksler dolu; diğerleri 0."""
    dst = src_lm + deltas
    corners = _corners(w, h)
    src_all = np.vstack([src_lm, corners])
    dst_all = np.vstack([dst, corners])
    return geometric_warp(image_bgr, src_all, dst_all)


def apply_smile(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """
    Görev 1 Düzeltmesi: Add Smile (Smooth RBF Falloff).
    Dudak köşelerini sadece dikey (dy) yönde yukarı kaldırır (dx=0).
    Mesh yırtılmasını (tearing/dimples) engellemek için geniş bir Gaussian (RBF) falloff kullanılır.
    Çene, burun ve gözler gibi uzak noktalar (anchor points) tamamen sabit kalır.
    """
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
            
        px = float(intensity)
        deltas = np.zeros_like(lm)
        face_sz = _face_scale(lm)
        
        # Smooth Stretching (RBF Falloff): Yırtılmayı önlemek için yumuşak ve geniş bir geçiş
        sigma = face_sz * 0.15
        
        w_left = _gaussian_falloff(lm, 61, sigma)
        w_right = _gaussian_falloff(lm, 291, sigma)
        
        center_x = (lm[61, 0] + lm[291, 0]) / 2.0
        half_width = abs(lm[291, 0] - lm[61, 0]) / 2.0 + 1e-6
        
        for i in range(len(lm)):
            # Sadece Yukarı (Minimal Genişleme): dx = 0, sadece dy = -px
            dy_left = w_left[i] * (-px)
            dy_right = w_right[i] * (-px)
            
            # Kademeli Kavis ve Merkez Koruması: Dudak merkezinin yukarı katlanmasını önle
            dist_to_center_x = abs(lm[i, 0] - center_x)
            dy_damp = 1.0 - np.exp(-0.5 * (dist_to_center_x / (half_width * 0.4)) ** 2)
            
            # Displacement uygulaması
            deltas[i, 1] += (dy_left + dy_right) * dy_damp
            
        # Anchor noktaları korumak için, çok düşük (ihmal edilebilir) hareketleri tam 0'a çekiyoruz
        # Böylece çene, burun ve gözler (uzak noktalar) KESİNLİKLE sabit kalır.
        deltas[np.abs(deltas) < 0.1] = 0.0
            
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_smile failed: %s", exc)
        return image_bgr.copy()


def apply_eyebrow_raise(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """FR-14: kaşları yukarı."""
    lm = detect_face_landmarks(image_bgr)
    if lm is None:
        return image_bgr
    a = _clamp_intensity(intensity)
    d = np.zeros_like(lm)
    up = 6.0 * a
    left = [70, 63, 105, 66, 107]
    right = [300, 293, 334, 296, 336]
    for i in left + right:
        d[i, 1] -= up
    return _prepare_warp(image_bgr, lm, d)


def apply_lip_widen(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """
    Görev 1 Düzeltmesi: Lip Widen (Distance-Based Weighted Displacement).
    Dudak köşesi etrafındaki noktaları Gauss ağırlığıyla yatayda kaydırarak
    yırtılmayı (pixel tearing) engeller.
    """
    try:
        lm = detect_face_landmarks(image_bgr)
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


def apply_face_slim(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """FR-17: yanak/çene hattını daralt (orta eksene doğru)."""
    lm = detect_face_landmarks(image_bgr)
    if lm is None:
        return image_bgr
    a = _clamp_intensity(intensity)
    d = np.zeros_like(lm)
    cx = float(np.mean(lm[:, 0]))
    pull = 5.0 * a
    jaw = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
    for i in jaw:
        dx = cx - lm[i, 0]
        if abs(dx) < 1e-3:
            continue
        d[i, 0] += np.sign(dx) * pull * 0.15
    return _prepare_warp(image_bgr, lm, d)