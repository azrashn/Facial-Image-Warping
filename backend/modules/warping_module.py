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
    a, b, c = tri
    return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0

def _has_duplicate_vertices(tri: np.ndarray, eps: float = 1e-4) -> bool:
    a, b, c = tri
    if np.linalg.norm(a - b) < eps or np.linalg.norm(b - c) < eps or np.linalg.norm(a - c) < eps:
        return True
    return False

_TASK_LANDMARKER = None
_TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

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
        _TASK_LANDMARKER = FaceLandmarker.create_from_model_path(_face_landmarker_model_path())
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
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as face_mesh:
            res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    return _landmarks_via_tasks(image_bgr, h, w)

def _corners(width: int, height: int) -> np.ndarray:
    return np.array([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]], dtype=np.float32)

def geometric_warp(image_bgr: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = np.zeros_like(image_bgr)
    try:
        tri = Delaunay(dst_pts)
    except Exception:
        return image_bgr.copy()

    warped_ok = 0
    for ia, ib, ic in tri.simplices:
        src_tri = np.asarray([src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32).reshape(3, 2)
        dst_tri = np.asarray([dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32).reshape(3, 2)

        if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri): continue
        if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3: continue

        r = cv2.boundingRect(dst_tri)
        bx, by, bw, bh = r
        bx, by = max(bx, 0), max(by, 0)
        bw, bh = min(bw, w - bx), min(bh, h - by)
        if bw <= 0 or bh <= 0: continue

        try:
            mask = np.zeros((bh, bw), dtype=np.uint8)
            dst_crop = np.asarray(dst_tri - [bx, by], dtype=np.float32).reshape(3, 2)
            src_crop = np.asarray(src_tri, dtype=np.float32).reshape(3, 2)
            cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)

            warp_mat = cv2.getAffineTransform(src_crop, dst_crop)
            warped_patch = cv2.warpAffine(image_bgr, warp_mat, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            roi = out[by : by + bh, bx : bx + bw]
            blended = np.where(mask[..., None] == 255, warped_patch, roi)
            out[by : by + bh, bx : bx + bw] = blended
            warped_ok += 1
        except Exception:
            continue

    if warped_ok == 0:
        return image_bgr.copy()
    return out

def _prepare_warp(image_bgr: np.ndarray, src_lm: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    try:
        dst = src_lm + deltas
        height, width = image_bgr.shape[:2]
        corners = _corners(width, height)
        return geometric_warp(image_bgr, np.vstack([src_lm, corners]), np.vstack([dst, corners]))
    except Exception:
        return image_bgr.copy()

def _gaussian_falloff(lm: np.ndarray, anchor_idx: int, sigma: float) -> np.ndarray:
    dists = np.linalg.norm(lm - lm[anchor_idx], axis=1)
    return np.exp(-0.5 * (dists / max(sigma, 1e-6)) ** 2)

def _face_scale(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[133] - lm[362]))

def apply_smile(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None: return image_bgr.copy()
        px = float(intensity)
        deltas = np.zeros_like(lm)
        sigma = _face_scale(lm) * 0.25
        w_left, w_right = _gaussian_falloff(lm, 61, sigma), _gaussian_falloff(lm, 291, sigma)
        
        center_x, half_width = (lm[61, 0] + lm[291, 0]) / 2.0, abs(lm[291, 0] - lm[61, 0]) / 2.0 + 1e-6
        move_x, move_y = px * 0.6, px * 1.0  
        
        for i in range(len(lm)):
            dy_damp = 1.0 - np.exp(-0.5 * (abs(lm[i, 0] - center_x) / (half_width * 0.6)) ** 2)
            deltas[i, 0] += w_left[i] * (-move_x) + w_right[i] * move_x
            deltas[i, 1] += (w_left[i] * (-move_y) + w_right[i] * (-move_y)) * dy_damp
            
        for idx in [1, 4, 5, 19, 94, 111, 117, 118, 119, 340, 346, 347, 348, 152, 148, 176, 149, 150, 377, 400, 378, 379, 365]:
            deltas[idx] = 0.0
        deltas[np.abs(deltas) < 0.1] = 0.0
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception: return image_bgr.copy()

def apply_eyebrow_raise(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None: return image_bgr.copy()
        deltas = np.zeros_like(lm)
        lift = _face_scale(lm) * 0.10 * _clamp_intensity(intensity)
        
        all_brow = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285]
        for idx in all_brow: deltas[idx, 1] -= lift

        sigma_fg = _face_scale(lm) * 0.25
        w_l, w_r = _gaussian_falloff(lm, 66, sigma_fg), _gaussian_falloff(lm, 296, sigma_fg)
        
        for i in range(len(lm)):
            if lm[i, 1] < lm[66, 1]: deltas[i, 1] -= w_l[i] * (lift * 0.4)
            if lm[i, 1] < lm[296, 1]: deltas[i, 1] -= w_r[i] * (lift * 0.4)
        for idx in all_brow: deltas[idx, 1] = -lift
        
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception: return image_bgr.copy()

def apply_lip_widen(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None: return image_bgr.copy()
        deltas = np.zeros_like(lm)
        sigma = _face_scale(lm) * 0.15
        w_l, w_r = _gaussian_falloff(lm, 61, sigma), _gaussian_falloff(lm, 291, sigma)
        
        for i in range(len(lm)):
            deltas[i, 0] += w_l[i] * (-float(intensity)) + w_r[i] * float(intensity)
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception: return image_bgr.copy()

def apply_face_slim(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
        strength = _clamp_intensity(intensity)
        deltas = np.zeros_like(lm)

        face_sz = _face_scale(lm)
        nose_tip = lm[1].copy()

        jaw_contour = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]

        jaw_positions = lm[jaw_contour]
        jaw_vecs = jaw_positions - nose_tip
        jaw_dists = np.linalg.norm(jaw_vecs, axis=1)
        max_jaw_dist = float(np.max(jaw_dists)) if np.max(jaw_dists) > 1e-3 else 1.0

        max_pull = face_sz * 0.10 * strength

        for i, idx in enumerate(jaw_contour):
            vec = lm[idx] - nose_tip
            dist = float(np.linalg.norm(vec))
            if dist < 1e-3: continue

            weight = (dist / max_jaw_dist) ** 2
            direction = -vec / dist 
            
            deltas[idx, 0] += direction[0] * weight * max_pull
            deltas[idx, 1] += direction[1] * weight * max_pull * 0.3 

        sigma_spread = face_sz * 0.15
        jaw_set = set(jaw_contour)
        for anchor_idx in jaw_contour:
            if abs(deltas[anchor_idx, 0]) < 1e-6 and abs(deltas[anchor_idx, 1]) < 1e-6: continue
            w = _gaussian_falloff(lm, anchor_idx, sigma_spread)
            for i in range(len(lm)):
                if i in jaw_set: continue 
                deltas[i, 0] += w[i] * deltas[anchor_idx, 0] * 0.3
                deltas[i, 1] += w[i] * deltas[anchor_idx, 1] * 0.3

        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error(f"apply_face_slim failed: {exc}")
        return image_bgr.copy()

def apply_eye_scaling(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None: return image_bgr.copy()
        deltas, factor = np.zeros_like(lm), max(-1.0, min(1.0, float(intensity) / 100.0))
        
        c_left = np.mean(lm[[33, 133, 160, 158, 153, 144, 159, 145]], axis=0)
        c_right = np.mean(lm[[362, 263, 387, 385, 380, 373, 386, 374]], axis=0)
        sigma = _face_scale(lm) * 0.12
        
        w_left = np.exp(-0.5 * (np.linalg.norm(lm - c_left, axis=1) / max(sigma, 1e-6)) ** 2)
        w_right = np.exp(-0.5 * (np.linalg.norm(lm - c_right, axis=1) / max(sigma, 1e-6)) ** 2)
        
        for i in range(len(lm)):
            deltas[i] += (lm[i] - c_left) * factor * w_left[i] + (lm[i] - c_right) * factor * w_right[i]
            
        for idx in [70, 63, 105, 66, 107, 46, 53, 52, 65, 55, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285, 168, 6, 197, 195, 5, 4, 116, 117, 118, 119, 100, 101, 345, 346, 347, 348, 329, 330, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]: deltas[idx] = 0.0
        deltas[np.abs(deltas) < 0.1] = 0.0
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception: return image_bgr.copy()

def apply_alien_emoji(image_bgr: np.ndarray, intensity: int = 100) -> np.ndarray:
    """
    👽 Uzaylı filtresi:
    - Ters üçgen kafa (çene ince, alın geniş)
    - Büyük siyah oval gözler
    - Yeşil cilt tonu
    """
    try:
        h, w = image_bgr.shape[:2]

        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr.copy()

        face_sz = _face_scale(lm)
        deltas = np.zeros_like(lm)

        nose_tip = lm[1].copy()

        chin_indices = [
            152, 377, 400, 378, 379, 365, 397, 288,
            361, 323, 148, 176, 149, 150, 136, 172, 58, 132
        ]
        for idx in set(chin_indices):
            vec = lm[idx] - nose_tip
            dist = float(np.linalg.norm(vec))
            if dist < 1e-3:
                continue
            pull = face_sz * 0.18
            direction = -vec / dist
            deltas[idx, 0] += direction[0] * pull * 0.8
            deltas[idx, 1] += direction[1] * pull * 0.3

        temple_indices = [234, 454, 127, 356, 162, 389]
        for idx in temple_indices:
            cx = w / 2.0
            dx = lm[idx, 0] - cx
            deltas[idx, 0] += np.sign(dx) * face_sz * 0.08

        left_eye_pts  = [33, 133, 160, 159, 158, 157, 163, 144, 145, 153, 154, 155, 173, 246, 161]
        right_eye_pts = [362, 263, 387, 386, 385, 384, 390, 373, 374, 380, 381, 382, 398, 466, 388]

        c_left  = lm[left_eye_pts].mean(axis=0)
        c_right = lm[right_eye_pts].mean(axis=0)

        eye_scale = 0.8
        sigma = face_sz * 0.20

        for i in range(len(lm)):
            d_left  = lm[i] - c_left
            d_right = lm[i] - c_right
            w_left  = np.exp(-0.5 * (np.linalg.norm(d_left)  / max(sigma, 1e-6)) ** 2)
            w_right = np.exp(-0.5 * (np.linalg.norm(d_right) / max(sigma, 1e-6)) ** 2)
            deltas[i] += d_left  * eye_scale * w_left
            deltas[i] += d_right * eye_scale * w_right

        fixed = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                 1, 4, 5, 168, 6, 197, 195]
        for idx in fixed:
            deltas[idx] = 0.0
        deltas[np.abs(deltas) < 0.1] = 0.0

        base = _prepare_warp(image_bgr, lm, deltas)
        base = apply_eyebrow_raise(base, 40)

        lm2 = detect_face_landmarks(base)
        if lm2 is None:
            lm2 = lm

        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        jaw_pts = np.array([[int(lm2[i][0]), int(lm2[i][1])] for i in jaw_indices], dtype=np.int32)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        face_mask_blur = cv2.GaussianBlur(face_mask, (31, 31), 0).astype(np.float32) / 255.0
        face_mask_3ch = np.stack([face_mask_blur] * 3, axis=-1)

        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_green = hsv.copy()
        hsv_green[:, :, 0] = 75.0
        hsv_green[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2 + 25, 0, 255)
        hsv_green[:, :, 2] = np.clip(hsv[:, :, 2] * 0.88, 0, 255)
        green_img = cv2.cvtColor(hsv_green.astype(np.uint8), cv2.COLOR_HSV2BGR)

        result = (
            green_img.astype(np.float32) * face_mask_3ch * 0.60
            + base.astype(np.float32) * (1.0 - face_mask_3ch * 0.60)
        ).astype(np.uint8)

        lm3 = detect_face_landmarks(result)
        if lm3 is None:
            lm3 = lm2

        c_left2  = lm3[left_eye_pts].mean(axis=0)
        c_right2 = lm3[right_eye_pts].mean(axis=0)

        eye_rx = int(face_sz * 0.28)
        eye_ry = int(face_sz * 0.22)

        eye_layer = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(eye_layer, (int(c_left2[0]),  int(c_left2[1])),  (eye_rx, eye_ry), 0, 0, 360, (12, 12, 12), -1)
        cv2.ellipse(eye_layer, (int(c_right2[0]), int(c_right2[1])), (eye_rx, eye_ry), 0, 0, 360, (12, 12, 12), -1)

        ho = int(eye_rx * 0.28)
        vo = int(eye_ry * 0.28)
        cv2.circle(eye_layer, (int(c_left2[0])  - ho, int(c_left2[1])  - vo), int(eye_rx * 0.12), (70, 70, 70), -1)
        cv2.circle(eye_layer, (int(c_right2[0]) - ho, int(c_right2[1]) - vo), int(eye_rx * 0.12), (70, 70, 70), -1)

        eye_mask = (eye_layer.sum(axis=2) > 0).astype(np.float32)
        eye_mask = cv2.GaussianBlur(eye_mask, (5, 5), 0)
        eye_mask_3ch = np.stack([eye_mask] * 3, axis=-1)

        result = (
            eye_layer.astype(np.float32) * eye_mask_3ch
            + result.astype(np.float32) * (1.0 - eye_mask_3ch)
        ).astype(np.uint8)

        return result

    except Exception as exc:
        logger.error(f"apply_alien_emoji failed: {exc}")
        return image_bgr.copy()


def apply_clown_emoji(image_bgr: np.ndarray, intensity: int = 100) -> np.ndarray:
    """
    🤡 Joker tarzı palyaço:
    - Beyaz yüz boyası
    - Büyük mavi eşkenar dörtgen göz makyajı
    - Kırmızı kaşlar
    - Büyük kırmızı dudak boyası
    - Çok büyük kırmızı top burun
    - Geniş kırmızı gülüş çizgisi
    """
    try:
        h, w = image_bgr.shape[:2]

        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr.copy()

        result = image_bgr.copy()
        face_sz = _face_scale(lm)

        # Yüz maskesi
        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        jaw_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in jaw_indices], dtype=np.int32)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        face_mask_blur = cv2.GaussianBlur(face_mask, (25, 25), 0).astype(np.float32) / 255.0
        face_mask_3ch = np.stack([face_mask_blur] * 3, axis=-1)

        # 1. Beyaz yüz boyası %55
        white = np.ones_like(result, dtype=np.float32) * 255
        result = (
            white * face_mask_3ch * 0.55
            + result.astype(np.float32) * (1.0 - face_mask_3ch * 0.55)
        ).astype(np.uint8)

        paint = np.zeros((h, w, 3), dtype=np.float32)

        # 2. Büyük mavi eşkenar dörtgen göz makyajı
        le_cx = int((lm[33][0]  + lm[133][0]) / 2)
        le_cy = int((lm[33][1]  + lm[133][1]) / 2)
        re_cx = int((lm[362][0] + lm[263][0]) / 2)
        re_cy = int((lm[362][1] + lm[263][1]) / 2)

        # Eşkenar dörtgen: 4 köşesi eşit uzaklıkta
        e_r = int(face_sz * 0.22)  # tüm yönlerde eşit yarıçap

        def rhombus_pts(cx, cy, r):
            return np.array([
                [cx - r, cy],   # sol
                [cx,     cy - r],  # üst
                [cx + r, cy],   # sağ
                [cx,     cy + r],  # alt
            ], dtype=np.int32)

        cv2.fillPoly(paint, [rhombus_pts(le_cx, le_cy, e_r)], (210, 90, 10))  # mavi BGR
        cv2.fillPoly(paint, [rhombus_pts(re_cx, re_cy, e_r)], (210, 90, 10))

        # 3. Kırmızı kaşlar
        left_brow_pts  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_brow_pts = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        lb_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in left_brow_pts],  dtype=np.int32)
        rb_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in right_brow_pts], dtype=np.int32)
        brow_thick = max(int(face_sz * 0.06), 3)
        cv2.polylines(paint, [lb_pts], False, (0, 0, 220), brow_thick)
        cv2.polylines(paint, [rb_pts], False, (0, 0, 220), brow_thick)

        # 4. Büyük kırmızı top burun
        nose_pt = (int(lm[4][0]), int(lm[4][1]))
        nose_r  = int(face_sz * 0.20)  # büyük
        cv2.circle(paint, nose_pt, nose_r, (0, 0, 240), -1)
        cv2.circle(paint,
                   (nose_pt[0] - int(nose_r * 0.3), nose_pt[1] - int(nose_r * 0.35)),
                   int(nose_r * 0.22), (100, 100, 255), -1)

        # 5. Büyük kırmızı dudak boyası
        outer_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        om_pts = np.array([[int(lm[i][0]), int(lm[i][1])] for i in outer_mouth], dtype=np.int32)
        om_center = om_pts.mean(axis=0).astype(int)
        om_big = ((om_pts - om_center) * 1.35 + om_center).astype(np.int32)
        cv2.fillPoly(paint, [om_big], (0, 0, 225))

        # 6. Geniş kırmızı gülüş çizgisi
        left_corner  = (int(lm[61][0]),  int(lm[61][1]))
        right_corner = (int(lm[291][0]), int(lm[291][1]))
        left_cheek   = (int(lm[205][0] - face_sz * 0.20), int(lm[205][1] + face_sz * 0.05))
        right_cheek  = (int(lm[425][0] + face_sz * 0.20), int(lm[425][1] + face_sz * 0.05))
        line_w = max(int(face_sz * 0.08), 4)
        cv2.line(paint, left_corner,  left_cheek,  (0, 0, 225), line_w)
        cv2.line(paint, right_corner, right_cheek, (0, 0, 225), line_w)

        # Makyajı blend et
        paint_blur  = cv2.GaussianBlur(paint, (9, 9), 0)
        paint_alpha = np.clip(paint_blur.sum(axis=2, keepdims=True) / 280.0, 0, 1)
        paint_alpha = np.repeat(paint_alpha, 3, axis=2)

        final = (
            paint_blur * paint_alpha * 0.85
            + result.astype(np.float32) * (1.0 - paint_alpha * 0.85)
        ).astype(np.uint8)

        return final

    except Exception as e:
        logger.error(f"apply_clown_emoji failed: {e}")
        return image_bgr.copy()
    
    
def apply_emoji_preset(image_bgr: np.ndarray, emoji_name: str) -> np.ndarray:
    try:
        preset_key = (emoji_name or "neutral").strip().lower()
        if preset_key == "alien":
            return apply_alien_emoji(image_bgr, 100)
        if preset_key == "clown" or preset_key == "joker":
            return apply_clown_emoji(image_bgr, 100)
            
        preset = EMOJI_PRESETS.get(preset_key, EMOJI_PRESETS["neutral"])
        out = image_bgr.copy()

        if preset.get("smile", 0.0) > 0: out = apply_smile(out, int(round(preset["smile"] * 100)))
        if preset.get("eyebrow_raise", 0.0) > 0: out = apply_eyebrow_raise(out, int(round(preset["eyebrow_raise"] * 100)))
        if preset.get("lip_widen", 0.0) > 0: out = apply_lip_widen(out, int(round(preset["lip_widen"] * 100)))
        if preset.get("eye_enlarge", 0.0) != 0: out = apply_eye_scaling(out, int(round(preset["eye_enlarge"] * 100)))

        return out
    except Exception as exc: return image_bgr.copy()
