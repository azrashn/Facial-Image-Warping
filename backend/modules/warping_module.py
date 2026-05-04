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

# Rol 5 / arayüz: Türkçe ve alternatif anahtarlar → EMOJI_PRESETS içindeki İngilizce anahtar
EMOJI_PRESET_ALIASES = {
    "mutlu": "happy",
    "şaşkın": "surprised",
    "saskin": "surprised",
    "sevinçli": "joyful",
    "sevincli": "joyful",
    "nötr": "neutral",
    "notr": "neutral",
}


def resolve_emoji_preset_name(emoji_name: str) -> str:
    """Gelen emoji / ifade adını EMOJI_PRESETS anahtarına çevir."""
    key = (emoji_name or "neutral").strip().lower()
    return EMOJI_PRESET_ALIASES.get(key, key)


def get_emoji_preset_params(emoji_name: str) -> dict:
    """Preset sözlüğünden parametre vektörü (yoksa nötr)."""
    canonical = resolve_emoji_preset_name(emoji_name)
    base = EMOJI_PRESETS.get(canonical, EMOJI_PRESETS["neutral"])
    return dict(base)


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
    """Piecewise affine warp with robust triangle validation."""
    h, w = image_bgr.shape[:2]
    # Orijinalden başla: konveks dışı siyah kalmaz. src≈dst olan (deforme olmayan)
    # üçgenler önce, en büyük ötelemeli üçgenler en son boyanır ki ağız vb. ezilmesin.
    out = image_bgr.copy()

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

    tri_jobs = []
    for ia, ib, ic in tri.simplices:
        src_tri = np.asarray(
            [src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32
        ).reshape(3, 2)
        dst_tri = np.asarray(
            [dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32
        ).reshape(3, 2)
        if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
            skipped_duplicate += 1
            continue
        if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
            skipped_degenerate += 1
            continue
        disp = float(np.max(np.abs(src_tri - dst_tri)))
        tri_jobs.append((disp, ia, ib, ic))

    tri_jobs.sort(key=lambda t: t[0])

    for _disp, ia, ib, ic in tri_jobs:
        src_tri = np.asarray(
            [src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32
        ).reshape(3, 2)
        dst_tri = np.asarray(
            [dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32
        ).reshape(3, 2)

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


def apply_smile(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
        px = float(intensity) * 0.4 # Genel çekim kuvvetini biraz artırdım
        deltas = np.zeros_like(lm)
        
        face_sz = _face_scale(lm)
        sigma = face_sz * 0.15 
        
        for corner_idx, dir_x in [(61, -1.0), (291, 1.0)]:
            w = _gaussian_falloff(lm, corner_idx, sigma)
            for i in range(len(lm)):
                deltas[i, 0] += w[i] * dir_x * px
                # 0.5'i 1.2 yaptık: Dudak köşeleri artık DİREKT YUKARI kalkacak
                deltas[i, 1] -= w[i] * px * 1.2 
                
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_smile failed: %s", exc)
        return image_bgr.copy()

def apply_eyebrow_raise(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """
    Rigid-body eyebrow lift: BOTH the upper and lower boundary
    landmarks of each brow translate by the exact same delta,
    preserving original thickness.  Gaussian falloff into the
    forehead prevents mesh tearing above the brow.
    """
    try:
        lm = detect_face_landmarks(image_bgr)
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


def apply_lip_widen(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
        px = float(intensity) * 0.3
        deltas = np.zeros_like(lm)
        
        face_sz = _face_scale(lm)
        sigma = face_sz * 0.12  # Sadece dudak çevresi
        
        for corner_idx, dir_x in [(61, -1.0), (291, 1.0)]:
            w = _gaussian_falloff(lm, corner_idx, sigma)
            for i in range(len(lm)):
                deltas[i, 0] += w[i] * dir_x * px
                # Dudak genişletmede dikey (yukarı) hareket olmaz
                
        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_lip_widen failed: %s", exc)
        return image_bgr.copy()

def apply_face_slim(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """
    Face slim with smooth radial contraction toward the nose tip.
    Each jaw/cheek landmark is pulled along the vector toward the
    nose tip with a radial falloff — strongest at the outer jaw,
    decaying smoothly toward inner cheeks.
    """
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr
        strength = _clamp_intensity(intensity)
        deltas = np.zeros_like(lm)

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


def apply_eye_scaling(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    """
    Radial eye scaling: her göz, kendi köşe merkezine göre radyal öteleme.
    Landmark 33, 133 (sol göz), 362, 263 (sağ göz) — merkez çiftleri ve
    displacement = (point - center) * intensity_factor; Delaunay _prepare_warp ile uyumlu.

    intensity > 0 büyütme (dışa), intensity < 0 küçültme (içe).
    """
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None:
            return image_bgr

        intensity_factor = max(-1.0, min(1.0, float(intensity) / 100.0))
        deltas = np.zeros_like(lm)

        # Sol / sağ göz için ayrı merkez (33–133 ve 362–263 köşe çiftleri).
        center_left = (lm[33] + lm[133]) * 0.5
        center_right = (lm[362] + lm[263]) * 0.5

        left_eye_idx = [33, 133, 160, 158, 153, 144, 159, 145]
        right_eye_idx = [362, 263, 387, 385, 380, 373, 386, 374]

        for idx in left_eye_idx:
            displacement = (lm[idx] - center_left) * intensity_factor
            deltas[idx] += displacement
        for idx in right_eye_idx:
            displacement = (lm[idx] - center_right) * intensity_factor
            deltas[idx] += displacement

        return _prepare_warp(image_bgr, lm, deltas)
    except Exception as exc:
        logger.error("apply_eye_scaling failed: %s – returning original image", exc)
        return image_bgr.copy()


def apply_emoji_preset(image_bgr: np.ndarray, emoji_name: str) -> np.ndarray:
    """
    Apply predefined expression presets by chaining existing warps.
    """
    try:
        preset = get_emoji_preset_params(emoji_name)
        out = image_bgr.copy()

        if preset.get("smile", 0.0) > 0:
            out = apply_smile(out, int(round(preset["smile"] * 100)))
        if preset.get("eyebrow_raise", 0.0) > 0:
            out = apply_eyebrow_raise(out, int(round(preset["eyebrow_raise"] * 100)))
        if preset.get("lip_widen", 0.0) > 0:
            out = apply_lip_widen(out, int(round(preset["lip_widen"] * 100)))
        if preset.get("eye_enlarge", 0.0) != 0:
            out = apply_eye_scaling(out, int(round(preset["eye_enlarge"] * 100)))

        return out
    except Exception as exc:
        logger.error("apply_emoji_preset failed: %s – returning original image", exc)
        return image_bgr.copy()


def _noise_beard_texture_mask(
    shape_hw: tuple[int, int],
    blur_sigma: float = 2.2,
    threshold_val: int = 145,
) -> np.ndarray:
    """Kalıp gibi boyamak yerine gerçekçi kıl (noise) dokusu üretir."""
    h, w = shape_hw
    noise = np.zeros((h, w), dtype=np.uint8)
    cv2.randu(noise, 0, 255)
    # Önce eşikleme yapıp aralıklı noktalar (kıllar) oluşturuyoruz
    _, hair_dots = cv2.threshold(noise, 200, 255, cv2.THRESH_BINARY)
    # Sonra o noktaları çok hafif yumuşatıyoruz ki gerçekçi dursun
    hair = cv2.GaussianBlur(hair_dots, (3, 3), 0.5)
    return hair

def _lower_face_beard_polygon(lm: np.ndarray) -> np.ndarray:
    """Çene (17,18,200,199,175) + alt dudak (0,12,15) ile alt yüz poligonu."""
    beard_poly_idx = [17, 18, 200, 199, 175, 15, 12, 0]
    return np.array([[lm[i][0], lm[i][1]] for i in beard_poly_idx], dtype=np.int32).reshape((-1, 1, 2))


def _mustache_polygon_from_landmarks(lm: np.ndarray) -> Optional[np.ndarray]:
    """Burun altı ile üst dudak arası — convex hull ile kapalı poligon."""
    # Burun altı + üst dudak üst kenarı (MediaPipe 468 mesh)
    nose_lower = [2, 97, 98, 326, 327]
    upper_lip_top = [0, 267, 269, 270, 409]
    pts = []
    for i in nose_lower + upper_lip_top:
        if i < len(lm):
            pts.append([int(lm[i][0]), int(lm[i][1])])
    if len(pts) < 3:
        return None
    arr = np.array(pts, dtype=np.int32)
    return cv2.convexHull(arr)


def _alpha_blend_texture(
    image_bgr: np.ndarray,
    region_mask: np.ndarray,
    hair_mask: np.ndarray,
    tint_bgr: tuple[float, float, float],
    intensity_alpha: float,
) -> np.ndarray:
    """Maske alanına yarı saydam düz renk basmak YERİNE, kıl dokusunu uygular."""
    # 1. Kıl dokusunu (hair_mask) sadece belirlenen poligon bölgesiyle (region_mask) sınırla
    actual_hair = cv2.bitwise_and(hair_mask, region_mask)
    
    # 2. Kıl olan yerleri maske olarak kullan (0-1 aralığına getir)
    hair_f = actual_hair.astype(np.float32) / 255.0
    
    # 3. Kılları renklendir (tint_bgr)
    tint = np.zeros_like(image_bgr, dtype=np.float32)
    tint[:, :, 0] = tint_bgr[0]
    tint[:, :, 1] = tint_bgr[1]
    tint[:, :, 2] = tint_bgr[2]
    
    # 4. SADECE kıl olan pikselleri orijinal görüntüyle harmanla (intensity_alpha ile)
    # Düz maskeyi harmanlamıyoruz! Sadece kılları harmanlıyoruz.
    alpha = hair_f[..., None] * intensity_alpha
    
    out = image_bgr.astype(np.float32) * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.unit8)

def apply_mustache(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None: return image_bgr
            
        must_idx = [61, 98, 97, 164, 326, 327, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        pts_list = []
        for i in must_idx:
            pts_list.append((int(lm[i][0]), int(lm[i][1])))
        pts = np.array(pts_list, dtype=np.int32)
        
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lip_list = []
        for i in lip_idx:
            lip_list.append((int(lm[i][0]), int(lm[i][1])))
        cv2.fillPoly(mask, [np.array(lip_list, dtype=np.int32)], 0)
        
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randu(noise, 0, 255)
        _, hair = cv2.threshold(noise, 160, 255, cv2.THRESH_BINARY)
        hair_mask = cv2.bitwise_and(hair, mask)
        hair_mask = cv2.GaussianBlur(hair_mask, (3, 3), 0)
        
        alpha = (float(intensity) / 100.0) * 0.9
        hair_f = hair_mask.astype(np.float32) / 255.0
        blend_alpha = hair_f[..., None] * alpha
        tint = np.zeros_like(image_bgr, dtype=np.float32)
        tint[:] = (30.0, 30.0, 30.0) 
        
        out = image_bgr.astype(np.float32) * (1.0 - blend_alpha) + tint * blend_alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    except:
        return image_bgr

def apply_beard(image_bgr: np.ndarray, intensity: int) -> np.ndarray:
    try:
        lm = detect_face_landmarks(image_bgr)
        if lm is None: return image_bgr
            
        jaw_idx = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
        lb_idx = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        pts_list = []
        for i in (jaw_idx + lb_idx):
            pts_list.append((int(lm[i][0]), int(lm[i][1])))
        pts = np.array(pts_list, dtype=np.int32)
        
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randu(noise, 0, 255)
        _, hair = cv2.threshold(noise, 160, 255, cv2.THRESH_BINARY)
        hair_mask = cv2.bitwise_and(hair, mask)
        hair_mask = cv2.GaussianBlur(hair_mask, (3, 3), 0)
        
        alpha = (float(intensity) / 100.0) * 0.9
        hair_f = hair_mask.astype(np.float32) / 255.0
        blend_alpha = hair_f[..., None] * alpha
        tint = np.zeros_like(image_bgr, dtype=np.float32)
        tint[:] = (30.0, 30.0, 30.0)
        
        out = image_bgr.astype(np.float32) * (1.0 - blend_alpha) + tint * blend_alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    except:
        return image_bgr