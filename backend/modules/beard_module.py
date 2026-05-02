from __future__ import annotations
 
import logging
 
import cv2
import numpy as np
 
logger = logging.getLogger("facial_pipeline.beard_module")
 
# MediaPipe landmark indeksleri
_JAW_BOTTOM    = [17, 18, 200, 199, 175]   # çene hattı alt noktaları
_LOWER_LIP     = [0, 12, 15, 16, 17]       # alt dudak alt kenarı
_CHIN_CONTOUR  = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
                  454, 356, 389, 251, 284, 332, 297, 338, 10]
 
_MUSTACHE_TOP    = [0, 267, 269, 270, 409]   # üst dudak üst kenarı
_MUSTACHE_BOTTOM = [37, 39, 40, 185, 61]     # üst dudak alt kenarı
_NOSE_BOTTOM     = [2, 326, 327, 4, 97, 98]  # burun alt noktaları
 
 
def _detect_skin_tone(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
   
    h, w = image_bgr.shape[:2]
 
    # Yanak landmark'ları: 234 (sol yanak), 454 (sağ yanak)
    cheek_indices = [234, 454, 205, 425]
    colors = []
 
    for idx in cheek_indices:
        if idx < len(landmarks):
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            x = max(5, min(w - 6, x))
            y = max(5, min(h - 6, y))
            patch = image_bgr[y - 5:y + 5, x - 5:x + 5]
            if patch.size > 0:
                colors.append(patch.mean(axis=(0, 1)))
 
    if not colors:
        return np.array([120, 100, 160], dtype=np.float32)  # varsayılan
 
    return np.mean(colors, axis=0).astype(np.float32)
 
 
def _make_beard_texture(
    shape: tuple[int, int, int],
    skin_color: np.ndarray,
    darkness: float = 0.5,
) -> np.ndarray:
   
    h, w = shape[:2]
 
    # Rastgele lif deseni
    noise = np.random.randint(0, 60, (h, w), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (3, 5), 0)
 
    # Sakal rengi = cilt tonu * (1 - darkness)
    beard_color = skin_color * (1.0 - darkness * 0.8)
    beard_color = np.clip(beard_color, 10, 200)
 
    texture = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        texture[:, :, c] = np.clip(
            beard_color[c] + noise.astype(np.float32) * 0.3, 0, 255
        ).astype(np.uint8)
 
    return texture
 
 
def apply_beard(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    intensity: int = 70,
    darkness: float = 0.6,
) -> np.ndarray:
   
    try:
        h, w = image_bgr.shape[:2]
        alpha = max(0.0, min(1.0, intensity / 100.0))
 
        # Cilt tonu tespiti (Rol 4)
        skin_color = _detect_skin_tone(image_bgr, landmarks)
 
        # Alt yüz maskesi: alt dudak altından çene hattına kadar
        beard_indices = _LOWER_LIP + _JAW_BOTTOM + [152, 148, 176, 149, 150, 136]
        pts = []
        for idx in beard_indices:
            if idx < len(landmarks):
                pts.append([int(landmarks[idx][0]), int(landmarks[idx][1])])
 
        if len(pts) < 3:
            logger.warning("apply_beard: insufficient landmarks")
            return image_bgr.copy()
 
        pts_array = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts_array)
 
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
 
        # Üst dudak üstünü maskeden çıkar (sadece alt dudak altı)
        upper_lip_pts = []
        for idx in _MUSTACHE_TOP + _MUSTACHE_BOTTOM:
            if idx < len(landmarks):
                upper_lip_pts.append([int(landmarks[idx][0]), int(landmarks[idx][1])])
 
        if len(upper_lip_pts) >= 3:
            upper_hull = cv2.convexHull(np.array(upper_lip_pts, dtype=np.int32))
            cv2.fillPoly(mask, [upper_hull], 0)
 
        # Doku oluştur
        texture = _make_beard_texture(image_bgr.shape, skin_color, darkness)
 
        # Kenarları yumuşat
        soft_mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)
 
        blend = alpha * soft_mask_3ch
        result = (
            texture.astype(np.float32) * blend
            + image_bgr.astype(np.float32) * (1.0 - blend)
        ).astype(np.uint8)
 
        logger.info("apply_beard: intensity=%d darkness=%.2f", intensity, darkness)
        return result
 
    except Exception as exc:
        logger.error("apply_beard failed: %s — returning original", exc)
        return image_bgr.copy()
 
 
def apply_mustache(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    intensity: int = 70,
    darkness: float = 0.7,
) -> np.ndarray:
   
    try:
        h, w = image_bgr.shape[:2]
        alpha = max(0.0, min(1.0, intensity / 100.0))
 
        skin_color = _detect_skin_tone(image_bgr, landmarks)
 
        # Bıyık bölgesi: burun alt noktaları + üst dudak üst kenarı
        mustache_indices = _NOSE_BOTTOM + _MUSTACHE_TOP
        pts = []
        for idx in mustache_indices:
            if idx < len(landmarks):
                pts.append([int(landmarks[idx][0]), int(landmarks[idx][1])])
 
        if len(pts) < 3:
            logger.warning("apply_mustache: insufficient landmarks")
            return image_bgr.copy()
 
        pts_array = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts_array)
 
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
 
        texture = _make_beard_texture(image_bgr.shape, skin_color, darkness)
 
        soft_mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)
 
        blend = alpha * soft_mask_3ch
        result = (
            texture.astype(np.float32) * blend
            + image_bgr.astype(np.float32) * (1.0 - blend)
        ).astype(np.uint8)
 
        logger.info("apply_mustache: intensity=%d darkness=%.2f", intensity, darkness)
        return result
 
    except Exception as exc:
        logger.error("apply_mustache failed: %s — returning original", exc)
        return image_bgr.copy()
