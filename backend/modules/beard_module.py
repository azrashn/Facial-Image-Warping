from __future__ import annotations
 
import logging
 
import cv2
import numpy as np
 
logger = logging.getLogger("facial_pipeline.beard_module")
 
# MediaPipe landmark indeksleri
_FULL_BEARD_POLY = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 378, 379, 365, 397, 288, 361, 323, 436, 426, 327, 164, 98, 206, 216]
_LIPS_OUTER_ORDERED = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
_MUSTACHE_POLY = [98, 327, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
_NOSE_EXCLUDE = [1, 2, 94, 97, 98, 326, 327]

def _detect_skin_tone(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
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
        return np.array([120, 100, 160], dtype=np.float32)
    return np.mean(colors, axis=0).astype(np.float32)

def _make_beard_texture(
    shape: tuple[int, int, int],
    skin_color: np.ndarray,
    darkness: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape[:2]
    
    # "Kıl Kıl" doku sentezi
    noise = np.zeros((h, w), dtype=np.uint8)
    cv2.randu(noise, 0, 255)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    _, binary = cv2.threshold(noise, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 100, 200)
    
    # Kenarları yumuşatarak float maskeye çevir
    edge_mask = cv2.GaussianBlur(edges, (3, 3), 0).astype(np.float32) / 255.0
    
    beard_color = skin_color * (1.0 - darkness * 0.8)
    beard_color = np.clip(beard_color, 10, 200)
    texture = np.full((h, w, 3), beard_color, dtype=np.float32)
    
    return texture, edge_mask

def apply_beard(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    intensity: int = 70,
    darkness: float = 0.6,
) -> np.ndarray:
    try:
        h, w = image_bgr.shape[:2]
        alpha = max(0.0, min(1.0, intensity / 100.0))
        skin_color = _detect_skin_tone(image_bgr, landmarks)

        # Tüm alt yüz maskesi
        pts = [[int(landmarks[idx][0]), int(landmarks[idx][1])] for idx in _FULL_BEARD_POLY if idx < len(landmarks)]
        if len(pts) < 3: return image_bgr.copy()
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

        # Dudak bölgesini çıkar (Dışlama)
        lip_pts = [[int(landmarks[idx][0]), int(landmarks[idx][1])] for idx in _LIPS_OUTER_ORDERED if idx < len(landmarks)]
        if len(lip_pts) >= 3:
            cv2.fillPoly(mask, [np.array(lip_pts, dtype=np.int32)], 0)

        texture, edge_mask = _make_beard_texture(image_bgr.shape, skin_color, darkness)

        # Yumuşak bölge maskesi ile "kıl" maskesini birleştir
        soft_mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        final_alpha = soft_mask * edge_mask * alpha
        final_alpha_3ch = np.stack([final_alpha] * 3, axis=-1)

        result = (texture * final_alpha_3ch + image_bgr.astype(np.float32) * (1.0 - final_alpha_3ch)).astype(np.uint8)
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

        # Bıyık bölgesi: burun altından üst dudağa kadar
        pts = [[int(landmarks[idx][0]), int(landmarks[idx][1])] for idx in _MUSTACHE_POLY if idx < len(landmarks)]
        if len(pts) < 3: return image_bgr.copy()

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

        # Burun deliklerini çıkar
        nose_pts = [[int(landmarks[idx][0]), int(landmarks[idx][1])] for idx in _NOSE_EXCLUDE if idx < len(landmarks)]
        if len(nose_pts) >= 3:
            hull = cv2.convexHull(np.array(nose_pts, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 0)

        texture, edge_mask = _make_beard_texture(image_bgr.shape, skin_color, darkness)

        soft_mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        final_alpha = soft_mask * edge_mask * alpha
        final_alpha_3ch = np.stack([final_alpha] * 3, axis=-1)

        result = (texture * final_alpha_3ch + image_bgr.astype(np.float32) * (1.0 - final_alpha_3ch)).astype(np.uint8)
        return result
    except Exception as exc:
        logger.error("apply_mustache failed: %s — returning original", exc)
        return image_bgr.copy()
