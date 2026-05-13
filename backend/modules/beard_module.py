from __future__ import annotations

import logging
import math
import random

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

def _draw_particle_hairs(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    landmarks: np.ndarray,
    skin_color: np.ndarray,
    darkness: float,
    alpha: float,
    is_mustache: bool = False
) -> np.ndarray:
    valid_y, valid_x = np.where(mask > 30)
    if len(valid_y) == 0:
        return image_bgr
        
    overlay = image_bgr.copy()
    
    face_width = np.linalg.norm(landmarks[234] - landmarks[454]) if len(landmarks) > 454 else 150.0
    scale = max(0.5, face_width / 150.0)

    # Sakal ve bıyık için kavisli ve uzun kıllar (Nokta nokta görünümünü kırmak için uzun çizgiler)
    if is_mustache:
        length_min = 8 * scale
        length_max = 16 * scale
        target_count = int(10000 * alpha)
    else:
        length_min = 12 * scale
        length_max = 28 * scale
        target_count = int(22000 * alpha)
        
    base_color = skin_color * (1.0 - darkness)
    base_color = np.clip(base_color, 10, 100)
    
    center_x = landmarks[164][0] if 164 < len(landmarks) else image_bgr.shape[1] / 2.0

    for _ in range(target_count):
        idx = random.randint(0, len(valid_y) - 1)
        x, y = valid_x[idx], valid_y[idx]
        
        # Yönelme (Directional Flow)
        if is_mustache:
            if x < center_x:
                base_angle = math.pi * 0.65 # Sola aşağı
            else:
                base_angle = math.pi * 0.35 # Sağa aşağı
            angle = base_angle + random.uniform(-0.25, 0.25)
        else:
            dx = center_x - x
            angle_bias = (dx / (face_width * 0.5)) * 0.25
            angle = math.pi / 2 + angle_bias + random.uniform(-0.3, 0.3)
            
        length = random.uniform(length_min, length_max)
        
        # Kavisli kıl için iki parçalı çizim
        curve_angle = angle + random.uniform(-0.4, 0.4)
        
        mid_x = int(x + math.cos(angle) * (length * 0.4))
        mid_y = int(y + math.sin(angle) * (length * 0.4))
        
        end_x = int(mid_x + math.cos(curve_angle) * (length * 0.6))
        end_y = int(mid_y + math.sin(curve_angle) * (length * 0.6))
        
        # Renk Derinliği
        noise_b = random.randint(-15, 20)
        noise_g = random.randint(-15, 20)
        noise_r = random.randint(-15, 20)
        
        c_b = int(np.clip(base_color[0] + noise_b, 0, 255))
        c_g = int(np.clip(base_color[1] + noise_g, 0, 255))
        c_r = int(np.clip(base_color[2] + noise_r, 0, 255))
        
        # Kök daha koyu
        color_root = (c_b, c_g, c_r)
        # Uçlar daha açık ve şeffaf etkisi vermek için ince
        color_tip = (min(255, c_b + 40), min(255, c_g + 40), min(255, c_r + 40))
        
        cv2.line(overlay, (x, y), (mid_x, mid_y), color_root, 2, cv2.LINE_AA)
        cv2.line(overlay, (mid_x, mid_y), (end_x, end_y), color_tip, 1, cv2.LINE_AA)

    # Genel yoğunluğu daha gerçekçi yapmak için alpha blending
    result = cv2.addWeighted(overlay, 0.85, image_bgr, 0.15, 0)
    return result

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

        # Maske kenarlarını çok hafif yumuşat ki kıllar keskin sınırda bitmesin
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return _draw_particle_hairs(image_bgr, mask, landmarks, skin_color, darkness, alpha, is_mustache=False)
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

        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return _draw_particle_hairs(image_bgr, mask, landmarks, skin_color, darkness, alpha, is_mustache=True)
    except Exception as exc:
        logger.error("apply_mustache failed: %s — returning original", exc)
        return image_bgr.copy()
