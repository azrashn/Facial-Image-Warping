from __future__ import annotations
 
import logging
 
import cv2
import mediapipe as mp
import numpy as np
 
logger = logging.getLogger("facial_pipeline.hair_module")
 
# ---------------------------------------------------------------------------
# MediaPipe Selfie Segmentation (saç maskesi için)
# ---------------------------------------------------------------------------
_mp_selfie = mp.solutions.selfie_segmentation
 
# Hedef saç rengi presetleri (HSV H kanalı, 0-179 arası OpenCV değerleri)
HAIR_COLOR_PRESETS = {
    "blonde":  {"h": 28,  "s_boost": 1.3, "v_boost": 1.1},
    "red":     {"h": 10,  "s_boost": 1.5, "v_boost": 1.0},
    "black":   {"h": 0,   "s_boost": 0.5, "v_boost": 0.2},
    "blue":    {"h": 110, "s_boost": 1.6, "v_boost": 0.9},
    "brown":   {"h": 18,  "s_boost": 1.2, "v_boost": 0.7},
}
 
 
def _get_hair_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    MediaPipe Selfie Segmentation ile saç/ön plan maskesi üret.
 
    Returns
    -------
    np.ndarray  uint8 maske, 0-255 (255 = saç bölgesi)
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
 
    with _mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        results = seg.process(image_rgb)
 
    if results.segmentation_mask is None:
        logger.warning("Segmentation mask is None — returning empty mask")
        return np.zeros((h, w), dtype=np.uint8)
 
    # Segmentasyon maskesini [0,1] → [0,255] uint8'e çevir
    raw_mask = (results.segmentation_mask * 255).astype(np.uint8)
 
    # Sadece üst yarı: saç genellikle kafanın üstünde
    upper_half = np.zeros_like(raw_mask)
    upper_half[: h // 2, :] = raw_mask[: h // 2, :]
 
    # Kenarları yumuşat
    blurred = cv2.GaussianBlur(upper_half, (21, 21), 0)
    _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
 
    return binary
 
 
def apply_hair_color(
    image_bgr: np.ndarray,
    target_color: str = "blonde",
    blend_strength: float = 0.6,
    saturation_scale: float | None = None,
    value_scale: float | None = None,
) -> np.ndarray:
 
    try:
        preset = HAIR_COLOR_PRESETS.get(target_color.lower(), HAIR_COLOR_PRESETS["blonde"])
        target_h  = preset["h"]
        s_scale   = saturation_scale if saturation_scale is not None else preset["s_boost"]
        v_scale   = value_scale      if value_scale      is not None else preset["v_boost"]
 
        hair_mask = _get_hair_mask(image_bgr)
 
        if cv2.countNonZero(hair_mask) == 0:
            logger.warning("apply_hair_color: empty hair mask — returning original")
            return image_bgr.copy()
 
        # HSV dönüşümü
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
 
        # Sadece maske bölgesinde renk değiştir
        mask_bool = hair_mask > 0
 
        hsv[mask_bool, 0] = float(target_h)
        hsv[mask_bool, 1] = np.clip(hsv[mask_bool, 1] * s_scale, 0, 255)
        hsv[mask_bool, 2] = np.clip(hsv[mask_bool, 2] * v_scale, 0, 255)
 
        colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
 
        # Gaussian blur ile maske kenarlarını yumuşat
        soft_mask = cv2.GaussianBlur(hair_mask, (31, 31), 0).astype(np.float32) / 255.0
        soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)
 
        # Alpha blend
        blend = blend_strength * soft_mask_3ch
        result = (
            colored.astype(np.float32) * blend
            + image_bgr.astype(np.float32) * (1.0 - blend)
        ).astype(np.uint8)
 
        logger.info("apply_hair_color: color=%s blend=%.2f", target_color, blend_strength)
        return result
 
    except Exception as exc:
        logger.error("apply_hair_color failed: %s — returning original", exc)
        return image_bgr.copy()
 
 
def apply_hair_length(
    image_bgr: np.ndarray,
    length_delta: int = 20,
) -> np.ndarray:
    """
    Saç uzunluğunu simüle et.
 
    Güncelleme Planı v2 — Rol 4: Morfolojik dilation/erosion (DSP spatial domain).
 
    Parameters
    ----------
    image_bgr    : Giriş görüntüsü (BGR)
    length_delta : Pozitif → uzat (dilation), negatif → kıs (erosion). Piksel cinsinden.
    """
    try:
        hair_mask = _get_hair_mask(image_bgr)
 
        if cv2.countNonZero(hair_mask) == 0:
            logger.warning("apply_hair_length: empty hair mask — returning original")
            return image_bgr.copy()
 
        iterations = max(1, abs(length_delta) // 5)
 
        # Dikey dikdörtgen kernel (3×15): sadece aşağı doğru uzama
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
 
        if length_delta > 0:
            # Uzat: dilation
            new_mask = cv2.dilate(hair_mask, kernel, iterations=iterations)
        else:
            # Kıs: erosion
            new_mask = cv2.erode(hair_mask, kernel, iterations=iterations)
 
        # Uzatılan yeni bölge = new_mask - original mask
        extended_region = cv2.subtract(new_mask, hair_mask)
 
        # Orijinal saç dokusunu uzatılan bölgeye yansıt
        # Saç bölgesinin ortalama rengini al ve doku oluştur
        hair_pixels = image_bgr[hair_mask > 0]
        if len(hair_pixels) == 0:
            return image_bgr.copy()
 
        mean_color = hair_pixels.mean(axis=0).astype(np.uint8)
 
        # Noise-based doku: saç dokusuna benzer rastgele desen
        noise = np.random.randint(-20, 20, image_bgr.shape, dtype=np.int16)
        texture = np.clip(
            mean_color.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)
 
        # Dikey yönde blur (saç dağılımı etkisi)
        texture = cv2.GaussianBlur(texture, (3, 21), 0)
 
        result = image_bgr.copy()
 
        # Uzatılan bölgeye dokuyu uygula
        ext_bool = extended_region > 0
        if ext_bool.any():
            soft_ext = cv2.GaussianBlur(extended_region, (15, 15), 0).astype(np.float32) / 255.0
            soft_ext_3ch = np.stack([soft_ext] * 3, axis=-1)
 
            result = (
                texture.astype(np.float32) * soft_ext_3ch
                + result.astype(np.float32) * (1.0 - soft_ext_3ch)
            ).astype(np.uint8)
 
        # Kısaltma durumunda: orijinal saç maskesi - new_mask bölgesini arka planla doldur
        if length_delta < 0:
            removed_region = cv2.subtract(hair_mask, new_mask)
            rem_bool = removed_region > 0
            if rem_bool.any():
                # Çevre piksellerin ortalamasıyla doldur (basit inpainting)
                inpainted = cv2.inpaint(result, removed_region, 5, cv2.INPAINT_TELEA)
                result[rem_bool] = inpainted[rem_bool]
 
        logger.info("apply_hair_length: delta=%d iterations=%d", length_delta, iterations)
        return result
 
    except Exception as exc:
        logger.error("apply_hair_length failed: %s — returning original", exc)
        return image_bgr.copy()
