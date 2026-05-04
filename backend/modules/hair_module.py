from __future__ import annotations
import logging
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import urllib.request

logger = logging.getLogger("facial_pipeline.hair_module")

_TASK_SEGMENTER = None
_SEG_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"

def _hair_segmenter_model_path() -> str:
    cache_dir = os.path.join(tempfile.gettempdir(), "facial_image_warping_mp")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "hair_segmenter.tflite")
    if not os.path.isfile(path) or os.path.getsize(path) < 1024 * 1024:
        urllib.request.urlretrieve(_SEG_MODEL_URL, path)
    return path

def _get_segmenter():
    global _TASK_SEGMENTER
    if _TASK_SEGMENTER is None:
        from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=_hair_segmenter_model_path()),
            output_confidence_masks=True,
            output_category_mask=False
        )
        _TASK_SEGMENTER = ImageSegmenter.create_from_options(options)
    return _TASK_SEGMENTER

HAIR_COLOR_PRESETS = {
    "blonde":  {"h": 28,  "s_boost": 1.3, "v_boost": 1.1},
    "red":     {"h": 10,  "s_boost": 1.5, "v_boost": 1.0},
    "black":   {"h": 0,   "s_boost": 0.5, "v_boost": 0.2},
    "blue":    {"h": 110, "s_boost": 1.6, "v_boost": 0.9},
    "brown":   {"h": 18,  "s_boost": 1.2, "v_boost": 0.7},
}

def _get_hair_mask(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        # İŞTE SENİ ÇILDIRTAN O HATALI IMPORTLARI BURADAN TAMAMEN YOK ETTİK
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = _get_segmenter().segment(mp_image)
        
        if not result.confidence_masks:
            logger.warning("Segmentation mask is None — returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)
            
        idx = 1 if len(result.confidence_masks) > 1 else 0
        raw_mask = (result.confidence_masks[idx].numpy_view() * 255).astype(np.uint8)
    except Exception as e:
        logger.warning("Hair segmenter failed: %s — falling back to empty mask", e)
        return np.zeros((h, w), dtype=np.uint8)

    blurred = cv2.GaussianBlur(raw_mask, (21, 21), 0)
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
            return image_bgr.copy()

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask_bool = hair_mask > 0

        hsv[mask_bool, 0] = float(target_h)
        hsv[mask_bool, 1] = np.clip(hsv[mask_bool, 1] * s_scale, 0, 255)
        hsv[mask_bool, 2] = np.clip(hsv[mask_bool, 2] * v_scale, 0, 255)

        colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        soft_mask = cv2.GaussianBlur(hair_mask, (31, 31), 0).astype(np.float32) / 255.0
        soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

        blend = blend_strength * soft_mask_3ch
        result = (
            colored.astype(np.float32) * blend
            + image_bgr.astype(np.float32) * (1.0 - blend)
        ).astype(np.uint8)

        return result

    except Exception as exc:
        logger.error("apply_hair_color failed: %s — returning original", exc)
        return image_bgr.copy()

def apply_hair_length(image_bgr: np.ndarray, length_delta: int = 20) -> np.ndarray:
    try:
        hair_mask = _get_hair_mask(image_bgr)
        if cv2.countNonZero(hair_mask) == 0:
            return image_bgr.copy()

        iterations = max(1, abs(length_delta) // 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

        if length_delta > 0:
            new_mask = cv2.dilate(hair_mask, kernel, iterations=iterations)
        else:
            new_mask = cv2.erode(hair_mask, kernel, iterations=iterations)

        extended_region = cv2.subtract(new_mask, hair_mask)
        hair_pixels = image_bgr[hair_mask > 0]
        if len(hair_pixels) == 0:
            return image_bgr.copy()

        mean_color = hair_pixels.mean(axis=0).astype(np.uint8)
        noise = np.random.randint(-20, 20, image_bgr.shape, dtype=np.int16)
        texture = np.clip(mean_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 21), 0)

        result = image_bgr.copy()
        ext_bool = extended_region > 0
        if ext_bool.any():
            soft_ext = cv2.GaussianBlur(extended_region, (15, 15), 0).astype(np.float32) / 255.0
            soft_ext_3ch = np.stack([soft_ext] * 3, axis=-1)
            result = (
                texture.astype(np.float32) * soft_ext_3ch
                + result.astype(np.float32) * (1.0 - soft_ext_3ch)
            ).astype(np.uint8)

        if length_delta < 0:
            removed_region = cv2.subtract(hair_mask, new_mask)
            rem_bool = removed_region > 0
            if rem_bool.any():
                inpainted = cv2.inpaint(result, removed_region, 5, cv2.INPAINT_TELEA)
                result[rem_bool] = inpainted[rem_bool]

        return result

    except Exception as exc:
        logger.error("apply_hair_length failed: %s — returning original", exc)
        return image_bgr.copy()
