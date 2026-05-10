import logging
import os
from pathlib import Path
 
import cv2
import numpy as np
 
logger = logging.getLogger("facial_pipeline.ai_module")
 

# Model paths  (relative to *this* file → ../models/)

_MODULE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _MODULE_DIR.parent / "models"

# Türkçe karakter sorunu için dinamik path
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

import logging
logger = logging.getLogger("facial_pipeline.ai_module")
logger.info("MODELS_DIR: %s", _MODELS_DIR)
logger.info("face_deploy exists: %s", (_MODELS_DIR / "face_deploy.prototxt").exists())
 
_FACE_PROTO = str(_MODELS_DIR / "face_deploy.prototxt")
_FACE_MODEL = str(_MODELS_DIR / "face_net.caffemodel")
_AGE_PROTO  = str(_MODELS_DIR / "age_deploy.prototxt")
_AGE_MODEL  = str(_MODELS_DIR / "age_net.caffemodel")
 
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
 
AGE_MIDPOINTS = np.array([1, 5, 10, 18, 29, 40, 50, 75], dtype=np.float64)
 
_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
 
# Güven aralığı sabiti (±yıl)
AGE_CONFIDENCE_INTERVAL = 3
 
 
def _load_nets():
    """Load face-detector and age-classifier networks (lazy, cached)."""
    if not hasattr(_load_nets, "_face_net"):
        for p in (_FACE_PROTO, _FACE_MODEL, _AGE_PROTO, _AGE_MODEL):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Model file not found: {p}")
        _load_nets._face_net = cv2.dnn.readNetFromCaffe(_FACE_PROTO, _FACE_MODEL)
        _load_nets._age_net  = cv2.dnn.readNetFromCaffe(_AGE_PROTO, _AGE_MODEL)
        logger.info("DNN models loaded from %s", _MODELS_DIR)
    return _load_nets._face_net, _load_nets._age_net
 
 
def estimate_age(image_bgr: np.ndarray, confidence_threshold: float = 0.4) -> dict:
    """
    Detect the largest face in *image_bgr* and return an estimated age.
 
    Returns
    -------
    dict
        ``{"status": "success", "estimated_age": int, "age_bucket": str,
           "confidence": float, "age_range": {"min": int, "max": int}}``
        or ``{"status": "failed", "error": str}`` on failure.
    """
    try:
        face_net, age_net = _load_nets()
    except FileNotFoundError as exc:
        return {"status": "failed", "error": str(exc)}
 
    h, w = image_bgr.shape[:2]
 
    # --- 1. Yüz tespiti ------------------------------------------------
    blob = cv2.dnn.blobFromImage(
        image_bgr, 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False,
    )
    face_net.setInput(blob)
    detections = face_net.forward()
 
    best_idx, best_conf = -1, 0.0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > best_conf:
            best_conf = conf
            best_idx = i
 
    if best_idx == -1 or best_conf < confidence_threshold:
        logger.warning("No face detected (best conf=%.3f) – using full image.", best_conf)
        face_crop = image_bgr.copy()
    else:
        box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        pad = int(0.2 * max(x2 - x1, y2 - y1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        face_crop = image_bgr[y1:y2, x1:x2]
 
    if face_crop.size == 0:
        return {"status": "failed", "error": "Face crop is empty."}
 
    # --- 2. Yaş tahmini ------------------------------------------------
    scales = [(227, 227), (256, 256)]
    all_preds = []
    for sz in scales:
        age_blob = cv2.dnn.blobFromImage(
            face_crop, 1.0, sz, _MODEL_MEAN, swapRB=False,
        )
        age_net.setInput(age_blob)
        preds = age_net.forward()
        all_preds.append(preds[0])
 
    flipped = cv2.flip(face_crop, 1)
    age_blob_flip = cv2.dnn.blobFromImage(
        flipped, 1.0, (227, 227), _MODEL_MEAN, swapRB=False,
    )
    age_net.setInput(age_blob_flip)
    preds_flip = age_net.forward()
    all_preds.append(preds_flip[0])
 
    avg_preds = np.mean(all_preds, axis=0)
 
    probs = avg_preds / (avg_preds.sum() + 1e-9)
    estimated_age = int(round(float(np.dot(probs, AGE_MIDPOINTS))))
    estimated_age = max(1, min(100, estimated_age))
 
    bucket_idx = int(avg_preds.argmax())
    age_bucket = AGE_BUCKETS[bucket_idx]
 
    # --- 3. Güven aralığı (±AGE_CONFIDENCE_INTERVAL yıl) --------------
    age_range = {
        "min": max(1, estimated_age - AGE_CONFIDENCE_INTERVAL),
        "max": min(100, estimated_age + AGE_CONFIDENCE_INTERVAL),
    }
 
    logger.info(
        "Age estimation → bucket=%s  age≈%d  face_conf=%.3f",
        age_bucket, estimated_age, best_conf,
    )
 
    return {
        "status": "success",
        "estimated_age": estimated_age,
        "age_bucket": age_bucket,
        "confidence": round(best_conf, 3),
        "age_range": age_range,
    }
 
 
def estimate_age_before_after(
    before_image_bgr: np.ndarray,
    after_image_bgr: np.ndarray,
    confidence_threshold: float = 0.4,
) -> dict:
    """
    İki görüntü için yaş tahmini yap ve farkı hesapla.
 
    Güncelleme Planı v2 — Rol 4: before/after yaş karşılaştırması.
 
    Returns
    -------
    dict  {
        "before": { estimated_age, age_bucket, confidence, age_range },
        "after":  { estimated_age, age_bucket, confidence, age_range },
        "age_difference": int,   # pozitif = yaşlandı, negatif = gençleşti
        "difference_label": str  # "Bu işlem görünen yaşı +8 artırdı" gibi
    }
    """
    before_result = estimate_age(before_image_bgr, confidence_threshold)
    after_result  = estimate_age(after_image_bgr, confidence_threshold)
 
    if before_result.get("status") != "success":
        return {"status": "failed", "error": f"Before image: {before_result.get('error')}"}
    if after_result.get("status") != "success":
        return {"status": "failed", "error": f"After image: {after_result.get('error')}"}
 
    before_age = before_result["estimated_age"]
    after_age  = after_result["estimated_age"]
    diff = after_age - before_age
 
    if diff > 0:
        label = f"Bu işlem görünen yaşı +{diff} artırdı"
    elif diff < 0:
        label = f"Bu işlem görünen yaşı {diff} azalttı"
    else:
        label = "Görünen yaş değişmedi"
 
    return {
        "status": "success",
        "before": before_result,
        "after": after_result,
        "age_difference": diff,
        "difference_label": label,
    }
