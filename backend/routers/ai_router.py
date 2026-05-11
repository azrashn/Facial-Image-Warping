import logging
import cv2
import numpy as np
import base64
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

try:
    from modules.ai_module import estimate_age, estimate_age_before_after
    from modules.hair_module import apply_hair_color, apply_hair_length
    from modules.beard_module import apply_beard, apply_mustache
    from modules.warping_module import detect_face_landmarks, apply_emoji_preset
    from modules.frequency_module import encode_image_to_base64
    from modules.metrics_module import compute_mse, compute_psnr, compute_ssim
except ModuleNotFoundError:
    from backend.modules.ai_module import estimate_age, estimate_age_before_after
    from backend.modules.hair_module import apply_hair_color, apply_hair_length
    from backend.modules.beard_module import apply_beard, apply_mustache
    from backend.modules.warping_module import detect_face_landmarks, apply_emoji_preset
    from backend.modules.frequency_module import encode_image_to_base64
    from backend.modules.metrics_module import compute_mse, compute_psnr, compute_ssim

router = APIRouter()
logger = logging.getLogger("facial_pipeline.ai_router")

# JSON modeli eksikti, eklendi
class EmojiRequest(BaseModel):
    image_b64: str
    preset_name: str
    description: Optional[str] = None

def _decode_upload(contents: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img

def _decode_b64_string(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    file_bytes = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")
    return img

def _data_url(image_bgr: np.ndarray) -> str:
    return f"data:image/png;base64,{encode_image_to_base64(image_bgr)}"

def _metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    return {
        "mse":  float(compute_mse(original, processed)["mse"]),
        "psnr": float(compute_psnr(original, processed)["psnr"]),
        "ssim": float(compute_ssim(original, processed)["ssim"]),
    }

# 1. Before/After Yaş Karşılaştırması
@router.post("/process/estimate-age-compare")
async def estimate_age_compare(before_image: UploadFile = File(...), after_image: UploadFile = File(...)):
    logger.info("estimate_age_compare.received")
    try:
        b_bytes = await before_image.read()
        a_bytes = await after_image.read()
        result = estimate_age_before_after(_decode_upload(b_bytes), _decode_upload(a_bytes))
        if result.get("status") != "success": raise HTTPException(status_code=422, detail=result.get("error"))
        return result
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# 2. Tek görüntü yaş tahmini
@router.post("/process/estimate-age")
async def estimate_age_single(image: UploadFile = File(...)):
    logger.info("estimate_age_single.received")
    try:
        contents = await image.read()
        result = estimate_age(_decode_upload(contents))
        if result.get("status") != "success": raise HTTPException(status_code=422, detail=result.get("error"))
        return result
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# 3. Saç Rengi Değiştirme
@router.post("/process/hair-color")
async def process_hair_color(image: UploadFile = File(...), target_color: str = Form("255,0,0"), intensity: float = Form(0.6)):
    logger.info("process_hair_color.received: color=%s intensity=%.2f", target_color, intensity)
    try:
        contents = await image.read()
        original = _decode_upload(contents)
        processed = apply_hair_color(original, target_color, intensity)
        metrics = _metrics(original, processed)
        return {"image_b64": _data_url(processed), "metrics": metrics, "orig_spectrum_b64": None, "proc_spectrum_b64": None, "energy": None}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# 4. Saç Uzunluğu Simülasyonu
@router.post("/process/hair-length")
async def process_hair_length(image: UploadFile = File(...), length_delta: int = Form(20)):
    logger.info("process_hair_length.received: delta=%d", length_delta)
    try:
        contents = await image.read()
        original = _decode_upload(contents)
        processed = apply_hair_length(original, length_delta)
        metrics = _metrics(original, processed)
        return {"image_b64": _data_url(processed), "metrics": metrics, "orig_spectrum_b64": None, "proc_spectrum_b64": None, "energy": None}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# 5. Sakal Ekleme
@router.post("/process/beard")
async def process_beard(image: UploadFile = File(...), intensity: int = Form(70), darkness: float = Form(0.6), style: str = Form("beard")):
    logger.info("process_beard.received: style=%s intensity=%d", style, intensity)
    try:
        contents = await image.read()
        original = _decode_upload(contents)
        landmarks = detect_face_landmarks(original)
        if landmarks is None: raise HTTPException(status_code=422, detail="Yüz landmark'ları tespit edilemedi.")
        if style.lower() == "mustache": processed = apply_mustache(original, landmarks, intensity, darkness)
        else: processed = apply_beard(original, landmarks, intensity, darkness)
        metrics = _metrics(original, processed)
        return {"image_b64": _data_url(processed), "metrics": metrics, "orig_spectrum_b64": None, "proc_spectrum_b64": None, "energy": None}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# 6. EMOJİ PRESET (Arayüz JSON Gönderdiği İçin Gerekliydi Ama Sendeki Kodda Yoktu!)
@router.post("/process/emoji-preset")
async def process_emoji_preset(req: EmojiRequest):
    logger.info(f"process_emoji_preset.received: {req.preset_name}")
    try:
        original = _decode_b64_string(req.image_b64)
        processed = apply_emoji_preset(original, req.preset_name)
        return {"image_b64": _data_url(processed), "metrics": _metrics(original, processed)}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))

# 7. PALYAÇO / JOKER EMOJİSİ (Arkadaşlarının Eski Kodu Yerine Bizim V3'e Giden Gerçek Köprü)
@router.post("/process/clown_transformation")
async def process_clown_transformation(image: UploadFile = File(...), intensity: Optional[float] = Form(100.0)):
    logger.info("process_clown_transformation.received")
    try:
        contents = await image.read()
        original = _decode_upload(contents)
        processed = apply_emoji_preset(original, "clown") 
        return {"image_b64": _data_url(processed), "metrics": _metrics(original, processed)}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))
