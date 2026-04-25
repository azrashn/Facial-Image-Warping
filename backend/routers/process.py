from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from modules.ai_module import AIProcessor
import numpy as np
import cv2
import os
import time

from modules.frequency_module import (
    compute_fft,
    compute_magnitude_spectrum,
    apply_aging_filter,
    apply_deaging_filter,
    compute_energy_analysis,
    encode_image_to_base64,
)

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- ROL 3: Sinyal İşleme (Aging/Deaging) ---
@router.post("/process/age")
async def process_age(
    image: UploadFile = File(...),
    intensity: float = Form(...),
    mode: str = Form(...)
):
    try:
        if mode not in ["age", "deage"]:
            raise HTTPException(status_code=400, detail="Mode must be 'age' or 'deage'.")

        if not (0.0 <= intensity <= 1.0):
            raise HTTPException(status_code=400, detail="Intensity must be between 0.0 and 1.0.")

        contents = await image.read()
        file_bytes = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        _, _, fft_shifted = compute_fft(img)
        spectrum = compute_magnitude_spectrum(fft_shifted)

        if mode == "age":
            processed = apply_aging_filter(img, intensity=intensity)
        else:
            processed = apply_deaging_filter(img, intensity=intensity)

        radius = int(10 + intensity * 40)
        energy = compute_energy_analysis(img, radius=radius)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp = int(time.time())
        output_filename = f"output_{mode}_{timestamp}.png"
        spectrum_filename = f"spectrum_{mode}_{timestamp}.png"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        spectrum_path = os.path.join(OUTPUT_DIR, spectrum_filename)

        output_saved = cv2.imwrite(output_path, processed)
        spectrum_saved = cv2.imwrite(spectrum_path, spectrum)

        if not output_saved or not spectrum_saved:
            raise HTTPException(status_code=500, detail="Image or spectrum could not be saved.")

        return {
            "message": "Image processed successfully.",
            "image_b64": encode_image_to_base64(processed),
            "spectrum_b64": encode_image_to_base64(spectrum),
            "energy": energy,
            "saved_output_path": output_path,
            "saved_spectrum_path": spectrum_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ROL 4: AI Spesiyali (Yaş Tahmini) ---
ai_processor = AIProcessor()
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

@router.post("/process/ai-age")
async def process_ai_age(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPG, PNG or WEBP."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded.")

    # ai_module.py içindeki analiz fonksiyonunu çağırır [cite: 25]
    analysis = ai_processor.analyze_age(img)

    if analysis.get("status") != "success":
        raise HTTPException(
            status_code=500,
            detail=analysis.get("error", "Unknown AI analysis error.")
        )

    return {
        "estimated_age": analysis["estimated_age"],
        "image_b64": analysis["image_b64"],
        "image_size": analysis["image_size"],
        "status": "success"
    }
