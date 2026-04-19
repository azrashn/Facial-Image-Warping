from fastapi import APIRouter, UploadFile, File, Form, HTTPException
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


@router.post("/process/age")
async def process_age(
    image: UploadFile = File(...),
    intensity: float = Form(...),
    mode: str = Form(...)
):
    """
    Process image with aging or de-aging filter.

    Expected form-data:
    - image: uploaded file
    - intensity: float (0.0 - 1.0)
    - mode: "age" or "deage"
    """
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

        if not output_saved:
            raise HTTPException(status_code=500, detail="Processed image could not be saved.")

        if not spectrum_saved:
            raise HTTPException(status_code=500, detail="Spectrum image could not be saved.")

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