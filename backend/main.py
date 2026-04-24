from pathlib import Path
import sys

# Kök dizindeki warping_module için import yolu
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64

import cv2
import numpy as np

import warping_module as wm

app = FastAPI(title="Facial Warping API - Group 14")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Backend API Sistemimiz Aktif!"}


@app.post("/apply_transformation")
async def apply_transformation(
    file: UploadFile = File(...),
    operation: str = Form("Smile"),
    intensity: int = Form(50),
):
    image_data = await file.read()

    arr = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = None

    processed_bgr = img
    if img is not None:
        op = (operation or "").strip().lower()
        if op == "smile":
            processed_bgr = wm.apply_smile(img, intensity)
        elif op == "eyebrow":
            processed_bgr = wm.apply_eyebrow_raise(img, intensity)
        elif op == "lip":
            processed_bgr = wm.apply_lip_widen(img, intensity)
        elif op == "slim":
            processed_bgr = wm.apply_face_slim(img, intensity)
        elif op in ("aging", "deaging", "fft"):
            pass  # Rol 3 burayı dolduracak
        else:
            processed_bgr = img

    dummy_metrics = {
        "mse": 14.57,
        "psnr": 27.74,
        "ssim": 0.885,
    }

    if processed_bgr is None:
        base64_encoded = base64.b64encode(image_data).decode("utf-8")
    else:
        ok, buf = cv2.imencode(".jpg", processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            base64_encoded = base64.b64encode(image_data).decode("utf-8")
        else:
            base64_encoded = base64.b64encode(buf.tobytes()).decode("utf-8")

    image_url = f"data:image/jpeg;base64,{base64_encoded}"

    return {
        "status": "success",
        "processed_image": image_url,
        "metrics": dummy_metrics,
    }
