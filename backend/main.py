import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from routers.ai_router import router as ai_router

try:
    from modules.frequency_module import (
        compute_energy_analysis,
        compute_fft,
        compute_magnitude_spectrum,
        encode_image_to_base64,
    )
    from modules.metrics_module import compute_mse, compute_psnr, compute_ssim
    from modules.warping_module import (
        apply_beard,
        apply_emoji_preset,
        apply_eye_scaling,
        apply_eyebrow_raise,
        apply_face_slim,
        apply_lip_widen,
        apply_smile,
    )
    from routers.export import router as export_router
    from routers.metrics import router as metrics_router
    from routers.process import router as process_router
    from routers.upload import router as upload_router
except ModuleNotFoundError:
    from modules.frequency_module import (
        compute_energy_analysis,
        compute_fft,
        compute_magnitude_spectrum,
        encode_image_to_base64,
    )
    from modules.metrics_module import compute_mse, compute_psnr, compute_ssim
    from modules.warping_module import (
        apply_beard,
        apply_emoji_preset,
        apply_eye_scaling,
        apply_eyebrow_raise,
        apply_face_slim,
        apply_lip_widen,
        apply_smile,
    )
    from routers.export import router as export_router
    from routers.metrics import router as metrics_router
    from routers.process import router as process_router
    from routers.upload import router as upload_router


app = FastAPI(title="Facial Warping API - Group 14")
WARP_OPS = {"smile", "eyebrow", "lip", "slim", "eye_scale", "emoji_preset", "beard"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

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


def _decode_upload(contents: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _data_url_from_image(image: np.ndarray) -> str:
    return f"data:image/png;base64,{encode_image_to_base64(image)}"


def _metrics_dict(original: np.ndarray, processed: np.ndarray) -> dict:
    return {
        "mse": float(compute_mse(original, processed)["mse"]),
        "psnr": float(compute_psnr(original, processed)["psnr"]),
        "ssim": float(compute_ssim(original, processed)["ssim"]),
    }


def _response_payload(
    image_b64: str,
    metrics: dict,
    orig_spectrum_b64: str | None = None,
    proc_spectrum_b64: str | None = None,
    energy: dict | None = None,
) -> dict:
    return {
        "image_b64": image_b64,
        "metrics": metrics,
        "orig_spectrum_b64": orig_spectrum_b64,
        "proc_spectrum_b64": proc_spectrum_b64,
        "energy": energy,
    }


@app.post("/process/warp")
async def process_warp_extended(
    image: UploadFile = File(...),
    operation: str = Form(...),
    intensity: float = Form(50),
    smoothing: float = Form(30),
    emoji_name: str = Form("neutral"),
):
    op = (operation or "").strip().lower()
    if op not in WARP_OPS:
        raise HTTPException(status_code=400, detail="Invalid operation for /process/warp.")

    try:
        contents = await image.read()
        original = _decode_upload(contents)

        if op == "smile":
            processed = apply_smile(original, intensity)
        elif op == "eyebrow":
            processed = apply_eyebrow_raise(original, intensity)
        elif op == "lip":
            processed = apply_lip_widen(original, intensity)
        elif op == "slim":
            processed = apply_face_slim(original, intensity)
        elif op == "eye_scale":
            processed = apply_eye_scaling(original, intensity)
        elif op == "emoji_preset":
            processed = apply_emoji_preset(original, emoji_name)
        else:
            processed = apply_beard(original, intensity)

        smooth_strength = max(0.0, min(1.0, float(smoothing) / 100.0))
        if smooth_strength > 0:
            smoothed = cv2.GaussianBlur(processed, (0, 0), 0.5 + smooth_strength * 2.0)
            processed = cv2.addWeighted(
                processed,
                1.0 - smooth_strength * 0.4,
                smoothed,
                smooth_strength * 0.4,
                0,
            )

        metrics = _metrics_dict(original, processed)

        orig_spectrum = compute_magnitude_spectrum(compute_fft(original)[2])
        proc_spectrum = compute_magnitude_spectrum(compute_fft(processed)[2])

        orig_spectrum_b64 = _data_url_from_image(cv2.cvtColor(orig_spectrum, cv2.COLOR_GRAY2BGR))
        proc_spectrum_b64 = _data_url_from_image(cv2.cvtColor(proc_spectrum, cv2.COLOR_GRAY2BGR))

        energy = compute_energy_analysis(
            processed,
            radius=int(10 + max(0.0, min(1.0, float(intensity) / 100.0)) * 40),
        )

        return _response_payload(
            image_b64=_data_url_from_image(processed),
            metrics=metrics,
            orig_spectrum_b64=orig_spectrum_b64,
            proc_spectrum_b64=proc_spectrum_b64,
            energy=energy,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(upload_router)
app.include_router(process_router)
app.include_router(metrics_router)
app.include_router(export_router)
app.include_router(ai_router)
