"""
main.py — FastAPI Gateway for Facial Image Warping & FFT Analysis
==================================================================
Production-ready API server with full validation, error handling,
and CORS support.

Endpoints:
    GET  /              → Health check
    POST /apply_transformation → Main processing pipeline

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import traceback
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from metrics_module import compute_all_metrics

# ── Module imports (will be created in subsequent stages) ──
# These are imported lazily so the app can start even if modules
# are being developed. The endpoint checks availability at runtime.
try:
    from warping_module import detect_landmarks, apply_warping
    WARPING_AVAILABLE = True
except ImportError:
    WARPING_AVAILABLE = False

try:
    from fft_module import apply_fft_filter, compute_magnitude_spectrum
    FFT_AVAILABLE = True
except ImportError:
    FFT_AVAILABLE = False


# ============================================================
# App & CORS Configuration
# ============================================================
app = FastAPI(
    title="Facial Image Warping & FFT Analysis API",
    description="Real-time face warping with Delaunay triangulation, "
                "FFT-based aging/de-aging, and quality metrics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "http://localhost:3000",
        "*",  # Dev fallback — restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Constants
# ============================================================
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}
MAX_FILE_SIZE_MB = 15
MAX_RESOLUTION = 4096  # px per side

VALID_OPERATIONS = {"smile", "thin_face", "eyebrow_raise", "aging", "de-aging"}

# Geometric ops (handled by warping_module)
GEOMETRIC_OPS = {"smile", "thin_face", "eyebrow_raise"}
# Frequency-domain ops (handled by fft_module)
FFT_OPS = {"aging", "de-aging"}


# ============================================================
# Helper Functions
# ============================================================
def _encode_image_to_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """Encodes a BGR numpy image to a base64 data-URI string."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("Failed to encode image to buffer.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _decode_upload_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """Decodes raw file bytes into a BGR numpy array via OpenCV."""
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the uploaded image.")
    return image


def _validate_resolution(image: np.ndarray) -> None:
    """Raises HTTPException if the image exceeds MAX_RESOLUTION."""
    h, w = image.shape[:2]
    if h > MAX_RESOLUTION or w > MAX_RESOLUTION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image resolution {w}x{h} exceeds the maximum "
                f"allowed {MAX_RESOLUTION}x{MAX_RESOLUTION}. "
                "Please resize your image and try again."
            ),
        )


def _clamp_intensity(intensity: int) -> float:
    """Clamps intensity to [0, 100] and normalizes to [0.0, 1.0]."""
    return max(0, min(100, intensity)) / 100.0


# ============================================================
# Routes
# ============================================================
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Facial Image Warping & FFT Analysis API",
        "version": "1.0.0",
        "modules": {
            "warping": WARPING_AVAILABLE,
            "fft": FFT_AVAILABLE,
        },
    }


@app.post("/apply_transformation")
async def apply_transformation(
    file: UploadFile = File(..., description="Image file (JPEG/PNG/WebP/BMP)"),
    operation: str = Form("smile", description="Transformation type"),
    intensity: int = Form(50, description="Intensity 0-100"),
    show_landmarks: Optional[bool] = Form(False, description="Overlay landmarks"),
):
    """
    Main processing pipeline:
        1. Validate & decode uploaded image
        2. Detect face landmarks (MediaPipe 468 points)
        3. Apply geometric warping OR FFT frequency filter
        4. Compute quality metrics (MSE, PSNR, SSIM)
        5. Return JSON with base64 images, landmarks, and metrics
    """

    # ── 1. Content-type validation ──
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ── 2. Read & size check ──
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File size {size_mb:.1f} MB exceeds the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            ),
        )

    # ── 3. Decode image ──
    try:
        original_image = _decode_upload_to_numpy(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _validate_resolution(original_image)

    # ── 4. Normalize operation name ──
    op = operation.strip().lower().replace(" ", "_")
    if op not in VALID_OPERATIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown operation: '{operation}'. "
                f"Valid operations: {', '.join(sorted(VALID_OPERATIONS))}"
            ),
        )

    intensity_normalized = _clamp_intensity(intensity)

    # ── 5. Processing Pipeline ──
    try:
        processed_image = original_image.copy()
        landmarks_list = []
        fft_spectrum_b64 = None

        # --- Landmark detection (needed for geometric ops and visualization) ---
        if WARPING_AVAILABLE:
            landmarks_raw = detect_landmarks(original_image)
            # landmarks_raw: list of (x, y) pixel coordinates (468 points)
            landmarks_list = [
                {"x": int(pt[0]), "y": int(pt[1])} for pt in landmarks_raw
            ]
        else:
            landmarks_raw = []

        # --- Geometric Warping ---
        if op in GEOMETRIC_OPS:
            if not WARPING_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="Warping module is not available. "
                           "Ensure warping_module.py is in the backend directory.",
                )
            if len(landmarks_raw) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No face detected in the uploaded image. "
                           "Please upload a clear frontal face photo.",
                )
            processed_image = apply_warping(
                original_image, landmarks_raw, op, intensity_normalized
            )

        # --- FFT Frequency Processing ---
        elif op in FFT_OPS:
            if not FFT_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="FFT module is not available. "
                           "Ensure fft_module.py is in the backend directory.",
                )
            processed_image = apply_fft_filter(
                original_image, op, intensity_normalized
            )

        # --- FFT Spectrum Visualization (always computed when module is available) ---
        if FFT_AVAILABLE:
            spectrum_img = compute_magnitude_spectrum(original_image)
            fft_spectrum_b64 = _encode_image_to_base64(spectrum_img)
        else:
            # Fallback: generate a basic spectrum without the dedicated module
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            f_transform = np.fft.fft2(gray.astype(np.float64))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
            fft_spectrum_b64 = _encode_image_to_base64(
                cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
            )

        # --- Draw Landmarks Overlay (if requested) ---
        if show_landmarks and len(landmarks_list) > 0:
            overlay = processed_image.copy()
            for pt in landmarks_list:
                cv2.circle(
                    overlay,
                    (pt["x"], pt["y"]),
                    radius=1,
                    color=(0, 255, 0),
                    thickness=-1,
                )
            processed_image = overlay

        # ── 6. Quality Metrics ──
        metrics = compute_all_metrics(original_image, processed_image)

        # ── 7. Encode Response ──
        processed_b64 = _encode_image_to_base64(processed_image)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "processed_image": processed_b64,
                "fft_spectrum": fft_spectrum_b64,
                "landmarks": landmarks_list,
                "metrics": metrics,
                "operation": op,
                "intensity": intensity,
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        # Catch-all for unexpected processing errors
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal processing error: {str(e)}",
        )


# ============================================================
# Entry Point (for direct execution)
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )