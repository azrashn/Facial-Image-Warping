"""
main.py — FastAPI backend for Facial Image Warping
====================================================
DSP Project — Group 14

Endpoints
---------
GET  /                   → health check
POST /apply_transformation
     file      : image file
     operation : smile | eyebrow_raise | lip_widen | face_slim | aging | deaging
     intensity : 0–100 (default 50)
     show_grid : bool (default false) — return deformation grid image too
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from warping import warp_image
from fastapi.staticfiles import StaticFiles
from routers.upload import router as upload_router

app = FastAPI(
    title="Facial Warping API — Group 14",
    
    description=(
        "Geometric image warping using Thin-Plate Spline RBF + "
        "inverse mapping + vectorized bilinear interpolation (pure NumPy)."
    ),
    version="2.0.0",
)

# Include CV & Input pipeline router (must come before the catch-all static mount)
app.include_router(upload_router)

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Allow all origins so the browser frontend can reach the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "message": "Facial Warping API is active.",
        "supported_operations": [
            "smile", "eyebrow_raise", "lip_widen",
            "face_slim", "aging", "deaging",
        ],
    }


@app.post("/apply_transformation")
async def apply_transformation(
    file:      UploadFile = File(...),
    operation: str        = Form("smile"),
    intensity: int        = Form(50),
    show_grid: bool       = Form(False),
):
    """
    Apply a geometric warp to the uploaded image.

    - **file**      : any image format Pillow can read (JPEG, PNG, BMP, …)
    - **operation** : warp type (see supported_operations)
    - **intensity** : strength of the effect, 0–100
    - **show_grid** : if true, also return a deformation-grid PNG
    """
    image_bytes = await file.read()

    result = warp_image(
        image_bytes=image_bytes,
        operation=operation,
        intensity=intensity,
        show_grid=show_grid,
    )

    return {
        "status":          "success",
        "processed_image": result["processed_image"],
        "grid_image":      result["grid_image"],
        "metrics":         result["metrics"],
        "algorithm_info":  result["algorithm_info"],
    }
from fastapi.staticfiles import StaticFiles
# En alta ekle:
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")