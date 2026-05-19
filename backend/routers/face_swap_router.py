"""
Face Swap Router — Complete API surface for static and live face swapping.

Endpoints:
  POST /face-swap/upload-source   Upload & cache the source face
  POST /face-swap/start           Verify engine readiness for live mode
  POST /face-swap/stop            Clear cached source and stop swap
  POST /face-swap                 Static face swap (target upload + cached source)
"""

import io
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

try:
    from modules.face_swap_module import face_swap_engine, FaceSwapError
    from modules.warping_module import detect_face_landmarks
    from modules.frequency_module import encode_image_to_base64
except ModuleNotFoundError:
    from backend.modules.face_swap_module import face_swap_engine, FaceSwapError
    from backend.modules.warping_module import detect_face_landmarks
    from backend.modules.frequency_module import encode_image_to_base64

logger = logging.getLogger(__name__)

router = APIRouter(tags=["face-swap"])


# ═══════════════════════════════════════════════════════════════════════════════
# 1) Upload & Cache Source Face
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/face-swap/upload-source")
async def upload_source(source: UploadFile = File(...)):
    """
    Read the source image, detect landmarks, compute Delaunay triangulation,
    and cache everything on the global FaceSwapEngine for subsequent calls.
    """
    try:
        source_bytes = await source.read()
        face_swap_engine.process_source_image(source_bytes)
        return {
            "status": "success",
            "message": "Source face cached.",
            "triangles": len(face_swap_engine.source_triangles)
                if face_swap_engine.source_triangles is not None else 0,
        }
    except FaceSwapError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Upload source error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ═══════════════════════════════════════════════════════════════════════════════
# 2) Start Live Face Swap (readiness check — NO request body required)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/face-swap/start")
async def start_face_swap():
    """
    Verify the engine has a cached source face ready for live swapping.
    Accepts an optional JSON body (blend_strength, stability, mask_softness)
    but does NOT require one — the endpoint will work with or without a body.
    """
    if not face_swap_engine.is_loaded:
        raise HTTPException(status_code=400, detail="No source face loaded.")

    tri_count = (
        len(face_swap_engine.source_triangles)
        if face_swap_engine.source_triangles is not None
        else 0
    )
    return {
        "status": "ok",
        "message": f"Face swap engine ready ({tri_count} triangles cached).",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3) Stop / Clear Face Swap
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/face-swap/stop")
async def stop_face_swap():
    """Clear the cached source face and reset the engine."""
    face_swap_engine.is_loaded = False
    face_swap_engine.source_image = None
    face_swap_engine.source_landmarks = None
    face_swap_engine.source_triangles = None
    return {"status": "success", "message": "Face swap engine cleared."}


# ═══════════════════════════════════════════════════════════════════════════════
# 4) Static Face Swap  (Target Image + Cached Source → Swapped Result)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/face-swap")
async def static_face_swap(
    target: UploadFile = File(...),
    source: UploadFile = File(None),
):
    """
    Perform a one-shot face swap on a static image.

    Flow:
      1. If a 'source' file is provided, cache it first (convenience path).
         Otherwise, rely on the previously cached source from /upload-source.
      2. Read the target image and detect its face landmarks.
      3. Call FaceSwapEngine.apply_face_swap(target_bgr, target_landmarks).
      4. Return the swapped image as a base64 data-URL (for the frontend)
         along with processing_time_ms.
    """
    t_start = time.perf_counter()

    # ── Optional inline source upload ────────────────────────────────────
    if source is not None:
        try:
            source_bytes = await source.read()
            if source_bytes:
                face_swap_engine.process_source_image(source_bytes)
        except FaceSwapError as e:
            raise HTTPException(status_code=400, detail=f"Source face error: {e}")
        except Exception as e:
            logger.error("Inline source processing failed: %s", e)
            raise HTTPException(status_code=500, detail="Source processing failed.")

    # ── Verify engine readiness ──────────────────────────────────────────
    if not face_swap_engine.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="No source face loaded. Upload a source face first via "
                   "/face-swap/upload-source or include it in this request.",
        )

    # ── Decode the target image ──────────────────────────────────────────
    try:
        target_bytes = await target.read()
        arr = np.frombuffer(target_bytes, np.uint8)
        target_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if target_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid target image: {e}")

    # ── Detect target landmarks ──────────────────────────────────────────
    target_landmarks = detect_face_landmarks(target_bgr)
    if target_landmarks is None or len(target_landmarks) < 100:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the target image.",
        )

    # ── Apply face swap ──────────────────────────────────────────────────
    try:
        swapped = face_swap_engine.apply_face_swap(target_bgr, target_landmarks)
    except Exception as e:
        logger.error("Face swap processing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Face swap failed: {e}")

    t_end = time.perf_counter()
    processing_ms = round((t_end - t_start) * 1000, 2)

    # ── Encode to base64 data URL for the frontend ───────────────────────
    b64_str = encode_image_to_base64(swapped)
    swapped_data_url = f"data:image/png;base64,{b64_str}"

    return {
        "status": "success",
        "swapped_image": swapped_data_url,
        "processing_time_ms": processing_ms,
    }
