import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from modules.frequency_module import encode_image_to_base64
except ModuleNotFoundError:
    from backend.modules.frequency_module import encode_image_to_base64

router = APIRouter(tags=["upload"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/upload")
async def upload_image(image: UploadFile = File(...)) -> dict:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}. Use JPG, PNG or WEBP.",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")

    nparr = np.frombuffer(contents, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded.")

    height, width = decoded.shape[:2]
    return {
        "image_b64": f"data:image/png;base64,{encode_image_to_base64(decoded)}",
        "width": width,
        "height": height,
        "filename": image.filename,
    }
