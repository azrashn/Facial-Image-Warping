"""Router for quality metric comparison endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/compare")
async def compare_metrics() -> dict:
    """Return mock metric comparison data for DSP and AI pipelines."""
    return {
        "dsp": {"mse": 10, "psnr": 30, "ssim": 0.85},
        "ai": {"mse": 8, "psnr": 32, "ssim": 0.88},
    }
