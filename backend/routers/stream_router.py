"""
Stream Router — WebSocket-based live video pipeline.

Endpoints
---------
  WS  /stream/ws?filter=<name>   Live processed frames via WebSocket
  POST /stream/export             Export the latest processed frame as JPEG
  GET  /stream/status              Pipeline health check
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["stream"])

# ── Module imports (dual-path for test / prod) ────────────────────────────────
try:
    from modules.stream_module import (
        DownsamplePipeline,
        FPSMeter,
        ThreadedStream,
        export_frame_to_pdf,
    )
    from modules.warping_module import detect_face_landmarks
except ModuleNotFoundError:
    from backend.modules.stream_module import (
        DownsamplePipeline,
        FPSMeter,
        ThreadedStream,
        export_frame_to_pdf,
    )
    from backend.modules.warping_module import detect_face_landmarks


# ── Shared state ──────────────────────────────────────────────────────────────
_active_stream: Optional[ThreadedStream] = None
_latest_processed_frame: Optional[np.ndarray] = None
_latest_raw_frame: Optional[np.ndarray] = None


def _get_or_create_stream(src: int = 0) -> ThreadedStream:
    """Lazily start the threaded capture."""
    global _active_stream
    if _active_stream is None or not _active_stream.is_opened:
        _active_stream = ThreadedStream(src=src)
    return _active_stream


# ── Filter registry (maps name → callable) ───────────────────────────────────
# We import lazily inside the function to avoid circular imports and to keep
# the module lightweight.

def _get_filter_fn(name: str):
    """Return a callable  f(image) → image  for the given filter name."""
    # Import from process.py where the preset functions live
    try:
        from routers.process import (
            _apply_alien,
            _apply_robot,
            _apply_clown,
            _apply_star_eyes,
            _apply_heart_eyes,
            _apply_crying,
        )
    except ModuleNotFoundError:
        from backend.routers.process import (
            _apply_alien,
            _apply_robot,
            _apply_clown,
            _apply_star_eyes,
            _apply_heart_eyes,
            _apply_crying,
        )

    registry = {
        "alien": _apply_alien,
        "robot": _apply_robot,
        "clown": _apply_clown,
        "star_eyes": _apply_star_eyes,
        "heart_eyes": _apply_heart_eyes,
        "crying": _apply_crying,
        "none": lambda img: img,
    }
    return registry.get(name, lambda img: img)


# ── WebSocket: live stream ────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_stream(ws: WebSocket, filter: str = "none", src: int = 0):
    """Stream processed frames over WebSocket as base64-encoded JPEGs.

    Query parameters
    ----------------
    filter : str
        Filter name (``alien``, ``robot``, ``clown``, ``star_eyes``,
        ``heart_eyes``, ``crying``, ``none``).
    src : int
        Camera index (default ``0``).
    """
    global _latest_processed_frame, _latest_raw_frame

    await ws.accept()
    logger.info("WebSocket stream started (filter=%s, src=%s)", filter, src)

    stream = _get_or_create_stream(src)
    pipeline = DownsamplePipeline(process_size=(480, 360))
    meter = FPSMeter()
    filter_fn = _get_filter_fn(filter)

    try:
        while True:
            frame = stream.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            _latest_raw_frame = frame.copy()

            # Check for filter change messages from client
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                if msg.startswith("filter:"):
                    new_filter = msg.split(":", 1)[1].strip().lower()
                    filter_fn = _get_filter_fn(new_filter)
                    logger.info("Stream filter changed to: %s", new_filter)
                elif msg == "export":
                    if _latest_processed_frame is not None:
                        path = export_frame_to_pdf(_latest_processed_frame)
                        await ws.send_text(f"exported:{path}")
                    continue
            except (asyncio.TimeoutError, Exception):
                pass

            # Process frame with downsampling pipeline
            processed = pipeline.process_frame(frame, filter_fn)
            _latest_processed_frame = processed.copy()

            fps = meter.tick()

            # Encode to JPEG and send as base64
            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode("ascii")
            await ws.send_text(f"data:image/jpeg;base64,{b64}")

            # Yield to the event loop — target ~30 FPS
            await asyncio.sleep(max(0, (1.0 / 30.0) - (1.0 / max(fps, 1))))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket stream error: %s", exc)
    finally:
        logger.info("WebSocket stream ended")


# ── REST: export latest frame ─────────────────────────────────────────────────

class ExportResponse(BaseModel):
    status: str
    filepath: str


@router.post("/export")
async def stream_export() -> ExportResponse:
    """Export the latest fully-processed hi-res frame as a JPEG.

    Saved to ``backend/assets/export/`` for the PDF module.
    """
    global _latest_processed_frame
    if _latest_processed_frame is None:
        raise HTTPException(status_code=404, detail="No processed frame available yet.")

    try:
        path = export_frame_to_pdf(_latest_processed_frame)
        return ExportResponse(status="success", filepath=path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── REST: pipeline status ─────────────────────────────────────────────────────

@router.get("/status")
async def stream_status():
    """Check whether the capture pipeline is active."""
    return {
        "active": _active_stream is not None and _active_stream.is_opened,
        "has_processed_frame": _latest_processed_frame is not None,
        "frame_size": _active_stream.frame_size if _active_stream and _active_stream.is_opened else None,
    }


# ── REST: stop pipeline ──────────────────────────────────────────────────────

@router.post("/stop")
async def stream_stop():
    """Gracefully stop the capture thread."""
    global _active_stream, _latest_processed_frame, _latest_raw_frame
    if _active_stream is not None:
        _active_stream.stop()
        _active_stream = None
    _latest_processed_frame = None
    _latest_raw_frame = None
    return {"status": "stopped"}
