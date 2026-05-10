"""
ThreadedStream — Asynchronous Video Capture & Processing Pipeline.

Provides:
  - ThreadedStream : Dedicated capture thread with a thread-safe single-frame
                     queue (always returns the most recent frame, drops stale).
  - DownsamplePipeline : Run landmark detection and filter geometry on a
                         low-resolution clone; scale masks back up for
                         hi-res compositing.
  - export_frame_to_pdf() : Snapshot hook that saves the current hi-res
                            processed frame to ``backend/assets/export/``.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# GÖREV 1a — ThreadedStream (async capture on a dedicated thread)
# ──────────────────────────────────────────────────────────────────────────────


class ThreadedStream:
    """Thread-safe video capture that always serves the freshest frame.

    Parameters
    ----------
    src : int | str
        Camera index (``0``) or video file path.
    resolution : tuple[int, int] | None
        Optional ``(width, height)`` to request from the capture device.
    """

    def __init__(
        self,
        src: int | str = 0,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        self._cap = cv2.VideoCapture(src)
        if resolution is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")

        # maxsize=1 → only the newest frame is ever stored.
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stopped = threading.Event()
        self._lock = threading.Lock()

        # Pre-read one frame so the queue is non-empty at start
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._q.put(frame)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("ThreadedStream started (src=%s)", src)

    # ── private capture loop ──────────────────────────────────────────────
    def _capture_loop(self) -> None:
        while not self._stopped.is_set():
            with self._lock:
                ok, frame = self._cap.read()
            if not ok or frame is None:
                # Stream ended or camera disconnected
                logger.warning("ThreadedStream: capture read failed — stopping")
                break

            # Latency guard: discard stale frame and push the newest one
            if not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

    # ── public API ────────────────────────────────────────────────────────
    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame (blocking up to 100 ms)."""
        try:
            return self._q.get(timeout=0.1)
        except queue.Empty:
            return None

    def read_nowait(self) -> Optional[np.ndarray]:
        """Return the latest frame immediately, or ``None`` if none ready."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened() and not self._stopped.is_set()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return ``(width, height)`` of the capture device."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            self._cap.release()
        logger.info("ThreadedStream stopped")


# ──────────────────────────────────────────────────────────────────────────────
# GÖREV 1b — Downsampling Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class DownsamplePipeline:
    """Run heavy landmark/filter computation at low-res, composite at hi-res.

    Parameters
    ----------
    process_size : tuple[int, int]
        ``(width, height)`` for the low-res processing clone.
        Recommend ``(480, 360)`` for ~30 FPS on a standard laptop.
    """

    def __init__(self, process_size: Tuple[int, int] = (480, 360)):
        self._pw, self._ph = process_size

    def downscale(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Resize *frame* to processing resolution.

        Returns
        -------
        low_res : np.ndarray
            Down-scaled image.
        sx : float
            Horizontal scale factor (hi / lo).
        sy : float
            Vertical scale factor (hi / lo).
        """
        h, w = frame.shape[:2]
        low = cv2.resize(frame, (self._pw, self._ph), interpolation=cv2.INTER_AREA)
        sx = w / self._pw
        sy = h / self._ph
        return low, sx, sy

    @staticmethod
    def upscale_mask(
        mask: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Scale a single-channel (or 3-ch) mask back to *target_size*.

        Uses bilinear interpolation to keep edges smooth.
        """
        return cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def scale_landmarks(
        landmarks: np.ndarray,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        """Multiply landmark coordinates by the scale factors."""
        out = landmarks.copy()
        out[:, 0] *= sx
        out[:, 1] *= sy
        return out

    def process_frame(
        self,
        hi_res: np.ndarray,
        filter_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Full pipeline: downsample → filter → upscale result.

        For simple filters that return a full image (not a mask), we run
        the filter at low-res and upscale the result back.  For filters that
        produce masks, use ``downscale / upscale_mask`` directly.
        """
        h, w = hi_res.shape[:2]
        low, sx, sy = self.downscale(hi_res)
        filtered_low = filter_fn(low)
        return cv2.resize(filtered_low, (w, h), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# GÖREV 1c — Frame Export Hook
# ──────────────────────────────────────────────────────────────────────────────

# Default export directory – relative to the backend package root
_EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "export")


def export_frame_to_pdf(
    frame: np.ndarray,
    export_dir: Optional[str] = None,
    quality: int = 95,
) -> str:
    """Save *frame* as a high-quality JPEG for the PDF module.

    Parameters
    ----------
    frame : np.ndarray
        Fully processed hi-res BGR frame.
    export_dir : str | None
        Target directory.  Defaults to ``backend/assets/export/``.
    quality : int
        JPEG quality (0-100).

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    dest = export_dir or _EXPORT_DIR
    os.makedirs(dest, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{timestamp}.jpg"
    filepath = os.path.join(dest, filename)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok = cv2.imwrite(filepath, frame, encode_params)
    if not ok:
        raise IOError(f"Failed to write frame to {filepath}")

    logger.info("export_frame_to_pdf → %s", filepath)
    return filepath


# ──────────────────────────────────────────────────────────────────────────────
# GÖREV 1d — FPS Meter (utility for the pipeline loop)
# ──────────────────────────────────────────────────────────────────────────────


class FPSMeter:
    """Exponential moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._last = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self._last, 1e-9)
        instant = 1.0 / dt
        self._fps = self._alpha * instant + (1.0 - self._alpha) * self._fps
        self._last = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps
