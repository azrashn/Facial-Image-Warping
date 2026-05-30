"""
RealtimeEngine — Central processing pipeline for the live filter system.

Pipeline:
  frame → landmark detection (persistent FaceMesh) → temporal smoothing
  → geometric warp → output

Reuses all existing warping algorithms from backend.modules.warping_module:
  - Delaunay triangulation
  - Piecewise affine warping
  - Gaussian landmark falloff
  - Head-pose-aware displacement
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from backend.modules.warping_module import (
    PersistentFaceMesh,
    detect_face_landmarks_live,
    apply_smile,
    apply_eyebrow_raise,
    apply_lip_widen,
    apply_face_slim,
    apply_eye_scaling,
    apply_beard,
    apply_emoji_preset,
)
from backend.modules.face_swap_module import (
    face_swap_engine,
    FaceSwapError,
)
from live.temporal_smoothing import TemporalSmoother

logger = logging.getLogger(__name__)


# Filter dispatch table — maps filter names to (function, arg_type) pairs
# arg_type: "intensity" for int-based filters, "emoji" for emoji preset name,
#           "face_swap" for the realtime face swap (handled specially)
_FILTER_DISPATCH: dict[str, tuple] = {
    "smile":          (apply_smile,         "intensity"),
    "eyebrow_raise":  (apply_eyebrow_raise, "intensity"),
    "lip_widen":      (apply_lip_widen,     "intensity"),
    "face_slim":      (apply_face_slim,     "intensity"),
    "eye_scaling":    (apply_eye_scaling,   "intensity"),
    "beard":          (apply_beard,         "intensity"),
    "emoji_happy":    (apply_emoji_preset,  "emoji"),
    "emoji_surprised":(apply_emoji_preset,  "emoji"),
    "emoji_joyful":   (apply_emoji_preset,  "emoji"),
    "face_swap":      (realtime_face_swap,  "face_swap"),
}


class RealtimeEngine:
    """
    Central pipeline that connects MediaPipe landmark tracking,
    temporal smoothing, and the backend geometric warping operations.

    Thread-safe: the PersistentFaceMesh is created once and reused.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        """
        :param alpha: EMA smoothing factor for temporal landmark stabilization.
        """
        try:
            self.mesh = PersistentFaceMesh()
            logger.info("PersistentFaceMesh initialized (video mode)")
        except Exception as e:
            logger.error("Failed to initialize PersistentFaceMesh: %s", e)
            raise

        self.smoother = TemporalSmoother(alpha=alpha)
        self._last_good_landmarks: Optional[np.ndarray] = None

    # ── Source face management ──────────────────────────────────────────

    def load_source_face(self, source_bgr: np.ndarray) -> bool:
        """
        Load a source face image for face swap mode.

        Parameters
        ----------
        source_bgr : np.ndarray
            BGR image containing a single face.

        Returns
        -------
        bool
            True if source was loaded successfully, False otherwise.
        """
        try:
            ok, encoded = cv2.imencode(".png", source_bgr)
            if not ok:
                return False
            face_swap_engine.process_source_image(encoded.tobytes())
            logger.info("Source face loaded for realtime swap")
            return True
        except FaceSwapError as e:
            logger.warning("Failed to load source face: %s", e)
            return False

    def load_source_face_from_path(self, path: str) -> bool:
        """
        Load source face from a file path.

        Parameters
        ----------
        path : str
            Path to a face image file (JPEG / PNG / WEBP).

        Returns
        -------
        bool
            True if source was loaded successfully, False otherwise.
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not read image from path: %s", path)
            return False
        return self.load_source_face(img)

    def clear_source_face(self) -> None:
        """Unload the source face from the cache."""
        face_swap_engine.is_loaded = False
        face_swap_engine.source_image = None
        face_swap_engine.source_landmarks = None
        face_swap_engine.source_triangles = None
        face_swap_engine.source_triangle_meta = []
        logger.info("Source face cleared")

    @property
    def source_face_loaded(self) -> bool:
        """Whether a source face is currently loaded."""
        return face_swap_engine.is_loaded

    def process_frame(
        self,
        frame: np.ndarray,
        filter_type: str,
        intensity: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full processing pipeline for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR webcam frame.
        filter_type : str
            Name of the filter to apply (e.g. "smile", "emoji_happy",
            "face_swap", "none").
        intensity : int
            Filter intensity (0–100).

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            (processed_frame, smoothed_landmarks)
        """
        if frame is None:
            return frame, None

        # ── 1. Realtime landmark detection (persistent FaceMesh) ──
        try:
            raw_landmarks = detect_face_landmarks_live(self.mesh, frame)
        except Exception as e:
            logger.warning("Landmark detection failed: %s", e)
            raw_landmarks = None

        # ── 2. Temporal smoothing (EMA with confidence gating) ──
        smoothed_landmarks = self.smoother.smooth(raw_landmarks)

        if smoothed_landmarks is not None:
            self._last_good_landmarks = smoothed_landmarks.copy()

        # No face detected — return original frame
        if smoothed_landmarks is None:
            return frame, None

        # ── 3. No filter selected — skip warping ──
        if filter_type == "none" or filter_type not in _FILTER_DISPATCH:
            return frame, smoothed_landmarks

        # ── 4. Apply geometric warp via dispatch table ──
        try:
            func, arg_type = _FILTER_DISPATCH[filter_type]

            if arg_type == "intensity":
                result = func(frame, intensity, landmarks=smoothed_landmarks)
            elif arg_type == "emoji":
                # Extract emoji name from filter_type (e.g. "emoji_happy" → "happy")
                emoji_name = filter_type.split("_", 1)[1] if "_" in filter_type else filter_type
                result = func(frame, emoji_name, landmarks=smoothed_landmarks)
            elif arg_type == "face_swap":
                # Realtime face swap — uses cached source + smoothed target landmarks
                if not face_swap_engine.is_loaded:
                    logger.debug("Face swap selected but no source loaded — skipping")
                    result = frame
                else:
                    result = face_swap_engine.apply_face_swap(frame, smoothed_landmarks)
            else:
                result = frame

        except Exception as e:
            logger.error("Filter '%s' failed: %s — returning original frame", filter_type, e)
            result = frame

        return result, smoothed_landmarks

    def close(self) -> None:
        """Clean up the underlying MediaPipe models."""
        try:
            self.mesh.close()
            logger.info("PersistentFaceMesh closed")
        except Exception as e:
            logger.error("Error closing PersistentFaceMesh: %s", e)
