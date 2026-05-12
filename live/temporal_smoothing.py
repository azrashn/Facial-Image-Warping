"""
Temporal landmark smoothing via Exponential Moving Average (EMA).

Reduces inter-frame jitter while preserving responsiveness to fast head
movements.  Includes confidence gating to reject sudden unstable jumps.
"""

import time
import numpy as np
from typing import Optional

DEFAULT_ALPHA: float = 0.7


class TemporalSmoother:
    """
    Applies Exponential Moving Average (EMA) to smooth out
    facial landmarks between frames, reducing jittering.

    Features:
      - Configurable alpha (blending weight)
      - Adaptive alpha: reduces smoothing during fast head motion
      - Confidence gating: rejects sudden large jumps (unstable detection)
      - Stale-state reset: re-initializes after prolonged face loss
    """

    # If mean landmark motion exceeds this fraction of face scale, reject frame
    JUMP_THRESHOLD_RATIO: float = 0.35
    # If no face detected for this many seconds, reset state
    STALE_TIMEOUT: float = 1.0
    # Adaptive alpha range during fast motion
    FAST_MOTION_ALPHA: float = 0.9

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        """
        :param alpha: Smoothing factor. Higher = more responsive (less smooth).
                      1.0 means no smoothing, 0.0 means no update.
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.prev_landmarks: Optional[np.ndarray] = None
        self._prev_time: float = 0.0
        self._face_lost_count: int = 0

    def smooth(self, current_landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply EMA smoothing to the provided landmarks.

        Parameters
        ----------
        current_landmarks : np.ndarray | None
            Raw (N, 2) float32 landmarks from the detector, or None if no face.

        Returns
        -------
        np.ndarray | None
            Smoothed landmarks, or None if no face is detected and state is stale.
        """
        now = time.perf_counter()

        if current_landmarks is None:
            self._face_lost_count += 1
            # Return previous landmarks briefly to avoid flicker (up to ~30 frames)
            if self.prev_landmarks is not None and self._face_lost_count < 30:
                return self.prev_landmarks.copy()
            # After prolonged loss, reset state
            if self._face_lost_count >= 30:
                self.prev_landmarks = None
            return None

        # Face detected — reset lost counter
        self._face_lost_count = 0

        # First frame or shape mismatch or stale state
        stale = (
            self.prev_landmarks is None
            or self.prev_landmarks.shape != current_landmarks.shape
            or (now - self._prev_time) > self.STALE_TIMEOUT
        )

        if stale:
            self.prev_landmarks = current_landmarks.copy()
            self._prev_time = now
            return self.prev_landmarks.copy()

        # --- Confidence gating: reject large sudden jumps ---
        mean_motion = float(np.mean(
            np.linalg.norm(current_landmarks - self.prev_landmarks, axis=1)
        ))
        # Estimate face scale from inter-eye distance (landmarks 133, 362)
        face_scale = 1.0
        if current_landmarks.shape[0] > 362:
            face_scale = max(
                float(np.linalg.norm(
                    current_landmarks[133] - current_landmarks[362]
                )),
                10.0,
            )

        max_allowed_jump = face_scale * self.JUMP_THRESHOLD_RATIO
        if mean_motion > max_allowed_jump:
            # Likely a detection glitch — keep previous stable landmarks
            self._prev_time = now
            return self.prev_landmarks.copy()

        # --- Adaptive alpha: more responsive during fast (but valid) motion ---
        motion_ratio = mean_motion / max(face_scale * 0.1, 1e-6)
        adaptive_alpha = self.alpha
        if motion_ratio > 0.5:
            # Fast motion — bias toward current frame
            adaptive_alpha = min(self.FAST_MOTION_ALPHA, self.alpha + 0.15)

        # Compute Exponential Moving Average
        smoothed = adaptive_alpha * current_landmarks + (1.0 - adaptive_alpha) * self.prev_landmarks

        # Save state for next frame
        self.prev_landmarks = smoothed.copy()
        self._prev_time = now

        return smoothed

    def reset(self) -> None:
        """Force-reset all internal state."""
        self.prev_landmarks = None
        self._prev_time = 0.0
        self._face_lost_count = 0
