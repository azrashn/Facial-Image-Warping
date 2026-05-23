"""
Temporal landmark smoothing via Exponential Moving Average (EMA).

Reduces inter-frame jitter while preserving responsiveness to fast head
movements.  Includes confidence gating to reject sudden unstable jumps.

Also provides a separate PoseSmoother for head-pose angle stabilization.
"""

import time
import numpy as np
from typing import Optional, Tuple

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
    # Hold last-good landmarks for up to this many frames during brief loss
    HOLD_FRAMES: int = 45  # ~1.5s at 30fps — increased from 30

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
            # Return previous landmarks briefly to avoid flicker
            if self.prev_landmarks is not None and self._face_lost_count < self.HOLD_FRAMES:
                return self.prev_landmarks.copy()
            # After prolonged loss, reset state
            if self._face_lost_count >= self.HOLD_FRAMES:
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


class PoseSmoother:
    """Separate EMA smoother for head-pose angles (yaw, pitch, roll).

    Uses a slower alpha than the landmark smoother to provide extra
    stability for the pose-gated blend factor display.
    """

    def __init__(self, alpha: float = 0.4):
        """
        :param alpha: Smoothing factor for pose angles.
                      Lower = more stable but laggier.
        """
        self.alpha = max(0.05, min(1.0, alpha))
        self._prev: Optional[Tuple[float, float, float]] = None
        self._prev_time: float = 0.0
        self._stale_timeout: float = 1.5

    def smooth(
        self,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> Tuple[float, float, float]:
        """Smooth the incoming pose angles.

        Parameters
        ----------
        yaw, pitch, roll : float
            Raw angles from solvePnP (degrees).

        Returns
        -------
        tuple[float, float, float]
            Smoothed (yaw, pitch, roll).
        """
        now = time.perf_counter()

        if self._prev is None or (now - self._prev_time) > self._stale_timeout:
            self._prev = (yaw, pitch, roll)
            self._prev_time = now
            return self._prev

        a = self.alpha
        smoothed = (
            a * yaw + (1.0 - a) * self._prev[0],
            a * pitch + (1.0 - a) * self._prev[1],
            a * roll + (1.0 - a) * self._prev[2],
        )
        self._prev = smoothed
        self._prev_time = now
        return smoothed

    def reset(self) -> None:
        """Force-reset pose smoothing state."""
        self._prev = None
        self._prev_time = 0.0
