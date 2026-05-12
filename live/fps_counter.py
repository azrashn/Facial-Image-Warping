"""
FPS counter with dual-mode tracking:
  - Windowed average (accurate over last N frames)
  - Exponential moving average (smooth display value)
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    Tracks the Frames Per Second (FPS) of the realtime pipeline.

    Uses a sliding window for accurate average and an EMA for smooth
    display rendering (avoids rapid FPS flickering in the UI overlay).
    """

    def __init__(self, history_size: int = 60, ema_alpha: float = 0.1):
        """
        :param history_size: Number of frame deltas to keep for windowed average.
        :param ema_alpha: Smoothing factor for the EMA display value.
        """
        self.history_size = history_size
        self.times: deque[float] = deque(maxlen=history_size)
        self.last_time: float = time.perf_counter()
        self._ema_fps: float = 0.0
        self._ema_alpha: float = ema_alpha
        self._frame_count: int = 0

    def update(self) -> None:
        """Update the timer with a new frame timestamp."""
        now = time.perf_counter()
        dt = now - self.last_time
        self.times.append(dt)
        self.last_time = now
        self._frame_count += 1

        # Update EMA
        if dt > 0:
            instant_fps = 1.0 / dt
            if self._ema_fps == 0.0:
                self._ema_fps = instant_fps
            else:
                self._ema_fps = (
                    self._ema_alpha * instant_fps
                    + (1.0 - self._ema_alpha) * self._ema_fps
                )

    def get_fps(self) -> float:
        """Calculate the windowed average FPS over the tracked history."""
        if not self.times:
            return 0.0
        avg_frame_time = sum(self.times) / len(self.times)
        if avg_frame_time <= 0:
            return 0.0
        return 1.0 / avg_frame_time

    def get_smooth_fps(self) -> float:
        """Return the EMA-smoothed FPS value (ideal for UI display)."""
        return self._ema_fps

    @property
    def frame_count(self) -> int:
        """Total number of frames processed since creation."""
        return self._frame_count

    def get_frame_time_ms(self) -> float:
        """Return the last frame's processing time in milliseconds."""
        if self.times:
            return self.times[-1] * 1000.0
        return 0.0
