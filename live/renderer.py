"""
Renderer — OpenCV window management, UI overlay, debug visualization.

Features:
  - HUD overlay (FPS, filter name, intensity)
  - Landmark visualization toggle
  - Face bounding box debug mode
  - Screenshot capture (P key)
  - Fullscreen toggle (F key)
  - Adaptive overlay sizing for different resolutions
  - GPU/CUDA backend detection
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _detect_gpu_backend() -> str:
    """Check if OpenCV was built with CUDA support."""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return f"CUDA ({cv2.cuda.getCudaEnabledDeviceCount()} GPU)"
    except Exception:
        pass
    return "CPU"


class Renderer:
    """
    Handles the OpenCV window, drawing UI overlays, and debug visualizations.

    Features:
      - FPS / filter / intensity HUD
      - Landmark dot rendering
      - Face bounding box debug mode
      - Screenshot capture to screenshots/ directory
      - Fullscreen toggle
    """

    # Colors (BGR)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_BG = (20, 20, 20)
    COLOR_LANDMARK = (0, 255, 100)
    COLOR_BBOX = (255, 100, 0)

    def __init__(self, window_name: str = "Realtime Face Filter"):
        self.window_name = window_name
        self._fullscreen = False
        self._gpu_backend = _detect_gpu_backend()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # Screenshot directory
        self._screenshot_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "screenshots"
        )

        logger.info("Renderer initialized — GPU backend: %s", self._gpu_backend)

    def draw_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        filter_name: str,
        intensity: int,
        show_landmarks: bool = False,
        show_bbox: bool = False,
        landmarks: Optional[np.ndarray] = None,
        frame_time_ms: float = 0.0,
    ) -> np.ndarray:
        """
        Draw debugging information, UI overlay, and optional landmarks.

        Parameters
        ----------
        frame : np.ndarray
            The processed BGR frame.
        fps : float
            Current FPS value.
        filter_name : str
            Name of the active filter.
        intensity : int
            Current filter intensity (0–100).
        show_landmarks : bool
            If True, draw landmark dots.
        show_bbox : bool
            If True, draw face bounding box.
        landmarks : np.ndarray | None
            (N, 2) landmark array.
        frame_time_ms : float
            Per-frame processing time in ms.
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # --- Draw landmarks ---
        if show_landmarks and landmarks is not None:
            for pt in landmarks:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(display, (x, y), 1, self.COLOR_LANDMARK, -1)

        # --- Draw face bounding box ---
        if show_bbox and landmarks is not None and len(landmarks) > 0:
            x_min = int(np.min(landmarks[:, 0]))
            y_min = int(np.min(landmarks[:, 1]))
            x_max = int(np.max(landmarks[:, 0]))
            y_max = int(np.max(landmarks[:, 1]))
            # Clamp to frame bounds
            x_min = max(0, x_min - 10)
            y_min = max(0, y_min - 10)
            x_max = min(w - 1, x_max + 10)
            y_max = min(h - 1, y_max + 10)
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), self.COLOR_BBOX, 2)

        # --- Adaptive HUD sizing ---
        scale = max(0.5, min(1.2, w / 640.0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55 * scale
        thickness = max(1, int(1.5 * scale))
        line_h = int(26 * scale)
        pad = int(8 * scale)

        # --- HUD background ---
        hud_lines = 5
        hud_w = int(260 * scale)
        hud_h = line_h * hud_lines + pad * 2
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        # --- HUD text ---
        y_pos = pad + line_h
        # FPS (green if >20, yellow if 10-20, red if <10)
        fps_color = self.COLOR_GREEN if fps >= 20 else (self.COLOR_YELLOW if fps >= 10 else self.COLOR_RED)
        cv2.putText(display, f"FPS: {fps:.1f}", (pad, y_pos), font, font_scale, fps_color, thickness)
        y_pos += line_h

        cv2.putText(display, f"Filter: {filter_name}", (pad, y_pos), font, font_scale, self.COLOR_CYAN, thickness)
        y_pos += line_h

        cv2.putText(display, f"Intensity: {intensity}%", (pad, y_pos), font, font_scale, self.COLOR_YELLOW, thickness)
        y_pos += line_h

        if frame_time_ms > 0:
            cv2.putText(
                display, f"Frame: {frame_time_ms:.1f}ms",
                (pad, y_pos), font, font_scale * 0.85, self.COLOR_WHITE, max(1, thickness - 1),
            )
        y_pos += line_h

        cv2.putText(
            display, f"Backend: {self._gpu_backend}",
            (pad, y_pos), font, font_scale * 0.75, (150, 150, 150), 1,
        )

        # --- Bottom instruction bar ---
        bar_h = int(28 * scale)
        bar_overlay = display.copy()
        cv2.rectangle(bar_overlay, (0, h - bar_h), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(bar_overlay, 0.75, display, 0.25, 0, display)

        instructions = "1-6: Filters | 7-9: Emoji | 0: Off | W/S: Intensity | L: Landmarks | B: BBox | P: Screenshot | F: Fullscreen"
        cv2.putText(
            display, instructions,
            (pad, h - int(8 * scale)),
            font, 0.38 * scale, self.COLOR_WHITE, 1,
        )

        return display

    def show(self, frame: np.ndarray) -> None:
        """Render the final frame to the OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and windowed mode."""
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL,
            )

    def save_screenshot(self, frame: np.ndarray) -> Optional[str]:
        """
        Save a clean (no overlay) screenshot to the screenshots/ directory.

        Returns the file path on success, or None on failure.
        """
        try:
            os.makedirs(self._screenshot_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = os.path.join(self._screenshot_dir, f"screenshot_{timestamp}.png")
            cv2.imwrite(filepath, frame)
            logger.info("Screenshot saved: %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Failed to save screenshot: %s", e)
            return None

    def close(self) -> None:
        """Destroy the OpenCV window."""
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
