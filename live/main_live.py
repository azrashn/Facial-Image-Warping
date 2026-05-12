"""
main_live.py — Entry point for the Realtime Snapchat-style Facial Filter Engine.

Pipeline:
  webcam (threaded) → frame capture → landmark tracking → temporal smoothing
  → geometric warp → rendering → display

Keyboard Controls:
  1-6  : Individual geometric filters
  7-9  : Emoji expression presets
  0    : No filter
  W/↑  : Increase intensity (+10)
  S/↓  : Decrease intensity (-10)
  L    : Toggle landmark visualization
  B    : Toggle face bounding box
  P    : Capture screenshot (clean, no overlay)
  F    : Toggle fullscreen
  ESC  : Quit

Usage:
  From the project root:
    python -m live.main_live
"""

from __future__ import annotations

import logging
import sys
import time

import cv2

from live.webcam_stream import WebcamStream
from live.realtime_engine import RealtimeEngine
from live.renderer import Renderer
from live.fps_counter import FPSCounter

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_live")

# ── Filter name display map ──
FILTER_DISPLAY_NAMES: dict[str, str] = {
    "none":             "None",
    "smile":            "Smile",
    "eyebrow_raise":    "Eyebrow Raise",
    "lip_widen":        "Lip Widen",
    "face_slim":        "Face Slim",
    "eye_scaling":      "Eye Enlarge",
    "beard":            "Beard",
    "emoji_happy":      "Emoji: Happy",
    "emoji_surprised":  "Emoji: Surprised",
    "emoji_joyful":     "Emoji: Joyful",
}

# ── Key constants (Windows cv2.waitKeyEx codes) ──
KEY_ESC = 27
KEY_UP_ARROW = 2490368
KEY_DOWN_ARROW = 2621440
KEY_UP_ARROW_LINUX = 65362
KEY_DOWN_ARROW_LINUX = 65364


def _print_banner() -> None:
    """Print startup banner with keyboard controls."""
    banner = """
+----------------------------------------------------------+
|        Realtime Facial Filter Engine v2.0                 |
|        Snapchat-style Mesh-Based Geometric Warping        |
+----------------------------------------------------------+
|  Filters:                                                 |
|    1: Smile        2: Eyebrow Raise   3: Lip Widen        |
|    4: Face Slim    5: Eye Enlarge     6: Beard             |
|    7: Emoji Happy  8: Emoji Surprised 9: Emoji Joyful     |
|    0: None (original)                                     |
|                                                           |
|  Controls:                                                |
|    W / UP   : Increase intensity (+10)                    |
|    S / DOWN : Decrease intensity (-10)                    |
|    L        : Toggle landmark visualization               |
|    B        : Toggle face bounding box                    |
|    P        : Capture screenshot                          |
|    F        : Toggle fullscreen                           |
|    ESC      : Quit                                        |
+----------------------------------------------------------+
"""
    print(banner)


def main() -> None:
    """Main entry point for the realtime facial filter engine."""
    _print_banner()

    # ── 1. Initialize components ──
    logger.info("Initializing webcam stream...")
    try:
        stream = WebcamStream(src=0, width=640, height=480).start()
    except Exception as e:
        logger.error("Failed to open webcam: %s", e)
        print(f"\n[ERROR] Cannot open webcam: {e}")
        print("Make sure your camera is connected and not in use by another app.")
        sys.exit(1)

    logger.info("Initializing realtime engine...")
    try:
        engine = RealtimeEngine(alpha=0.7)
    except Exception as e:
        logger.error("Failed to initialize engine: %s", e)
        stream.stop()
        print(f"\n[ERROR] Engine initialization failed: {e}")
        sys.exit(1)

    renderer = Renderer("Facial Filter Engine")
    fps_counter = FPSCounter(history_size=60)

    # ── State ──
    filter_type: str = "none"
    intensity: int = 50
    show_landmarks: bool = False
    show_bbox: bool = False
    _last_cam_warn: float = 0.0

    # Give the camera a moment to warm up
    time.sleep(0.5)
    logger.info("Engine ready -- entering main loop")

    # ── 2. Main processing loop ──
    try:
        while True:
            # Fetch frame from capture thread
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                # Camera might be temporarily unavailable — throttle warning
                now = time.time()
                if not stream.is_opened and (now - _last_cam_warn) > 3.0:
                    logger.warning("Camera disconnected -- waiting for reconnection...")
                    _last_cam_warn = now
                time.sleep(0.03)
                continue

            # Update FPS tracking
            fps_counter.update()

            # Process the frame through the pipeline
            # (landmark detection → smoothing → geometric warp)
            processed_frame, landmarks = engine.process_frame(
                frame, filter_type, intensity
            )

            # Keep a clean copy for screenshot (before overlay)
            clean_frame = processed_frame

            # Render UI overlay and display
            display_name = FILTER_DISPLAY_NAMES.get(filter_type, filter_type)
            display_frame = renderer.draw_overlay(
                frame=processed_frame,
                fps=fps_counter.get_smooth_fps(),
                filter_name=display_name,
                intensity=intensity,
                show_landmarks=show_landmarks,
                show_bbox=show_bbox,
                landmarks=landmarks,
                frame_time_ms=fps_counter.get_frame_time_ms(),
            )

            renderer.show(display_frame)

            # ── 3. Handle keyboard input ──
            key = cv2.waitKeyEx(1)

            if key == KEY_ESC:
                break

            # Filter selection
            elif key == ord("1"):
                filter_type = "smile"
            elif key == ord("2"):
                filter_type = "eyebrow_raise"
            elif key == ord("3"):
                filter_type = "lip_widen"
            elif key == ord("4"):
                filter_type = "face_slim"
            elif key == ord("5"):
                filter_type = "eye_scaling"
            elif key == ord("6"):
                filter_type = "beard"
            elif key == ord("7"):
                filter_type = "emoji_happy"
            elif key == ord("8"):
                filter_type = "emoji_surprised"
            elif key == ord("9"):
                filter_type = "emoji_joyful"
            elif key == ord("0"):
                filter_type = "none"

            # Intensity control
            elif key in (KEY_UP_ARROW, KEY_UP_ARROW_LINUX, ord("w"), ord("W")):
                intensity = min(100, intensity + 10)
                logger.info("Intensity: %d%%", intensity)
            elif key in (KEY_DOWN_ARROW, KEY_DOWN_ARROW_LINUX, ord("s"), ord("S")):
                intensity = max(0, intensity - 10)
                logger.info("Intensity: %d%%", intensity)

            # Toggle features
            elif key in (ord("l"), ord("L")):
                show_landmarks = not show_landmarks
                logger.info("Landmarks: %s", "ON" if show_landmarks else "OFF")
            elif key in (ord("b"), ord("B")):
                show_bbox = not show_bbox
                logger.info("Bounding box: %s", "ON" if show_bbox else "OFF")

            # Screenshot (save the clean frame without overlay)
            elif key in (ord("p"), ord("P")):
                path = renderer.save_screenshot(clean_frame)
                if path:
                    logger.info("Screenshot saved: %s", path)

            # Fullscreen toggle
            elif key in (ord("f"), ord("F")):
                renderer.toggle_fullscreen()

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error("Unexpected error in main loop: %s", e, exc_info=True)
    finally:
        logger.info("Shutting down...")
        stream.stop()
        engine.close()
        renderer.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
