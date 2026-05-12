"""
Threaded webcam capture with automatic reconnection and disconnect recovery.

Handles:
  - Non-blocking frame capture in a dedicated daemon thread
  - Automatic camera reconnection on disconnect
  - Thread-safe frame sharing via lock
  - Graceful shutdown
"""

import cv2
import threading
import time
import logging
import platform
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WebcamStream:
    """
    Handles webcam capture in a separate thread to prevent blocking
    the main processing and rendering loop.

    Features:
      - Dedicated capture thread (daemon)
      - Automatic reconnection on camera disconnect
      - Always serves the freshest frame
      - Thread-safe read via lock
    """

    # Maximum reconnection attempts before giving up
    MAX_RECONNECT_ATTEMPTS: int = 10
    # Delay between reconnection attempts (seconds)
    RECONNECT_DELAY: float = 1.0

    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height

        # On Windows, DSHOW backend is more reliable for USB cameras
        if platform.system() == "Windows" and isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)
        self._configure_capture()

        self.grabbed: bool = False
        self.frame: Optional[np.ndarray] = None
        self.started: bool = False
        self.read_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self._consecutive_failures: int = 0

        # Initial frame acquisition with retry (cameras need warm-up)
        for attempt in range(10):
            if self.cap.isOpened():
                self.grabbed, self.frame = self.cap.read()
                if self.grabbed and self.frame is not None:
                    logger.info("Initial frame acquired on attempt %d", attempt + 1)
                    break
                time.sleep(0.1)
        else:
            logger.warning("Could not acquire initial frame after 10 attempts")

    def _configure_capture(self) -> None:
        """Apply capture device settings."""
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Request MJPEG for faster USB camera throughput
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            logger.info(
                "WebcamStream configured: src=%s, %dx%d",
                self.src, self.width, self.height,
            )

    def start(self) -> "WebcamStream":
        """Start the video capture thread."""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update_loop, args=(), daemon=True)
        self.thread.start()
        return self

    def _update_loop(self) -> None:
        """Continuously read frames from the webcam with reconnection logic."""
        while self.started:
            if not self.cap.isOpened():
                self._attempt_reconnect()
                continue

            grabbed, frame = self.cap.read()

            if not grabbed or frame is None:
                self._consecutive_failures += 1
                if self._consecutive_failures > 30:
                    logger.warning(
                        "WebcamStream: %d consecutive read failures — attempting reconnect",
                        self._consecutive_failures,
                    )
                    self._attempt_reconnect()
                else:
                    time.sleep(0.005)
                continue

            # Successful read — reset failure counter
            self._consecutive_failures = 0
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def _attempt_reconnect(self) -> None:
        """Try to reconnect to the camera source."""
        self._consecutive_failures = 0
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            if not self.started:
                return
            logger.info(
                "WebcamStream: reconnect attempt %d/%d ...",
                attempt, self.MAX_RECONNECT_ATTEMPTS,
            )
            try:
                self.cap.release()
            except Exception:
                pass

            time.sleep(self.RECONNECT_DELAY)
            self.cap = cv2.VideoCapture(self.src)
            self._configure_capture()

            if self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    with self.read_lock:
                        self.grabbed = grabbed
                        self.frame = frame
                    logger.info("WebcamStream: reconnected on attempt %d", attempt)
                    return

        logger.error(
            "WebcamStream: failed to reconnect after %d attempts",
            self.MAX_RECONNECT_ATTEMPTS,
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently read frame (thread-safe copy)."""
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            else:
                return False, None

    @property
    def is_opened(self) -> bool:
        """Check if the capture device is open AND has delivered at least one frame."""
        with self.read_lock:
            return self.cap.isOpened() and self.grabbed

    def stop(self) -> None:
        """Stop the video capture thread and release the camera."""
        self.started = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        try:
            self.cap.release()
        except Exception:
            pass
        logger.info("WebcamStream stopped")
