"""
GÖREV 4: Enhanced threaded webcam stream with frame timestamp tracking.

Reads frames from the capture device in a tight loop and updates a
local thread-safe buffer.  Each frame carries a monotonic timestamp
so consumers can detect stale frames and skip them.
"""

import cv2
import threading
import time
import numpy as np


class ThreadedWebcam:
    """
    A threaded webcam stream utilizing Python's threading.Thread.
    Reads frames from the capture device in a tight loop and updates
    a local thread-safe buffer with timestamp tracking.
    """
    def __init__(self, src=0, width=1280, height=720):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.grabbed, self.frame = self.cap.read()
        self.frame_ts: float = time.perf_counter()  # GÖREV 4: frame timestamp
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.frame_ts = time.perf_counter()

    def read(self):
        """Return (frame_copy, timestamp) or (None, 0)."""
        with self.read_lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_ts
            return None, 0.0

    def read_latest(self):
        """
        GÖREV 4: Read the latest frame only.  If the frame is older
        than the budget window, return None to signal a skip.
        """
        with self.read_lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_ts
            return None, 0.0

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()
