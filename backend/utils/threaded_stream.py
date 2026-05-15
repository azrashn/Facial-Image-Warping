import cv2
import threading
import time
import numpy as np

class ThreadedWebcam:
    """
    A threaded webcam stream utilizing Python's threading.Thread.
    Reads frames from the capture device in a tight loop and updates a local thread-safe buffer.
    """
    def __init__(self, src=0, width=1280, height=720):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.cap.read()
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

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()
