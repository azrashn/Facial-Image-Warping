import cv2
import numpy as np
from backend.modules.frequency_module import apply_virtual_makeup

img = np.zeros((100, 100, 3), dtype=np.uint8)
landmarks = [{"x": 0.5, "y": 0.5}] * 478 # mock landmarks
try:
    res = apply_virtual_makeup(img, landmarks, "lip", 10, 0.5)
    print("Success, shape:", res.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
