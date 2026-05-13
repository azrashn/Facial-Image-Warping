import cv2
import urllib.request
import numpy as np
import traceback
import sys

from backend.modules.input_module import get_landmarks, preprocess_image
from backend.modules.frequency_module import apply_virtual_makeup

try:
    # Use a clear frontal face image
    url = "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_001009.jpg"
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get landmarks
    landmarks = get_landmarks(preprocess_image(img_rgb))
    
    if not landmarks:
        print("No landmarks detected.")
    else:
        print(f"Detected {len(landmarks)} landmarks.")
        
        # Apply lips makeup
        res_lip = apply_virtual_makeup(img, landmarks, region="lips", hue=10, opacity=1.0)
        cv2.imwrite("test_lips.png", res_lip)
        
        # Apply blush
        res_blush = apply_virtual_makeup(img, landmarks, region="blush", hue=10, opacity=1.0)
        cv2.imwrite("test_blush.png", res_blush)
        
        # Apply eyeshadow
        res_eye = apply_virtual_makeup(img, landmarks, region="eyeshadow", hue=10, opacity=1.0)
        cv2.imwrite("test_eyeshadow.png", res_eye)
        print("Images saved successfully!")
except Exception as e:
    traceback.print_exc()
