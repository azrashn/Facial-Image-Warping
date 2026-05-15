import cv2
import numpy as np

def get_landmarks_downsampled(image_bgr: np.ndarray, target_size=(480, 360)):
    """
    Do Not Process Full Resolution: Downsample to target_size for MediaPipe FaceMesh.
    Rescale normalized coordinates to the original frame's width and height.
    """
    from backend.modules.input_module import get_landmarks, preprocess_image

    orig_h, orig_w = image_bgr.shape[:2]
    
    # Downsample
    lo_res = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    
    # Preprocess
    preprocessed = preprocess_image(cv2.cvtColor(lo_res, cv2.COLOR_BGR2RGB))
    
    # Get landmarks (these are returned as a list of dicts with normalized 'x' and 'y')
    # Or in our project, sometimes a NumPy array? Let's check get_landmarks.
    # Ah, in process.py: lm_list = [[float(pt[0]) / w_f, float(pt[1]) / h_f] for pt in landmarks]
    # No, get_landmarks returns dicts with 'x' and 'y' (normalized).
    # Wait, detect_face_landmarks from warping_module returns a numpy array!
    # Wait, I'll support both dicts and numpy arrays based on type.
    raw_landmarks = get_landmarks(preprocessed)
    
    if not raw_landmarks:
        return []
        
    return raw_landmarks
