import os
import cv2
import numpy as np
import threading
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

_CLOTHING_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "clothing")
)

# Coordinates of the key points in the 1024x1024 clothing templates
# In 1024x1024 space:
# Right points are on the viewer's left (smaller x)
# Left points are on the viewer's right (greater x)
CLOTHING_COORDS = {
    "tshirt": {
        "right_shoulder": [270.0, 160.0],
        "left_shoulder": [754.0, 160.0],
        "right_hip": [310.0, 900.0],
        "left_hip": [714.0, 900.0],
        "right_sleeve_end": [180.0, 260.0],
        "left_sleeve_end": [844.0, 260.0]
    },
    "shirt": {
        "right_shoulder": [270.0, 160.0],
        "left_shoulder": [754.0, 160.0],
        "right_hip": [310.0, 900.0],
        "left_hip": [714.0, 900.0],
        "right_sleeve_end": [180.0, 260.0],
        "left_sleeve_end": [844.0, 260.0]
    },
    "tanktop": {
        "right_shoulder": [270.0, 160.0],
        "left_shoulder": [754.0, 160.0],
        "right_hip": [310.0, 900.0],
        "left_hip": [714.0, 900.0]
    }
}

# MediaPipe Pose landmarks mapped to local names
POSE_LANDMARKS_MAP = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip"
}


class PersistentPoseTracker:
    """
    Thread-safe persistent MediaPipe Pose detector for real-time video streaming.
    Includes temporal smoothing (EMA filter), downsampling, and non-blocking asynchronous detection.
    """

    def __init__(self, alpha: float = 0.65, min_visibility: float = 0.5, downsample_size: Tuple[int, int] = (320, 240)):
        self._lock = threading.Lock()
        self._pose = None
        self.alpha = alpha
        self.min_visibility = min_visibility
        self.downsample_size = downsample_size
        
        # Tracking history
        self.prev_raw_pts: Optional[np.ndarray] = None
        self.prev_landmarks: Optional[Dict[str, Tuple[float, float]]] = None
        self.prev_visibilities: Optional[Dict[str, float]] = None
        
        # Async background worker states
        self.last_frame: Optional[np.ndarray] = None
        self.is_processing = False
        self.tracked_indices = [11, 12, 13, 14, 15, 16, 23, 24]

    def _get_pose(self):
        if self._pose is None:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Use model 0 (Lite) for maximum real-time FPS
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose initialized successfully (model complexity 0 with SSL bypass).")
        return self._pose

    def detect(self, image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Synchronous landmark detection.
        Downsamples internally, extracts points, filters with EMA, and maps back to original scale.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        h_orig, w_orig = image_bgr.shape[:2]
        
        # Downsample for fast execution
        low_w, low_h = self.downsample_size
        resized = cv2.resize(image_bgr, (low_w, low_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        with self._lock:
            pose_detector = self._get_pose()
            try:
                results = pose_detector.process(rgb)
            except Exception as exc:
                logger.error("MediaPipe Pose process failed: %s", exc)
                return None

            if not results.pose_landmarks:
                self.prev_raw_pts = None
                self.prev_landmarks = None
                self.prev_visibilities = None
                return None

            landmarks = results.pose_landmarks.landmark
            current_pts = []
            valid = True

            # We need at least shoulders and hips to warp the clothing
            critical_indices = [11, 12, 23, 24]
            for idx in self.tracked_indices:
                lm = landmarks[idx]
                if idx in critical_indices and lm.visibility < self.min_visibility:
                    valid = False
                current_pts.append([lm.x, lm.y])

            if not valid:
                self.prev_raw_pts = None
                self.prev_landmarks = None
                self.prev_visibilities = None
                return None

            current_pts = np.array(current_pts, dtype=np.float32)

            # Apply temporal Exponential Moving Average (EMA) smoothing
            if self.prev_raw_pts is None or self.prev_raw_pts.shape != current_pts.shape:
                smoothed_pts = current_pts
            else:
                smoothed_pts = self.alpha * current_pts + (1.0 - self.alpha) * self.prev_raw_pts

            self.prev_raw_pts = smoothed_pts.copy()

            # Map back to original pixels and build dictionary
            pixel_landmarks = {}
            visibilities = {}
            for i, idx in enumerate(self.tracked_indices):
                name = POSE_LANDMARKS_MAP[idx]
                x_pix = float(smoothed_pts[i, 0] * w_orig)
                y_pix = float(smoothed_pts[i, 1] * h_orig)
                pixel_landmarks[name] = (x_pix, y_pix)
                visibilities[name] = float(landmarks[idx].visibility)

            # Neck point calculated as midpoint between shoulders
            ls = pixel_landmarks["left_shoulder"]
            rs = pixel_landmarks["right_shoulder"]
            pixel_landmarks["neck"] = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            visibilities["neck"] = min(visibilities["left_shoulder"], visibilities["right_shoulder"])

            # Store visibilities in a special key
            pixel_landmarks["visibilities"] = visibilities

            self.prev_landmarks = pixel_landmarks
            self.prev_visibilities = visibilities

            return pixel_landmarks

    def detect_async(self, image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Asynchronous, non-blocking landmark detection.
        If the background thread is busy, immediately returns the last known landmarks.
        Otherwise, spawns a thread to process the new frame.
        """
        if image_bgr is None or image_bgr.size == 0:
            return self._build_result()

        with self._lock:
            self.last_frame = image_bgr.copy()
            if not self.is_processing:
                self.is_processing = True
                threading.Thread(target=self._bg_detect_worker, daemon=True).start()

        return self._build_result()

    def _bg_detect_worker(self):
        """Worker function running in a background thread."""
        try:
            with self._lock:
                frame_to_process = self.last_frame
                if frame_to_process is None:
                    self.is_processing = False
                    return

            # Perform the detection on the saved frame
            h_orig, w_orig = frame_to_process.shape[:2]
            low_w, low_h = self.downsample_size
            resized = cv2.resize(frame_to_process, (low_w, low_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            pose_detector = self._get_pose()
            results = pose_detector.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                current_pts = []
                valid = True
                critical_indices = [11, 12, 23, 24]
                
                for idx in self.tracked_indices:
                    lm = landmarks[idx]
                    if idx in critical_indices and lm.visibility < self.min_visibility:
                        valid = False
                    current_pts.append([lm.x, lm.y])

                if valid:
                    current_pts = np.array(current_pts, dtype=np.float32)
                    
                    # Apply EMA
                    if self.prev_raw_pts is None or self.prev_raw_pts.shape != current_pts.shape:
                        smoothed_pts = current_pts
                    else:
                        smoothed_pts = self.alpha * current_pts + (1.0 - self.alpha) * self.prev_raw_pts

                    self.prev_raw_pts = smoothed_pts.copy()

                    # Convert to pixel space of the original high-resolution frame
                    pixel_landmarks = {}
                    visibilities = {}
                    for i, idx in enumerate(self.tracked_indices):
                        name = POSE_LANDMARKS_MAP[idx]
                        x_pix = float(smoothed_pts[i, 0] * w_orig)
                        y_pix = float(smoothed_pts[i, 1] * h_orig)
                        pixel_landmarks[name] = (x_pix, y_pix)
                        visibilities[name] = float(landmarks[idx].visibility)

                    # Neck
                    ls = pixel_landmarks["left_shoulder"]
                    rs = pixel_landmarks["right_shoulder"]
                    pixel_landmarks["neck"] = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
                    visibilities["neck"] = min(visibilities["left_shoulder"], visibilities["right_shoulder"])

                    # Save to state
                    with self._lock:
                        self.prev_landmarks = pixel_landmarks
                        self.prev_visibilities = visibilities
                else:
                    with self._lock:
                        self.prev_raw_pts = None
                        self.prev_landmarks = None
                        self.prev_visibilities = None
            else:
                with self._lock:
                    self.prev_raw_pts = None
                    self.prev_landmarks = None
                    self.prev_visibilities = None

        except Exception as exc:
            logger.error("Background pose detection worker failed: %s", exc)
        finally:
            with self._lock:
                self.is_processing = False

    def _build_result(self) -> Optional[Dict[str, Any]]:
        """Safely fetch current tracked landmarks state."""
        with self._lock:
            if self.prev_landmarks is None:
                return None
            res = dict(self.prev_landmarks)
            if self.prev_visibilities is not None:
                res["visibilities"] = dict(self.prev_visibilities)
            return res

    def close(self):
        """Release MediaPipe resources."""
        with self._lock:
            if self._pose is not None:
                try:
                    self._pose.close()
                except Exception:
                    pass
                self._pose = None


_CLOTHING_CACHE = {}
_CACHE_LOCK = threading.Lock()


def load_clothing_template(clothing_type: str) -> Optional[np.ndarray]:
    """Load a clothing template image (BGRA) from disk with caching and edge bleeding prevention."""
    global _CLOTHING_CACHE
    with _CACHE_LOCK:
        if clothing_type in _CLOTHING_CACHE and clothing_type != "tanktop":
            return _CLOTHING_CACHE[clothing_type]

    path = os.path.join(_CLOTHING_ROOT, f"{clothing_type}_model.png")
    if not os.path.exists(path):
        logger.error("Clothing template file not found: %s", path)
        return None
    try:
        overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            logger.error("Failed to load clothing template image: %s", path)
            return None
        
        # Prevent edge color bleeding during bilinear interpolation by clearing BGR values of transparent pixels
        if overlay.shape[2] == 4:
            overlay = overlay.copy()
            bgr = overlay[:, :, :3]
            alpha = overlay[:, :, 3]
            bgr[alpha == 0] = 0
            overlay[:, :, :3] = bgr

        with _CACHE_LOCK:
            _CLOTHING_CACHE[clothing_type] = overlay
        return overlay
    except Exception as exc:
        logger.error("Exception loading clothing template: %s", exc)
        return None





def apply_clothing_overlay(
    frame: np.ndarray,
    landmarks: Optional[Dict[str, Any]],
    clothing_type: str = "tshirt",
    mode: str = "auto"
) -> np.ndarray:
    """
    Warp and fit clothing onto the user using advanced pose and aspect-ratio compensation.
    Supports single Homography and Segmented (torso + sleeves) warping modes.
    """
    if landmarks is None:
        return frame

    required_keys = ["right_shoulder", "left_shoulder", "right_hip", "left_hip"]
    if not all(k in landmarks for k in required_keys):
        return frame

    if clothing_type not in CLOTHING_COORDS:
        logger.error("Unknown clothing type: %s", clothing_type)
        return frame

    overlay = load_clothing_template(clothing_type)
    if overlay is None:
        return frame

    h_frame, w_frame = frame.shape[:2]
    coords = CLOTHING_COORDS[clothing_type]

    # --- DYNAMIC SCALING & POSTURE ANGLE COMPENSATION ---
    rs_raw = np.array(landmarks["right_shoulder"])
    ls_raw = np.array(landmarks["left_shoulder"])
    rh_raw = np.array(landmarks["right_hip"])
    lh_raw = np.array(landmarks["left_hip"])
    neck = np.array(landmarks["neck"])

    # Compute shoulder axis direction
    v_shoulder = ls_raw - rs_raw
    shoulder_dist = np.linalg.norm(v_shoulder)
    if shoulder_dist < 1.0:
        shoulder_dist = 1.0
    u_sh = v_shoulder / shoulder_dist

    # Compute torso vertical direction
    hip_center_calib = (rh_raw + lh_raw) / 2.0
    v_torso = hip_center_calib - neck
    torso_height = np.linalg.norm(v_torso)
    if torso_height < 1.0:
        torso_height = 1.0
    u_to = v_torso / torso_height

    # Milimetrik Boyun ve Omuz Kalibrasyonu (12% vertical offset adjustment upwards)
    offset_up = -0.12 * shoulder_dist * u_to
    rs_calib = rs_raw + offset_up
    ls_calib = ls_raw + offset_up
    rh_calib = rh_raw + offset_up
    lh_calib = lh_raw + offset_up
    
    neck_calibrated = (rs_calib + ls_calib) / 2.0

    # Template measurements
    sh_temp_dist = np.linalg.norm(np.array(coords["left_shoulder"]) - np.array(coords["right_shoulder"]))
    hip_temp_dist = np.linalg.norm(np.array(coords["left_hip"]) - np.array(coords["right_hip"]))
    hip_dist_ratio = hip_temp_dist / (sh_temp_dist + 1e-6)

    # Aspect Ratio Protection (Yaw/Pitch Turn Compensation)
    to_temp_height = np.linalg.norm((np.array(coords["left_hip"]) + np.array(coords["right_hip"])) / 2.0 - np.array([512.0, 160.0]))
    ratio_temp = sh_temp_dist / to_temp_height

    min_ratio = ratio_temp * 0.82
    max_ratio = ratio_temp * 1.25
    current_ratio = shoulder_dist / torso_height

    shoulder_dist_comp = shoulder_dist
    torso_height_comp = torso_height

    if current_ratio < min_ratio:
        shoulder_dist_comp = torso_height * min_ratio
    elif current_ratio > max_ratio:
        torso_height_comp = shoulder_dist / max_ratio

    # Apply Padding and Scaling Factors
    w_scale = 1.15
    h_scale = 1.10
    
    scale_w_factor = (shoulder_dist_comp / shoulder_dist) * w_scale
    scale_h_factor = (torso_height_comp / torso_height) * h_scale

    # Calculate padded coordinates relative to landmarks to preserve yaw/pitch asymmetry
    rs_padded = neck_calibrated + (rs_calib - neck_calibrated) * scale_w_factor
    ls_padded = neck_calibrated + (ls_calib - neck_calibrated) * scale_w_factor
    
    hip_center_padded = neck_calibrated + (hip_center_calib - neck_calibrated) * scale_h_factor
    # Remove waist pinch effect by scaling bottom width naturally according to user actual contours
    scale_waist_factor = scale_w_factor * 1.08
    rh_padded = hip_center_padded + (rh_calib - hip_center_calib) * scale_waist_factor
    lh_padded = hip_center_padded + (lh_calib - hip_center_calib) * scale_waist_factor

    dst_torso = np.array([rs_padded, ls_padded, rh_padded, lh_padded], dtype=np.float32)
    src_torso = np.array([coords["right_shoulder"], coords["left_shoulder"], coords["right_hip"], coords["left_hip"]], dtype=np.float32)

    # Determine if arms/sleeves are visible and segment-warp is possible
    vis = landmarks.get("visibilities", {})
    arms_visible = False
    
    if mode in ("auto", "segmented"):
        required_arm_keys = ["right_elbow", "left_elbow", "right_wrist", "left_wrist"]
        if all(k in landmarks for k in required_arm_keys):
            # Safe threshold for arm visibility
            if all(vis.get(k, 0.0) >= 0.4 for k in required_arm_keys):
                arms_visible = True

    if clothing_type == "tanktop":
        return _apply_homography_warp(frame, overlay, src_torso, dst_torso)

    if (mode == "segmented" or (mode == "auto" and arms_visible)) and overlay.shape[2] == 4:
        return _apply_segmented_warp(
            frame, overlay, landmarks, coords,
            rs_padded, ls_padded, dst_torso, src_torso, clothing_type
        )
    else:
        return _apply_homography_warp(frame, overlay, src_torso, dst_torso)


def _apply_homography_warp(
    frame: np.ndarray,
    overlay: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray
) -> np.ndarray:
    """Warp the entire clothing template using a single Homography matrix in ROI space."""
    h_frame, w_frame = frame.shape[:2]
    xs = dst_pts[:, 0]
    ys = dst_pts[:, 1]
    
    # Calculate scale-invariant margins based on shoulders and torso
    shoulder_dist = np.linalg.norm(dst_pts[0] - dst_pts[1])
    torso_height = np.linalg.norm((dst_pts[0] + dst_pts[1])/2.0 - (dst_pts[2] + dst_pts[3])/2.0)
    
    margin_x = int(0.45 * shoulder_dist)
    margin_y_top = int(0.25 * shoulder_dist)
    margin_y_bottom = int(0.25 * torso_height)
    
    xmin = max(0, int(np.min(xs) - margin_x))
    xmax = min(w_frame, int(np.max(xs) + margin_x))
    ymin = max(0, int(np.min(ys) - margin_y_top))
    ymax = min(h_frame, int(np.max(ys) + margin_y_bottom))
    
    roi_w = xmax - xmin
    roi_h = ymax - ymin
    if roi_w <= 4 or roi_h <= 4:
        return frame
        
    roi_frame = frame[ymin:ymax, xmin:xmax]
    
    dst_pts_roi = dst_pts.copy()
    dst_pts_roi[:, 0] -= xmin
    dst_pts_roi[:, 1] -= ymin
    
    try:
        H = cv2.getPerspectiveTransform(src_pts, dst_pts_roi)
    except Exception as exc:
        logger.error("Homography transform matrix failed: %s", exc)
        return frame

    if overlay.shape[2] == 4:
        bgr = overlay[:, :, :3]
        alpha = overlay[:, :, 3]
    else:
        bgr = overlay
        alpha = np.ones(overlay.shape[:2], dtype=np.uint8) * 255

    try:
        warped_bgr = cv2.warpPerspective(bgr, H, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpPerspective(alpha, H, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
    except Exception as exc:
        logger.error("cv2.warpPerspective failed: %s", exc)
        return frame

    alpha_mask = warped_alpha.astype(np.float32) / 255.0
    alpha_mask = np.expand_dims(alpha_mask, axis=2)

    blended = roi_frame.astype(np.float32) * (1.0 - alpha_mask) + warped_bgr.astype(np.float32) * alpha_mask
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    out = frame.copy()
    out[ymin:ymax, xmin:xmax] = blended
    return out


def _apply_segmented_warp(
    frame: np.ndarray,
    overlay: np.ndarray,
    landmarks: Dict[str, Any],
    coords: Dict[str, Any],
    rs_padded: np.ndarray,
    ls_padded: np.ndarray,
    dst_torso: np.ndarray,
    src_torso: np.ndarray,
    clothing_type: str
) -> np.ndarray:
    """
    Segment the template into torso, left arm, and right arm.
    Warp each segment independently in ROI space to align sleeves, then composite.
    """
    h_frame, w_frame = frame.shape[:2]
    
    # 1. Collect all destination points to find bounding box
    all_pts = list(dst_torso)
    all_pts.append(rs_padded)
    all_pts.append(ls_padded)
    
    r_el = np.array(landmarks.get("right_elbow", [0.0, 0.0]))
    r_wr = np.array(landmarks.get("right_wrist", [0.0, 0.0]))
    l_el = np.array(landmarks.get("left_elbow", [0.0, 0.0]))
    l_wr = np.array(landmarks.get("left_wrist", [0.0, 0.0]))
    
    for pt in (r_el, r_wr, l_el, l_wr):
        if pt[0] > 0 and pt[1] > 0:
            all_pts.append(pt)
            
    all_pts = np.array(all_pts, dtype=np.float32)
    xs = all_pts[:, 0]
    ys = all_pts[:, 1]
    
    shoulder_dist = np.linalg.norm(dst_torso[0] - dst_torso[1])
    torso_height = np.linalg.norm((dst_torso[0] + dst_torso[1])/2.0 - (dst_torso[2] + dst_torso[3])/2.0)
    
    margin_x = int(0.45 * shoulder_dist)
    margin_y_top = int(0.25 * shoulder_dist)
    margin_y_bottom = int(0.25 * torso_height)
    
    xmin = max(0, int(np.min(xs) - margin_x))
    xmax = min(w_frame, int(np.max(xs) + margin_x))
    ymin = max(0, int(np.min(ys) - margin_y_top))
    ymax = min(h_frame, int(np.max(ys) + margin_y_bottom))
    
    roi_w = xmax - xmin
    roi_h = ymax - ymin
    if roi_w <= 4 or roi_h <= 4:
        return frame
        
    roi_frame = frame[ymin:ymax, xmin:xmax]
    
    # 2. Adjust target coordinates to ROI space
    dst_torso_roi = dst_torso.copy()
    dst_torso_roi[:, 0] -= xmin
    dst_torso_roi[:, 1] -= ymin
    
    rs_padded_roi = rs_padded - [xmin, ymin]
    ls_padded_roi = ls_padded - [xmin, ymin]
    
    r_el_roi = r_el - [xmin, ymin]
    r_wr_roi = r_wr - [xmin, ymin]
    l_el_roi = l_el - [xmin, ymin]
    l_wr_roi = l_wr - [xmin, ymin]
    
    h_temp, w_temp = overlay.shape[:2]

    # Define seam boundaries in template space
    right_seam = 272
    left_seam = 752
    x_coords = np.arange(w_temp)
    mask_r_sleeve = (x_coords < right_seam)[np.newaxis, :].repeat(h_temp, axis=0)
    mask_l_sleeve = (x_coords > left_seam)[np.newaxis, :].repeat(h_temp, axis=0)
    mask_torso = ((x_coords >= right_seam - 4) & (x_coords <= left_seam + 4))[np.newaxis, :].repeat(h_temp, axis=0)

    bgr = overlay[:, :, :3]
    alpha = overlay[:, :, 3]

    # Torso Homography in ROI
    try:
        H_torso = cv2.getPerspectiveTransform(src_torso, dst_torso_roi)
        torso_bgr = np.where(mask_torso[:, :, np.newaxis], bgr, 0)
        torso_alpha = np.where(mask_torso, alpha, 0)
        
        warped_torso_bgr = cv2.warpPerspective(torso_bgr, H_torso, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
        warped_torso_alpha = cv2.warpPerspective(torso_alpha, H_torso, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
    except Exception as exc:
        logger.error("Torso segment warp failed: %s", exc)
        return _apply_homography_warp(frame, overlay, src_torso, dst_torso)

    warped_composite_bgr = warped_torso_bgr.astype(np.float32)
    warped_composite_alpha = warped_torso_alpha.astype(np.float32)

    # Right arm / sleeve Affine warp in ROI
    try:
        r_sl_end_dst = rs_padded_roi + 0.4 * (r_el_roi - rs_padded_roi)
        src_r_arm = np.array([coords["right_shoulder"], coords["right_hip"], coords["right_sleeve_end"]], dtype=np.float32)
        hip_mid = (dst_torso_roi[2] + dst_torso_roi[3]) / 2.0
        dst_r_arm = np.array([rs_padded_roi, hip_mid, r_sl_end_dst], dtype=np.float32)
        M_r = cv2.getAffineTransform(src_r_arm, dst_r_arm)

        r_sleeve_bgr = np.where(mask_r_sleeve[:, :, np.newaxis], bgr, 0)
        r_sleeve_alpha = np.where(mask_r_sleeve, alpha, 0)
        
        warped_r_bgr = cv2.warpAffine(r_sleeve_bgr, M_r, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
        warped_r_alpha = cv2.warpAffine(r_sleeve_alpha, M_r, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
        
        mask_r = (warped_r_alpha > 0)
        warped_composite_alpha[mask_r] = np.maximum(warped_composite_alpha[mask_r], warped_r_alpha[mask_r])
        warped_composite_bgr[mask_r] = warped_r_bgr[mask_r]
    except Exception as exc:
        logger.debug("Right sleeve segmented warp failed or skipped: %s", exc)

    # Left arm / sleeve Affine warp in ROI
    try:
        l_sl_end_dst = ls_padded_roi + 0.4 * (l_el_roi - ls_padded_roi)
        src_l_arm = np.array([coords["left_shoulder"], coords["left_hip"], coords["left_sleeve_end"]], dtype=np.float32)
        hip_mid = (dst_torso_roi[2] + dst_torso_roi[3]) / 2.0
        dst_l_arm = np.array([ls_padded_roi, hip_mid, l_sl_end_dst], dtype=np.float32)
        M_l = cv2.getAffineTransform(src_l_arm, dst_l_arm)

        l_sleeve_bgr = np.where(mask_l_sleeve[:, :, np.newaxis], bgr, 0)
        l_sleeve_alpha = np.where(mask_l_sleeve, alpha, 0)
        
        warped_l_bgr = cv2.warpAffine(l_sleeve_bgr, M_l, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
        warped_l_alpha = cv2.warpAffine(l_sleeve_alpha, M_l, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
        
        mask_l = (warped_l_alpha > 0)
        warped_composite_alpha[mask_l] = np.maximum(warped_composite_alpha[mask_l], warped_l_alpha[mask_l])
        warped_composite_bgr[mask_l] = warped_l_bgr[mask_l]
    except Exception as exc:
        logger.debug("Left sleeve segmented warp failed or skipped: %s", exc)

    alpha_mask = np.clip(warped_composite_alpha, 0, 255) / 255.0
    alpha_mask = np.expand_dims(alpha_mask, axis=2)

    blended = roi_frame.astype(np.float32) * (1.0 - alpha_mask) + warped_composite_bgr * alpha_mask
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    out = frame.copy()
    out[ymin:ymax, xmin:xmax] = blended
    return out


def draw_pose_skeleton(
    frame: np.ndarray,
    landmarks: Optional[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw detected upper-body skeleton landmarks and connection lines.
    """
    if landmarks is None:
        return frame

    out = frame.copy()

    connections = [
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip")
    ]

    for p1_name, p2_name in connections:
        if p1_name in landmarks and p2_name in landmarks:
            pt1 = (int(round(landmarks[p1_name][0])), int(round(landmarks[p1_name][1])))
            pt2 = (int(round(landmarks[p2_name][0])), int(round(landmarks[p2_name][1])))
            cv2.line(out, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

    for name, pt in landmarks.items():
        if name == "visibilities":
            continue
        center = (int(round(pt[0])), int(round(pt[1])))
        if name == "neck":
            color = (0, 0, 255)
        elif "shoulder" in name:
            color = (255, 0, 0)
        elif "hip" in name:
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)

        cv2.circle(out, center, 6, color, -1, cv2.LINE_AA)

    return out


# Singleton tracker for convenience across routers
_pose_tracker: Optional[PersistentPoseTracker] = None
_tracker_lock = threading.Lock()


def get_pose_tracker() -> PersistentPoseTracker:
    """Lazily create and return the singleton PersistentPoseTracker."""
    global _pose_tracker
    with _tracker_lock:
        if _pose_tracker is None:
            _pose_tracker = PersistentPoseTracker()
    return _pose_tracker


def process_clothing_frame(
    frame: np.ndarray,
    clothing_type: str = "tshirt",
    show_skeleton: bool = False,
    mode: str = "auto",
    async_mode: bool = True
) -> np.ndarray:
    """
    Main frame processing entry point.
    Runs pose tracker (either blocking or async for live stream) and applies clothing warp.
    """
    tracker = get_pose_tracker()
    
    # Select between synchronous and non-blocking asynchronous detection
    if async_mode:
        landmarks = tracker.detect_async(frame)
    else:
        landmarks = tracker.detect(frame)

    result = frame
    if landmarks is not None:
        result = apply_clothing_overlay(result, landmarks, clothing_type, mode)
        if show_skeleton:
            result = draw_pose_skeleton(result, landmarks)

    return result
