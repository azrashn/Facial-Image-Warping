import logging
from typing import Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import Delaunay

try:
    from modules.warping_module import detect_face_landmarks, _has_duplicate_vertices, triangle_area
except ModuleNotFoundError:
    from backend.modules.warping_module import detect_face_landmarks, _has_duplicate_vertices, triangle_area

logger = logging.getLogger(__name__)

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

class FaceSwapError(Exception):
    pass

class FaceSwapEngine:
    """
    Core engine for realtime face swapping.
    Caches source face, landmarks, and Delaunay triangulation for 30+ FPS performance.
    Uses an ROI-based cv2.seamlessClone approach for optimal speed.
    """
    def __init__(self):
        self.source_image: Optional[np.ndarray] = None
        self.source_landmarks: Optional[np.ndarray] = None
        self.source_triangles: Optional[np.ndarray] = None
        self.is_loaded = False
        
        # We also cache the source indices we triangulate over
        self.used_indices: Optional[list[int]] = None

    def process_source_image(self, image_bytes: bytes) -> None:
        if not image_bytes:
            raise FaceSwapError("Empty image data provided.")

        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceSwapError("Could not decode source image bytes.")

        lm = detect_face_landmarks(img)
        if lm is None or len(lm) < 100:
            raise FaceSwapError("No face detected in source image.")

        self.source_image = img
        self.source_landmarks = lm

        # Use Face Oval + some interior anchor points
        interior_indices = [
            1, 2, 4, 5, 6, 19, 94, 168, # Nose
            33, 133, 160, 158, 153, 144, 159, 145, # Left eye
            362, 263, 387, 385, 380, 373, 386, 374, # Right eye
            61, 291, 0, 17, 78, 308, 13, 14, 87, 317, # Mouth outer
            82, 312, 311, 310, 415, 324, 318, 402, 95, 88, # Mouth inner
            205, 425, 50, 280, 117, 346, 118, 347, # Cheeks
            111, 340, # Under eye
        ]
        
        n_lm = self.source_landmarks.shape[0]
        self.used_indices = sorted(set(
            [i for i in FACE_OVAL_INDICES if i < n_lm] +
            [i for i in interior_indices if i < n_lm]
        ))
        
        src_pts = self.source_landmarks[self.used_indices].astype(np.float32)
        try:
            tri = Delaunay(src_pts)
            self.source_triangles = tri.simplices
        except Exception as exc:
            raise FaceSwapError(f"Delaunay triangulation failed on source face: {exc}")

        self.is_loaded = True
        logger.info(f"FaceSwapEngine: source loaded, {len(self.source_triangles)} triangles cached.")

    def apply_face_swap(self, target_frame: np.ndarray, target_landmarks: np.ndarray) -> np.ndarray:
        if not self.is_loaded or target_landmarks is None:
            return target_frame

        h, w = target_frame.shape[:2]
        
        # Extract target points matching our triangulated subset
        try:
            src_pts = self.source_landmarks[self.used_indices].astype(np.float32)
            dst_pts = target_landmarks[self.used_indices].astype(np.float32)
        except IndexError:
            return target_frame

        # Compute Bounding Box (ROI) on target
        dst_oval_pts = np.array([target_landmarks[i] for i in FACE_OVAL_INDICES if i < len(target_landmarks)], dtype=np.int32)
        if len(dst_oval_pts) < 3:
            return target_frame
            
        x_min, y_min, w_roi, h_roi = cv2.boundingRect(dst_oval_pts)
        
        # Add padding
        pad = int(max(w_roi, h_roi) * 0.15)
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_min + w_roi + 2*pad)
        y_max = min(h, y_min + h_roi + 2*pad)
        w_roi = x_max - x_min
        h_roi = y_max - y_min
        
        if w_roi <= 0 or h_roi <= 0:
            return target_frame

        target_roi = target_frame[y_min:y_max, x_min:x_max]
        warped_roi = np.zeros_like(target_roi)
        
        # Warp Source to Target (within ROI)
        for ia, ib, ic in self.source_triangles:
            src_tri = np.array([src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32).reshape(3, 2)
            dst_tri = np.array([dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32).reshape(3, 2)

            if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
                continue
            if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
                continue
                
            # Shift dst_tri to ROI coordinates
            dst_tri_roi = dst_tri - [x_min, y_min]

            r = cv2.boundingRect(dst_tri_roi)
            bx, by, bw, bh = r
            bx = max(bx, 0)
            by = max(by, 0)
            bw = min(bw, w_roi - bx)
            bh = min(bh, h_roi - by)
            if bw <= 0 or bh <= 0:
                continue

            try:
                mask = np.zeros((bh, bw), dtype=np.uint8)
                dst_crop = (dst_tri_roi - [bx, by]).astype(np.float32).reshape(3, 2)
                src_crop = src_tri.astype(np.float32).reshape(3, 2)
                cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)

                warp_mat = cv2.getAffineTransform(src_crop, dst_crop)
                warped_patch = cv2.warpAffine(
                    self.source_image, warp_mat, (bw, bh),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )

                roi_patch = warped_roi[by: by + bh, bx: bx + bw]
                blended = np.where(mask[..., None] == 255, warped_patch, roi_patch)
                warped_roi[by: by + bh, bx: bx + bw] = blended
            except Exception:
                continue

        # ROI seamless cloning
        # Shift oval points to ROI coordinates for mask
        dst_oval_pts_roi = dst_oval_pts - [x_min, y_min]
        
        clone_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        hull = cv2.convexHull(dst_oval_pts_roi)
        cv2.fillConvexPoly(clone_mask, hull, 255)

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w_roi // 25), max(3, h_roi // 25)))
        clone_mask = cv2.erode(clone_mask, erode_kernel, iterations=1)
        clone_mask = cv2.GaussianBlur(clone_mask, (15, 15), 15 * 0.3)

        center_x = int(np.mean(dst_oval_pts_roi[:, 0]))
        center_y = int(np.mean(dst_oval_pts_roi[:, 1]))
        center_x = max(1, min(w_roi - 2, center_x))
        center_y = max(1, min(h_roi - 2, center_y))

        try:
            blended_roi = cv2.seamlessClone(
                warped_roi,
                target_roi,
                clone_mask,
                (center_x, center_y),
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            # Fallback
            mask_f = clone_mask.astype(np.float32) / 255.0
            mask_3 = mask_f[..., np.newaxis]
            blended_roi = (warped_roi.astype(np.float32) * mask_3 + target_roi.astype(np.float32) * (1.0 - mask_3))
            blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

        # Place the blended ROI back into the target frame
        result_frame = target_frame.copy()
        result_frame[y_min:y_max, x_min:x_max] = blended_roi

        return result_frame

# Global engine instance
face_swap_engine = FaceSwapEngine()
