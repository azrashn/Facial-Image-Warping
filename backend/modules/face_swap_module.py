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

# Inner mouth (lip interior) landmarks for mouth occlusion mask.
# These trace the inside edge of the lips so that the target's real
# mouth void (teeth, tongue, shadow) is preserved over the warped source.
INNER_MOUTH_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
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
        self.running = False
        
        # We also cache the source indices we triangulate over
        self.used_indices: Optional[list[int]] = None

    def process_source_image(self, image_bytes: bytes) -> None:
        if not image_bytes:
            raise FaceSwapError("Empty image data provided.")

        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceSwapError("Could not decode source image bytes.")

        # CRITICAL: Ensure source image is uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

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
        logger.info(f"[FACE_SWAP] source loaded, {len(self.source_triangles)} triangles cached.")

    def apply_face_swap(self, target_frame: np.ndarray, target_landmarks: np.ndarray) -> np.ndarray:
        logger.info(f"[FACE_SWAP] apply_face_swap called | source_loaded={self.is_loaded} | running={self.running}")

        if not self.is_loaded or target_landmarks is None:
            logger.debug("[FACE_SWAP] Skipping: not loaded or no target landmarks")
            return target_frame

        # CRITICAL: Enforce uint8 on input frame
        if target_frame.dtype != np.uint8:
            target_frame = np.clip(target_frame, 0, 255).astype(np.uint8)

        h, w = target_frame.shape[:2]
        
        # Extract target points matching our triangulated subset
        try:
            src_pts = self.source_landmarks[self.used_indices].astype(np.float32)
            dst_pts = target_landmarks[self.used_indices].astype(np.float32)
        except IndexError:
            logger.warning("[FACE_SWAP] IndexError accessing landmarks subset")
            return target_frame

        # Compute Bounding Box (ROI) on target
        # CRITICAL: Force int32 for all point arrays passed to OpenCV
        dst_oval_pts = np.array(
            [target_landmarks[i] for i in FACE_OVAL_INDICES if i < len(target_landmarks)],
            dtype=np.int32
        )
        if len(dst_oval_pts) < 3:
            logger.warning("[FACE_SWAP] Not enough oval points for bounding rect")
            return target_frame

        # SAFETY ASSERTION: points must be int32 for cv2.boundingRect
        assert dst_oval_pts.dtype == np.int32, f"dst_oval_pts dtype={dst_oval_pts.dtype}, expected int32"
            
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
            logger.warning("[FACE_SWAP] ROI has zero dimensions")
            return target_frame

        target_roi = target_frame[y_min:y_max, x_min:x_max].copy()
        warped_roi = np.zeros_like(target_roi)
        
        logger.debug(f"[FACE_SWAP] Processing {len(self.source_triangles)} triangles in ROI ({w_roi}x{h_roi})")

        # Warp Source to Target (within ROI)
        triangles_warped = 0
        for ia, ib, ic in self.source_triangles:
            src_tri = np.array([src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32).reshape(3, 2)
            dst_tri = np.array([dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32).reshape(3, 2)

            if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
                continue
            if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
                continue
                
            # Shift dst_tri to ROI coordinates
            dst_tri_roi = dst_tri - np.array([x_min, y_min], dtype=np.float32)

            # CRITICAL: cv2.boundingRect needs int32 input for float arrays
            dst_tri_roi_int = np.int32(dst_tri_roi)
            r = cv2.boundingRect(dst_tri_roi_int)
            bx, by, bw, bh = r
            bx = max(bx, 0)
            by = max(by, 0)
            bw = min(bw, w_roi - bx)
            bh = min(bh, h_roi - by)
            if bw <= 0 or bh <= 0:
                continue

            try:
                # CRITICAL: mask must be uint8
                mask = np.zeros((bh, bw), dtype=np.uint8)
                dst_crop = (dst_tri_roi - np.array([bx, by], dtype=np.float32)).astype(np.float32).reshape(3, 2)
                src_crop = src_tri.astype(np.float32).reshape(3, 2)
                # CRITICAL: fillConvexPoly needs int32 points
                cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)

                warp_mat = cv2.getAffineTransform(src_crop, dst_crop)
                warped_patch = cv2.warpAffine(
                    self.source_image, warp_mat, (bw, bh),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )

                # Ensure warped_patch is uint8
                if warped_patch.dtype != np.uint8:
                    warped_patch = np.clip(warped_patch, 0, 255).astype(np.uint8)

                roi_patch = warped_roi[by: by + bh, bx: bx + bw]
                blended = np.where(mask[..., None] == 255, warped_patch, roi_patch)
                warped_roi[by: by + bh, bx: bx + bw] = blended
                triangles_warped += 1
            except Exception as exc:
                logger.debug(f"[FACE_SWAP] Triangle ({ia},{ib},{ic}) warp failed: {exc}")
                continue

        logger.debug(f"[FACE_SWAP] Warped {triangles_warped}/{len(self.source_triangles)} triangles")

        if triangles_warped == 0:
            logger.warning("[FACE_SWAP] No triangles warped successfully, returning original")
            return target_frame

        # ROI seamless cloning
        # Shift oval points to ROI coordinates for mask
        dst_oval_pts_roi = dst_oval_pts - np.array([x_min, y_min], dtype=np.int32)
        
        # CRITICAL: clone_mask must be uint8
        clone_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        
        # CRITICAL: convexHull needs int32 points
        hull = cv2.convexHull(np.array(dst_oval_pts_roi, dtype=np.int32))
        cv2.fillConvexPoly(clone_mask, hull, 255)

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w_roi // 25), max(3, h_roi // 25)))
        clone_mask = cv2.erode(clone_mask, erode_kernel, iterations=1)
        clone_mask = cv2.GaussianBlur(clone_mask, (15, 15), 15 * 0.3)

        # CRITICAL: Ensure clone_mask is still uint8 after blur
        if clone_mask.dtype != np.uint8:
            clone_mask = np.clip(clone_mask, 0, 255).astype(np.uint8)

        # Compute center for seamlessClone — must be int tuple inside ROI
        center_x = int(np.mean(dst_oval_pts_roi[:, 0]))
        center_y = int(np.mean(dst_oval_pts_roi[:, 1]))
        center_x = max(1, min(w_roi - 2, center_x))
        center_y = max(1, min(h_roi - 2, center_y))

        # SAFETY ASSERTIONS before seamlessClone
        assert warped_roi.dtype == np.uint8, f"warped_roi dtype={warped_roi.dtype}"
        assert target_roi.dtype == np.uint8, f"target_roi dtype={target_roi.dtype}"
        assert clone_mask.dtype == np.uint8, f"clone_mask dtype={clone_mask.dtype}"

        try:
            blended_roi = cv2.seamlessClone(
                warped_roi,
                target_roi,
                clone_mask,
                (center_x, center_y),
                cv2.NORMAL_CLONE
            )
            logger.debug("[FACE_SWAP] seamlessClone succeeded")
        except cv2.error as e:
            logger.warning(f"[FACE_SWAP] seamlessClone failed: {e}, using alpha fallback")
            # Fallback: manual alpha blending
            mask_f = clone_mask.astype(np.float32) / 255.0
            mask_3 = mask_f[..., np.newaxis]
            blended_roi = (warped_roi.astype(np.float32) * mask_3 + target_roi.astype(np.float32) * (1.0 - mask_3))
            blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

        # ══════════════════════════════════════════════════════════════════
        # MOUTH OCCLUSION PASS
        # After seamlessClone blends skin tones, we restore the target's
        # real inner-mouth pixels (teeth, tongue, shadow) on top.
        # Math:  result = target * mouth_alpha + swapped * (1 - mouth_alpha)
        # ══════════════════════════════════════════════════════════════════
        try:
            n_lm = len(target_landmarks)
            inner_mouth_pts = np.array(
                [target_landmarks[i] for i in INNER_MOUTH_INDICES if i < n_lm],
                dtype=np.int32,
            )
            if len(inner_mouth_pts) >= 6:
                # Shift to ROI coordinates
                inner_mouth_roi = inner_mouth_pts - np.array([x_min, y_min], dtype=np.int32)

                # Build mouth mask (white = mouth interior)
                mouth_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
                cv2.fillConvexPoly(mouth_mask, cv2.convexHull(inner_mouth_roi), 255)

                # Feather edges with Gaussian blur for smooth transition
                blur_k = max(7, (min(w_roi, h_roi) // 30) | 1)  # ensure odd
                mouth_mask = cv2.GaussianBlur(mouth_mask, (blur_k, blur_k), blur_k * 0.4)

                # Erode slightly so the lip boundary stays from the swap
                erode_k = max(3, min(w_roi, h_roi) // 50)
                erode_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
                mouth_mask = cv2.erode(mouth_mask, erode_el, iterations=1)
                mouth_mask = cv2.GaussianBlur(mouth_mask, (blur_k, blur_k), blur_k * 0.3)

                # Alpha blend: where mouth_mask is white → use target_roi (real mouth)
                alpha = mouth_mask.astype(np.float32) / 255.0
                alpha_3 = alpha[..., np.newaxis]
                blended_roi = (
                    target_roi.astype(np.float32) * alpha_3
                    + blended_roi.astype(np.float32) * (1.0 - alpha_3)
                )
                blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
                logger.debug("[FACE_SWAP] Mouth occlusion applied (%d inner-mouth pts)", len(inner_mouth_pts))
        except Exception as exc:
            logger.warning("[FACE_SWAP] Mouth occlusion pass failed (non-fatal): %s", exc)

        # Place the blended ROI back into the target frame
        result_frame = target_frame.copy()
        result_frame[y_min:y_max, x_min:x_max] = blended_roi

        logger.info("[FACE_SWAP] frame_processed=True")
        return result_frame

# Global engine instance
face_swap_engine = FaceSwapEngine()
