import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import Delaunay

try:
    from modules.pose_module import PoseEstimate, estimate_head_pose, landmark_stability_confidence
    from modules.warping_module import detect_face_landmarks, _has_duplicate_vertices, triangle_area
except ModuleNotFoundError:
    from backend.modules.pose_module import PoseEstimate, estimate_head_pose, landmark_stability_confidence
    from backend.modules.warping_module import detect_face_landmarks, _has_duplicate_vertices, triangle_area

logger = logging.getLogger(__name__)

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]
# Outer lip boundary — used together with INNER_LIP_INDICES to create
# a full mouth preservation mask that covers the entire oral cavity.
# Without this, wide mouth openings produce a hard seam between the
# swapped face and the original mouth interior.
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # Upper lip top edge
    185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Lower lip bottom edge
    146, 91, 181, 84, 17, 314, 405, 321, 375, 61,
]


@dataclass
class FaceSwapSession:
    prev_pose: Optional[PoseEstimate] = None
    prev_mask_alpha: Optional[np.ndarray] = None
    prev_landmarks: Optional[np.ndarray] = None
    stability_score: float = 1.0


@dataclass
class TriangleSourceMeta:
    src_tri: np.ndarray


class FaceSwapError(Exception):
    pass


def _alpha_blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, alpha_gain: float = 1.0) -> np.ndarray:
    mask_f = np.clip((mask.astype(np.float32) / 255.0) * alpha_gain, 0.0, 1.0)
    mask_3 = mask_f[..., np.newaxis]
    mixed = src.astype(np.float32) * mask_3 + dst.astype(np.float32) * (1.0 - mask_3)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _lab_color_transfer(src_bgr: np.ndarray, dst_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if src_bgr.size == 0 or dst_bgr.size == 0:
        return src_bgr
    valid = mask > 8
    if int(np.count_nonzero(valid)) < 64:
        return src_bgr
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_vals = src_lab[valid]
    dst_vals = dst_lab[valid]
    src_mean = src_vals.mean(axis=0)
    src_std = np.maximum(src_vals.std(axis=0), 1.0)
    dst_mean = dst_vals.mean(axis=0)
    dst_std = np.maximum(dst_vals.std(axis=0), 1.0)
    
    std_ratio = dst_std / src_std
    # Limit contrast reduction so the face doesn't become too pale
    std_ratio = np.clip(std_ratio, 0.75, 1.25)
    
    # Keep some of the source face's original color to make it obvious it's a different face
    blended_mean = 0.5 * dst_mean + 0.5 * src_mean

    out_lab = src_lab.copy()
    out_lab[valid] = ((src_lab[valid] - src_mean) * std_ratio) + blended_mean
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


class FaceSwapEngine:
    def __init__(self):
        self.source_image: Optional[np.ndarray] = None
        self.source_landmarks: Optional[np.ndarray] = None
        self.source_triangles: Optional[np.ndarray] = None
        self.source_triangle_meta: list[TriangleSourceMeta] = []
        self.is_loaded = False
        self.running = False
        self.used_indices: Optional[list[int]] = None
        self._default_session = FaceSwapSession()

    def process_source_image(self, image_bytes: bytes) -> None:
        if not image_bytes:
            raise FaceSwapError("Empty image data provided.")
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceSwapError("Could not decode source image bytes.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        lm = detect_face_landmarks(img)
        if lm is None or len(lm) < 100:
            raise FaceSwapError("No face detected in source image.")

        self.source_image = img
        self.source_landmarks = lm
        interior_indices = [
            1, 2, 4, 5, 6, 19, 94, 168, 33, 133, 160, 158, 153, 144, 159, 145,
            362, 263, 387, 385, 380, 373, 386, 374, 61, 291, 0, 17, 78, 308, 13, 14,
            87, 317, 82, 312, 311, 310, 415, 324, 318, 402, 95, 88, 205, 425, 50, 280,
            117, 346, 118, 347, 111, 340,
            # Eyebrows
            46, 52, 53, 55, 63, 65, 66, 70, 105, 107,
            276, 282, 283, 285, 293, 295, 296, 300, 334, 336
        ]
        n_lm = self.source_landmarks.shape[0]
        self.used_indices = sorted(
            set([i for i in FACE_OVAL_INDICES if i < n_lm] + [i for i in interior_indices if i < n_lm])
        )
        src_pts = self.source_landmarks[self.used_indices].astype(np.float32)
        try:
            tri = Delaunay(src_pts)
            self.source_triangles = tri.simplices
            self.source_triangle_meta = []
            for ia, ib, ic in self.source_triangles:
                src_tri = np.array([src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32).reshape(3, 2)
                self.source_triangle_meta.append(TriangleSourceMeta(src_tri=src_tri))
        except Exception as exc:
            raise FaceSwapError(f"Delaunay triangulation failed on source face: {exc}")
        self.is_loaded = True
        self._default_session = FaceSwapSession()
        logger.info("[FACE_SWAP] source loaded, %d triangles cached.", len(self.source_triangles))

    def _pose_aware_mask(self, mask: np.ndarray, yaw: float, attenuation: float) -> np.ndarray:
        """Attenuate the blend mask based on head yaw for graceful degradation.

        Two-layer attenuation:
          1. Global yaw_factor: ramps from 1.0 (full swap) at <=20° to 0.0
             (no swap) at >=40°.  Provides smooth fade-out so the swap
             doesn't vanish abruptly.
          2. Directional gradient: the occluded half of the face gets extra
             attenuation via a column-wise linear ramp.  Cap raised from
             0.55 to 0.85 so the hidden side is much more aggressively
             faded — the old 0.55 cap left 45% opacity on geometry that
             was severely distorted.
        """
        h, w = mask.shape[:2]
        if w <= 2 or h <= 2:
            return mask
        m = mask.astype(np.float32)

        abs_yaw = abs(yaw)

        # ── Layer 1: global yaw-dependent fade ──
        # 0–20°: full strength (1.0)
        # 20–40°: linear ramp down
        # 40°+: completely off (0.0)
        fade_start = 20.0
        fade_end = 40.0
        yaw_factor = 1.0 - float(np.clip(
            (abs_yaw - fade_start) / (fade_end - fade_start), 0.0, 1.0
        ))

        # ── Layer 2: directional column gradient (occluded side) ──
        if abs_yaw > 10.0:
            xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
            if yaw > 0:
                # Face turned right → left columns (high 1-xs) are occluded
                dir_atten = 1.0 - np.clip((1.0 - xs) * (abs_yaw / 50.0), 0.0, 0.85)
            else:
                # Face turned left → right columns (high xs) are occluded
                dir_atten = 1.0 - np.clip(xs * (abs_yaw / 50.0), 0.0, 0.85)
            m *= dir_atten

        m *= attenuation * yaw_factor
        return np.clip(m, 0, 255).astype(np.uint8)

    def apply_face_swap(
        self,
        target_frame: np.ndarray,
        target_landmarks: np.ndarray,
        runtime_hints: Optional[dict] = None,
        session: Optional[FaceSwapSession] = None,
    ) -> np.ndarray:
        session = session or self._default_session
        hints = runtime_hints or {}
        degraded_mode = bool(hints.get("degraded_mode", False))

        if not self.is_loaded or target_landmarks is None:
            return target_frame
        if target_frame.dtype != np.uint8:
            target_frame = np.clip(target_frame, 0, 255).astype(np.uint8)

        h, w = target_frame.shape[:2]
        frame_diag = float(np.hypot(h, w))
        pose = estimate_head_pose(target_landmarks, w, h)
        stability = landmark_stability_confidence(target_landmarks, session.prev_landmarks, frame_diag)
        session.stability_score = 0.5 * session.stability_score + 0.5 * stability
        confidence = float(np.clip(0.65 * pose.confidence + 0.35 * session.stability_score, 0.0, 1.0))

        # ── Dynamic pose gating ──
        # Hard yaw gate: beyond 40° the frontal Delaunay triangulation
        # produces degenerate geometry — don't even attempt the swap.
        abs_yaw = abs(pose.yaw)
        if abs_yaw > 40.0:
            session.prev_landmarks = target_landmarks.copy()
            session.prev_pose = pose
            return target_frame
        # Raised from 0.2 → 0.35: the old threshold let yaw≈35° through
        # (old confidence ~0.46) causing stretched triangles.  With the
        # new pose_module confidence curve, 0.35 corresponds to ~33° yaw.
        if confidence < 0.35:
            session.prev_landmarks = target_landmarks.copy()
            return target_frame
        if session.prev_pose is not None:
            pose = PoseEstimate(
                yaw=0.6 * pose.yaw + 0.4 * session.prev_pose.yaw,
                pitch=0.6 * pose.pitch + 0.4 * session.prev_pose.pitch,
                roll=0.6 * pose.roll + 0.4 * session.prev_pose.roll,
                confidence=confidence,
            )
        session.prev_pose = pose
        session.prev_landmarks = target_landmarks.copy()

        try:
            src_pts = self.source_landmarks[self.used_indices].astype(np.float32)
            dst_pts = target_landmarks[self.used_indices].astype(np.float32)
        except IndexError:
            return target_frame
        dst_oval_pts = np.array([target_landmarks[i] for i in FACE_OVAL_INDICES if i < len(target_landmarks)], dtype=np.int32)
        if len(dst_oval_pts) < 3:
            return target_frame

        x_min, y_min, w_roi, h_roi = cv2.boundingRect(dst_oval_pts)
        pad = int(max(w_roi, h_roi) * 0.15)
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_min + w_roi + 2 * pad)
        y_max = min(h, y_min + h_roi + 2 * pad)
        w_roi = x_max - x_min
        h_roi = y_max - y_min
        if w_roi <= 0 or h_roi <= 0:
            return target_frame

        target_roi = target_frame[y_min:y_max, x_min:x_max].copy()
        warped_roi = np.zeros_like(target_roi)
        triangles_warped = 0
        for t_idx, (ia, ib, ic) in enumerate(self.source_triangles):
            src_tri = self.source_triangle_meta[t_idx].src_tri
            dst_tri = np.array([dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32).reshape(3, 2)
            if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
                continue
            if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
                continue
            dst_tri_roi = dst_tri - np.array([x_min, y_min], dtype=np.float32)
            bx, by, bw, bh = cv2.boundingRect(np.int32(dst_tri_roi))
            bx = max(0, bx)
            by = max(0, by)
            bw = min(bw, w_roi - bx)
            bh = min(bh, h_roi - by)
            if bw <= 0 or bh <= 0:
                continue
            mask = np.zeros((bh, bw), dtype=np.uint8)
            dst_crop = (dst_tri_roi - np.array([bx, by], dtype=np.float32)).astype(np.float32).reshape(3, 2)
            cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)
            warp_mat = cv2.getAffineTransform(src_tri, dst_crop)
            warped_patch = cv2.warpAffine(
                self.source_image,
                warp_mat,
                (bw, bh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            roi_patch = warped_roi[by:by + bh, bx:bx + bw]
            warped_roi[by:by + bh, bx:bx + bw] = np.where(mask[..., None] == 255, warped_patch, roi_patch)
            triangles_warped += 1
        if triangles_warped == 0:
            return target_frame

        # ── Black-gap fallback ──
        # Skipped/degenerate triangles leave zero-valued (black) pixels in
        # warped_roi.  Without this fallback these black holes bleed through
        # alpha blending at 75%+ opacity, producing the black band artifacts
        # visible in rotated-face screenshots.  Filling with target_roi
        # pixels makes gaps invisible.
        black_mask = (warped_roi.sum(axis=2) == 0)
        warped_roi[black_mask] = target_roi[black_mask]

        dst_oval_pts_roi = dst_oval_pts - np.array([x_min, y_min], dtype=np.int32)
        clone_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        hull = cv2.convexHull(np.array(dst_oval_pts_roi, dtype=np.int32))
        cv2.fillConvexPoly(clone_mask, hull, 255)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w_roi // 35), max(3, h_roi // 35)))
        clone_mask = cv2.erode(clone_mask, erode_kernel, iterations=1)
        clone_mask = cv2.GaussianBlur(clone_mask, (21, 21), 7.0)

        attenuation = 0.75 + 0.25 * confidence
        clone_mask = self._pose_aware_mask(clone_mask, pose.yaw, attenuation)
        mask_float = clone_mask.astype(np.float32)
        if session.prev_mask_alpha is not None and session.prev_mask_alpha.shape == mask_float.shape:
            mask_float = 0.62 * mask_float + 0.38 * session.prev_mask_alpha
        session.prev_mask_alpha = mask_float.copy()
        clone_mask = np.clip(mask_float, 0, 255).astype(np.uint8)

        if not degraded_mode:
            warped_roi = _lab_color_transfer(warped_roi, target_roi, clone_mask)
        center_x = int(np.clip(np.mean(dst_oval_pts_roi[:, 0]), 1, w_roi - 2))
        center_y = int(np.clip(np.mean(dst_oval_pts_roi[:, 1]), 1, h_roi - 2))

        # Disable seamlessClone to make it obvious that it is a different face
        do_clone = False # (not degraded_mode) and confidence > 0.45
        if do_clone:
            try:
                blended_roi = cv2.seamlessClone(
                    warped_roi,
                    target_roi,
                    clone_mask,
                    (center_x, center_y),
                    cv2.NORMAL_CLONE,
                )
            except cv2.error:
                blended_roi = _alpha_blend(warped_roi, target_roi, clone_mask, alpha_gain=attenuation)
        else:
            blended_roi = _alpha_blend(warped_roi, target_roi, clone_mask, alpha_gain=attenuation)

        try:
            n_lm = len(target_landmarks)
            # Combine inner + outer lip indices for full mouth coverage.
            # The inner ring alone leaves the lip-skin boundary exposed,
            # causing a hard seam when the mouth opens wide.
            all_mouth_indices = list(set(INNER_LIP_INDICES + OUTER_LIP_INDICES))
            mouth_pts = np.array(
                [target_landmarks[i] for i in all_mouth_indices if i < n_lm],
                dtype=np.int32,
            )
            if len(mouth_pts) >= 10:
                mouth_pts_roi = mouth_pts - np.array([x_min, y_min], dtype=np.int32)
                mouth_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
                # Use convexHull so that the entire oral cavity is covered
                # even when individual landmark polygons leave gaps.
                hull = cv2.convexHull(mouth_pts_roi)
                cv2.fillConvexPoly(mouth_mask, hull, 255)
                # Dilate slightly to extend coverage into the lip-skin
                # transition zone (~3% of ROI width).
                dilate_sz = max(3, w_roi // 35)
                dilate_k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (dilate_sz, dilate_sz)
                )
                mouth_mask = cv2.dilate(mouth_mask, dilate_k, iterations=1)
                # Soft feathering: (5,5)/1.5 was only ~3px — clearly visible
                # at 640x480.  (15,15)/4.0 gives ~12px feather.
                mouth_mask_f = cv2.GaussianBlur(
                    mouth_mask.astype(np.float32) / 255.0, (15, 15), 4.0
                )
                mouth_strength = float(np.clip(0.6 + 0.4 * confidence, 0.35, 1.0))
                alpha_3 = (mouth_mask_f * mouth_strength)[..., np.newaxis]
                blended_roi = (
                    target_roi.astype(np.float32) * alpha_3
                    + blended_roi.astype(np.float32) * (1.0 - alpha_3)
                )
                blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
        except Exception:
            pass

        result_frame = target_frame.copy()
        result_frame[y_min:y_max, x_min:x_max] = blended_roi
        return result_frame


face_swap_engine = FaceSwapEngine()
