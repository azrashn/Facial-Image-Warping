"""
Face Swap Module — Production-grade realtime face swapping engine.

Features:
  - Cached source face (landmarks + Delaunay triangulation computed once)
  - ROI-based piecewise affine warping
  - LAB color transfer for skin tone matching
  - Pose-aware swap gating (attenuate/disable at extreme angles)
  - Multi-stage edge feathering (erode + distance transform + Gaussian)
  - Mouth occlusion pass (preserves target teeth/tongue)
  - Per-frame statistics (FPS, pose, success rate)
  - Optional frame downscaling for performance

Architecture:
  A single global ``face_swap_engine`` instance is shared across all
  endpoints (WebSocket live, HTTP static, desktop OpenCV).  Never create
  multiple engine instances.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import Delaunay

try:
    from modules.warping_module import (
        detect_face_landmarks,
        _has_duplicate_vertices,
        triangle_area,
        estimate_head_pose,
        validate_landmarks,
    )
except ModuleNotFoundError:
    from backend.modules.warping_module import (
        detect_face_landmarks,
        _has_duplicate_vertices,
        triangle_area,
        estimate_head_pose,
        validate_landmarks,
    )

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Landmark index sets
# ═══════════════════════════════════════════════════════════════════════════════

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

# MediaPipe Inner Lip Contour — exact ordered loop for cv2.fillPoly.
# Traces the inner edge of the lips (the teeth/void boundary), NOT the
# outer lip surface.  Using an ordered contour with fillPoly avoids the
# convexHull inflation that causes double-lip / smudge artifacts.
INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

# Interior anchor points for dense triangulation
_INTERIOR_INDICES = [
    1, 2, 4, 5, 6, 19, 94, 168,                         # Nose
    33, 133, 160, 158, 153, 144, 159, 145,               # Left eye
    362, 263, 387, 385, 380, 373, 386, 374,               # Right eye
    61, 291, 0, 17, 78, 308, 13, 14, 87, 317,            # Mouth outer
    82, 312, 311, 310, 415, 324, 318, 402, 95, 88,       # Mouth inner
    205, 425, 50, 280, 117, 346, 118, 347,               # Cheeks
    111, 340,                                             # Under eye
]

# ═══════════════════════════════════════════════════════════════════════════════
# Pose gating thresholds
# ═══════════════════════════════════════════════════════════════════════════════

_POSE_FULL_SWAP_DEG = 25.0    # < this angle → full blend
_POSE_DISABLE_DEG = 50.0      # > this angle → swap disabled
_POSE_RANGE = _POSE_DISABLE_DEG - _POSE_FULL_SWAP_DEG


class FaceSwapError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Color correction utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _color_transfer_lab(
    source_roi: np.ndarray,
    target_roi: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Transfer colour statistics from *target_roi* to *source_roi* in LAB space.

    This makes the source face adopt the target's skin tone, lighting, and
    contrast so the swap blends naturally.

    Parameters
    ----------
    source_roi : np.ndarray
        BGR uint8 image — the warped source face pixels.
    target_roi : np.ndarray
        BGR uint8 image — the original target face pixels (same shape).
    mask : np.ndarray | None
        Optional single-channel uint8 mask.  If provided, statistics are
        computed only within the masked region.

    Returns
    -------
    np.ndarray
        Colour-corrected *source_roi* (BGR uint8).
    """
    if source_roi.size == 0 or target_roi.size == 0:
        return source_roi

    src_lab = cv2.cvtColor(source_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is not None and mask.any():
        mask_bool = mask > 127
        # Compute per-channel mean/std within the mask
        for ch in range(3):
            src_ch = src_lab[:, :, ch]
            tgt_ch = tgt_lab[:, :, ch]

            s_mean = float(np.mean(src_ch[mask_bool]))
            s_std = float(np.std(src_ch[mask_bool])) + 1e-6
            t_mean = float(np.mean(tgt_ch[mask_bool]))
            t_std = float(np.std(tgt_ch[mask_bool])) + 1e-6

            # Shift + scale source stats → target stats
            src_lab[:, :, ch] = (src_ch - s_mean) * (t_std / s_std) + t_mean
    else:
        for ch in range(3):
            s_mean = float(np.mean(src_lab[:, :, ch]))
            s_std = float(np.std(src_lab[:, :, ch])) + 1e-6
            t_mean = float(np.mean(tgt_lab[:, :, ch]))
            t_std = float(np.std(tgt_lab[:, :, ch])) + 1e-6
            src_lab[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

    src_lab = np.clip(src_lab, 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _compute_pose_blend_factor(
    yaw: float, pitch: float, roll: float,
) -> float:
    """Return a blend factor in [0, 1] based on head pose angles.

    - < 25° max angle → 1.0 (full swap)
    - 25°–50° → linear ramp down
    - > 50° → 0.0 (swap disabled)
    """
    max_angle = max(abs(yaw), abs(pitch), abs(roll))
    if max_angle < _POSE_FULL_SWAP_DEG:
        return 1.0
    if max_angle > _POSE_DISABLE_DEG:
        return 0.0
    return 1.0 - (max_angle - _POSE_FULL_SWAP_DEG) / _POSE_RANGE


def _build_feathered_mask(
    shape: Tuple[int, int],
    hull_pts: np.ndarray,
    erode_fraction: float = 0.04,
    blur_size: int = 21,
) -> np.ndarray:
    """Build a soft-edged face mask with multi-stage feathering.

    Pipeline:
      1. Fill convex hull → binary mask
      2. Erode inward (removes edge pixels touching the boundary)
      3. Distance transform → smooth gradient at boundary
      4. Gaussian blur → final feathering

    Returns a single-channel float32 mask in [0, 1].
    """
    h_roi, w_roi = shape
    # Step 1: binary mask
    mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_pts, 255)

    # Step 2: erode
    ek_w = max(3, int(w_roi * erode_fraction))
    ek_h = max(3, int(h_roi * erode_fraction))
    # Ensure odd kernel
    ek_w = ek_w | 1
    ek_h = ek_h | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek_w, ek_h))
    mask = cv2.erode(mask, kernel, iterations=1)

    # Step 3: distance transform for smooth boundary gradient
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(np.max(dist)) if np.max(dist) > 0 else 1.0
    # Normalize but keep the core at 1.0
    feather_width = max(3.0, min(max_dist * 0.3, 15.0))
    mask_f = np.clip(dist / feather_width, 0.0, 1.0).astype(np.float32)

    # Step 4: Gaussian blur
    blur_k = blur_size | 1  # ensure odd
    mask_f = cv2.GaussianBlur(mask_f, (blur_k, blur_k), blur_k * 0.3)
    return np.clip(mask_f, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Face Swap Engine
# ═══════════════════════════════════════════════════════════════════════════════

class FaceSwapEngine:
    """
    Production-grade realtime face swap engine.

    Caches source face, landmarks, and Delaunay triangulation so that
    per-frame work is limited to:
      1. Target landmark extraction (done externally)
      2. Piecewise affine warp (triangles)
      3. Colour transfer + seamless blending
      4. Mouth occlusion restore

    Thread safety: the engine is designed to be called from a single
    processing thread at a time (the executor thread in the WebSocket
    handler or the main loop in the desktop path).
    """

    def __init__(self) -> None:
        # ── Source cache (computed once per source upload) ──
        self.source_image: Optional[np.ndarray] = None
        self.source_landmarks: Optional[np.ndarray] = None
        self.source_triangles: Optional[np.ndarray] = None
        self.is_loaded: bool = False
        self.running: bool = False

        # Indices used for triangulation (cached)
        self.used_indices: Optional[list[int]] = None

        # Pre-computed source data for fast per-frame access
        self._src_pts_cached: Optional[np.ndarray] = None

        # ── Runtime statistics ──
        self._frame_count: int = 0
        self._swap_success_count: int = 0
        self._swap_skip_pose: int = 0
        self._swap_skip_error: int = 0
        self._last_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_blend_factor: float = 0.0
        self._last_process_ms: float = 0.0
        self._stats_log_interval: int = 100  # log summary every N frames

    # ── Source face management ────────────────────────────────────────────

    def process_source_image(self, image_bytes: bytes) -> None:
        """Decode source image, detect face, compute Delaunay, cache everything."""
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
        if not validate_landmarks(lm, min_count=100):
            raise FaceSwapError("No face detected in source image.")

        self.source_image = img
        self.source_landmarks = lm

        # Build index set: face oval + interior anchors
        n_lm = self.source_landmarks.shape[0]
        self.used_indices = sorted(set(
            [i for i in FACE_OVAL_INDICES if i < n_lm] +
            [i for i in _INTERIOR_INDICES if i < n_lm]
        ))

        src_pts = self.source_landmarks[self.used_indices].astype(np.float32)
        try:
            tri = Delaunay(src_pts)
            self.source_triangles = tri.simplices
        except Exception as exc:
            raise FaceSwapError(f"Delaunay triangulation failed on source face: {exc}")

        # Pre-cache source points for per-frame reuse
        self._src_pts_cached = src_pts.copy()

        # Reset statistics on new source
        self._frame_count = 0
        self._swap_success_count = 0
        self._swap_skip_pose = 0
        self._swap_skip_error = 0

        self.is_loaded = True
        logger.info(
            "[FACE_SWAP] Source loaded: %d triangles cached, %d indices used",
            len(self.source_triangles), len(self.used_indices),
        )

    def load_source_bgr(self, source_bgr: np.ndarray) -> None:
        """Load a source face from a BGR ndarray (for the desktop path)."""
        if source_bgr is None or source_bgr.size == 0:
            raise FaceSwapError("Empty source image array.")
        if source_bgr.dtype != np.uint8:
            source_bgr = np.clip(source_bgr, 0, 255).astype(np.uint8)
        # Encode to bytes and reuse the standard path
        ok, buf = cv2.imencode(".png", source_bgr)
        if not ok:
            raise FaceSwapError("Failed to encode source image.")
        self.process_source_image(buf.tobytes())

    # ── Core swap pipeline ────────────────────────────────────────────────

    def apply_face_swap(
        self,
        target_frame: np.ndarray,
        target_landmarks: np.ndarray,
        process_scale: float = 1.0,
    ) -> np.ndarray:
        """Apply face swap from cached source onto *target_frame*.

        Parameters
        ----------
        target_frame : np.ndarray
            BGR uint8 webcam/target frame.
        target_landmarks : np.ndarray
            (N, 2) float32 pixel landmarks of the target face.
        process_scale : float
            If < 1.0, downscale frame for processing and upscale result.
            Use 0.75 for ~1.8× speedup with minimal quality loss.

        Returns
        -------
        np.ndarray
            The frame with the face swapped (BGR uint8).
        """
        t_start = time.perf_counter()
        self._frame_count += 1

        if not self.is_loaded or not validate_landmarks(target_landmarks):
            logger.debug("[FACE_SWAP] Skipping: not loaded or invalid target landmarks")
            return target_frame

        # CRITICAL: Enforce uint8 on input frame
        if target_frame.dtype != np.uint8:
            target_frame = np.clip(target_frame, 0, 255).astype(np.uint8)

        h, w = target_frame.shape[:2]

        # ── Pose estimation & gating ──────────────────────────────────
        yaw, pitch, roll = estimate_head_pose(target_landmarks, w, h)
        self._last_pose = (yaw, pitch, roll)

        blend_factor = _compute_pose_blend_factor(yaw, pitch, roll)
        self._last_blend_factor = blend_factor

        if blend_factor < 0.01:
            self._swap_skip_pose += 1
            self._log_periodic_stats()
            return target_frame

        # ── Optional downscaling ──────────────────────────────────────
        if 0.0 < process_scale < 1.0:
            proc_w = max(160, int(w * process_scale))
            proc_h = max(120, int(h * process_scale))
            proc_frame = cv2.resize(target_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            sx, sy = proc_w / w, proc_h / h
            proc_landmarks = target_landmarks.copy()
            proc_landmarks[:, 0] *= sx
            proc_landmarks[:, 1] *= sy
            result = self._swap_core(proc_frame, proc_landmarks, blend_factor)
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            result = self._swap_core(target_frame, target_landmarks, blend_factor)

        self._swap_success_count += 1
        self._last_process_ms = (time.perf_counter() - t_start) * 1000.0
        self._log_periodic_stats()
        return result

    def _swap_core(
        self,
        target_frame: np.ndarray,
        target_landmarks: np.ndarray,
        blend_factor: float,
    ) -> np.ndarray:
        """Internal: full swap pipeline without scaling or pose checks."""
        h, w = target_frame.shape[:2]

        # Extract target points matching our triangulated subset
        try:
            src_pts = self._src_pts_cached
            dst_pts = target_landmarks[self.used_indices].astype(np.float32)
        except (IndexError, TypeError):
            logger.debug("[FACE_SWAP] IndexError accessing landmarks subset")
            self._swap_skip_error += 1
            return target_frame

        # ── Compute ROI bounding box ──────────────────────────────────
        dst_oval_pts = np.array(
            [target_landmarks[i] for i in FACE_OVAL_INDICES if i < len(target_landmarks)],
            dtype=np.int32
        )
        if len(dst_oval_pts) < 3:
            self._swap_skip_error += 1
            return target_frame

        x_min, y_min, w_roi, h_roi = cv2.boundingRect(dst_oval_pts)

        # Add padding
        pad = int(max(w_roi, h_roi) * 0.15)
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_min + w_roi + 2 * pad)
        y_max = min(h, y_min + h_roi + 2 * pad)
        w_roi = x_max - x_min
        h_roi = y_max - y_min

        if w_roi <= 0 or h_roi <= 0:
            self._swap_skip_error += 1
            return target_frame

        target_roi = target_frame[y_min:y_max, x_min:x_max].copy()
        warped_roi = np.zeros_like(target_roi)

        # ── Piecewise affine warp (triangles) ─────────────────────────
        triangles_warped = 0
        for ia, ib, ic in self.source_triangles:
            src_tri = np.array(
                [src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32
            ).reshape(3, 2)
            dst_tri = np.array(
                [dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32
            ).reshape(3, 2)

            if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
                continue
            if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
                continue

            # Shift dst_tri to ROI coordinates
            dst_tri_roi = dst_tri - np.array([x_min, y_min], dtype=np.float32)

            # CRITICAL: cv2.boundingRect needs int32
            r = cv2.boundingRect(np.int32(dst_tri_roi))
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
                dst_crop = (dst_tri_roi - np.array([bx, by], dtype=np.float32)).astype(
                    np.float32
                ).reshape(3, 2)
                src_crop = src_tri.astype(np.float32).reshape(3, 2)
                # CRITICAL: fillConvexPoly needs int32 points
                cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)

                warp_mat = cv2.getAffineTransform(src_crop, dst_crop)
                warped_patch = cv2.warpAffine(
                    self.source_image, warp_mat, (bw, bh),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )

                # Ensure warped_patch is uint8
                if warped_patch.dtype != np.uint8:
                    warped_patch = np.clip(warped_patch, 0, 255).astype(np.uint8)

                roi_patch = warped_roi[by: by + bh, bx: bx + bw]
                blended = np.where(mask[..., None] == 255, warped_patch, roi_patch)
                warped_roi[by: by + bh, bx: bx + bw] = blended
                triangles_warped += 1
            except Exception:
                continue

        if triangles_warped == 0:
            logger.debug("[FACE_SWAP] No triangles warped — returning original")
            self._swap_skip_error += 1
            return target_frame

        # ── LAB colour correction ─────────────────────────────────────
        # Build a rough mask for stats computation
        dst_oval_pts_roi = dst_oval_pts - np.array([x_min, y_min], dtype=np.int32)
        hull = cv2.convexHull(np.array(dst_oval_pts_roi, dtype=np.int32))

        color_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.fillConvexPoly(color_mask, hull, 255)

        try:
            warped_roi = _color_transfer_lab(warped_roi, target_roi, mask=color_mask)
        except Exception as exc:
            logger.debug("[FACE_SWAP] Color transfer failed (non-fatal): %s", exc)

        # ── Seamless cloning / alpha blending ─────────────────────────
        # Build feathered mask
        feathered_mask = _build_feathered_mask(
            (h_roi, w_roi), hull,
            erode_fraction=0.04, blur_size=21,
        )

        # Apply pose-aware blend factor
        if blend_factor < 1.0:
            feathered_mask = feathered_mask * blend_factor

        # Convert to uint8 for seamlessClone
        clone_mask = np.clip(feathered_mask * 255, 0, 255).astype(np.uint8)

        # Compute center for seamlessClone
        center_x = int(np.mean(dst_oval_pts_roi[:, 0]))
        center_y = int(np.mean(dst_oval_pts_roi[:, 1]))
        center_x = max(1, min(w_roi - 2, center_x))
        center_y = max(1, min(h_roi - 2, center_y))

        # Ensure correct dtypes
        if warped_roi.dtype != np.uint8:
            warped_roi = np.clip(warped_roi, 0, 255).astype(np.uint8)

        try:
            blended_roi = cv2.seamlessClone(
                warped_roi, target_roi, clone_mask,
                (center_x, center_y), cv2.NORMAL_CLONE,
            )
        except cv2.error:
            # Fallback: manual alpha blending using feathered mask
            mask_3 = feathered_mask[..., np.newaxis]
            blended_roi = (
                warped_roi.astype(np.float32) * mask_3
                + target_roi.astype(np.float32) * (1.0 - mask_3)
            )
            blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

        # ── Mouth occlusion pass ──────────────────────────────────────
        # Restore target's real inner-mouth pixels on top of the swap.
        try:
            n_lm = len(target_landmarks)
            inner_lip_pts = np.array(
                [target_landmarks[i] for i in INNER_LIP_INDICES if i < n_lm],
                dtype=np.int32,
            )
            if len(inner_lip_pts) >= 10:
                inner_lip_roi = inner_lip_pts - np.array([x_min, y_min], dtype=np.int32)
                mouth_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
                cv2.fillPoly(mouth_mask, [inner_lip_roi], 255)

                mouth_mask_f = mouth_mask.astype(np.float32) / 255.0
                mouth_mask_f = cv2.GaussianBlur(mouth_mask_f, (5, 5), 1.5)

                alpha_3 = mouth_mask_f[..., np.newaxis]
                blended_roi = (
                    target_roi.astype(np.float32) * alpha_3
                    + blended_roi.astype(np.float32) * (1.0 - alpha_3)
                )
                blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
        except Exception as exc:
            logger.debug("[FACE_SWAP] Mouth occlusion failed (non-fatal): %s", exc)

        # ── Composite result ──────────────────────────────────────────
        result_frame = target_frame.copy()
        result_frame[y_min:y_max, x_min:x_max] = blended_roi
        return result_frame

    # ── Statistics & logging ──────────────────────────────────────────────

    def _log_periodic_stats(self) -> None:
        """Log a summary every N frames to avoid spamming the log."""
        if self._frame_count % self._stats_log_interval == 0:
            success_rate = (
                self._swap_success_count / max(1, self._frame_count) * 100
            )
            yaw, pitch, roll = self._last_pose
            logger.info(
                "[FACE_SWAP] stats: frames=%d success_rate=%.1f%% "
                "pose=(yaw=%.1f pitch=%.1f roll=%.1f) blend=%.2f "
                "skipped_pose=%d skipped_error=%d last_ms=%.1f",
                self._frame_count, success_rate,
                yaw, pitch, roll, self._last_blend_factor,
                self._swap_skip_pose, self._swap_skip_error,
                self._last_process_ms,
            )

    @property
    def stats(self) -> dict:
        """Return current engine statistics as a dict."""
        return {
            "frame_count": self._frame_count,
            "swap_success_count": self._swap_success_count,
            "swap_skip_pose": self._swap_skip_pose,
            "swap_skip_error": self._swap_skip_error,
            "success_rate": self._swap_success_count / max(1, self._frame_count),
            "last_pose": {
                "yaw": self._last_pose[0],
                "pitch": self._last_pose[1],
                "roll": self._last_pose[2],
            },
            "last_blend_factor": self._last_blend_factor,
            "last_process_ms": self._last_process_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Global singleton — shared across all endpoints
# ═══════════════════════════════════════════════════════════════════════════════

face_swap_engine = FaceSwapEngine()
