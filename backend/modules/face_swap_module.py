"""
Face Swap Module — Static image face swapping engine.

Reuses the project's existing landmark detection (detect_face_landmarks),
Delaunay triangulation, and geometric_warp / _prepare_warp utilities from
warping_module.py.  No new landmark frameworks are introduced.

Pipeline
--------
1. load_source_face()          — decode source image from bytes / path
2. extract_source_landmarks()  — detect 468 MediaPipe FaceMesh landmarks
3. create_face_mask()          — convex-hull mask from face-oval indices
4. warp_source_to_target()     — piecewise affine warp via Delaunay
5. blend_face()                — cv2.seamlessClone + Gaussian feathering
6. apply_face_swap()           — full pipeline orchestrator
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import Delaunay

try:
    from modules.warping_module import (
        detect_face_landmarks,
        geometric_warp,
        _corners,
        triangle_area,
        _has_duplicate_vertices,
        _estimate_head_pose,
    )
except ModuleNotFoundError:
    from backend.modules.warping_module import (
        detect_face_landmarks,
        geometric_warp,
        _corners,
        triangle_area,
        _has_duplicate_vertices,
        _estimate_head_pose,
    )

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe FACEMESH_FACE_OVAL indices — defines the inner face region to swap.
# Excludes hair, ears, and background.
# ══════════════════════════════════════════════════════════════════════════════

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]


class FaceSwapError(Exception):
    """Raised when face swap cannot complete."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# 1. load_source_face
# ──────────────────────────────────────────────────────────────────────────────

def load_source_face(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR OpenCV image.

    Parameters
    ----------
    image_bytes : bytes
        Raw image data (JPEG, PNG, WEBP, etc.).

    Returns
    -------
    np.ndarray
        Decoded BGR image.

    Raises
    ------
    FaceSwapError
        If the image cannot be decoded.
    """
    if not image_bytes:
        raise FaceSwapError("Empty image data provided.")

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FaceSwapError("Could not decode source image bytes.")
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 2. extract_source_landmarks
# ──────────────────────────────────────────────────────────────────────────────

def extract_source_landmarks(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect 468 MediaPipe FaceMesh landmarks on the given image.

    Uses the project's existing ``detect_face_landmarks()`` from
    ``warping_module.py``.

    Parameters
    ----------
    image_bgr : np.ndarray
        BGR image containing exactly one face.

    Returns
    -------
    np.ndarray
        (N, 2) float32 pixel-coordinate landmarks.

    Raises
    ------
    FaceSwapError
        If no face is detected or landmark extraction fails.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise FaceSwapError("Invalid image for landmark extraction.")

    lm = detect_face_landmarks(image_bgr)
    if lm is None or len(lm) < 100:
        raise FaceSwapError(
            "No face detected in the image. Ensure the image contains "
            "a clearly visible face."
        )
    return lm


# ──────────────────────────────────────────────────────────────────────────────
# 3. create_face_mask
# ──────────────────────────────────────────────────────────────────────────────

def create_face_mask(
    image_shape: tuple[int, int, int],
    landmarks: np.ndarray,
    feather_amount: int = 15,
) -> np.ndarray:
    """
    Create a smooth face-region mask from the MediaPipe face oval landmarks.

    The mask covers the inner face (cheeks, eyes, nose, mouth) and excludes
    hair, ears, and background.  Edges are feathered with Gaussian blur for
    smooth blending.

    Parameters
    ----------
    image_shape : tuple
        (H, W, C) shape of the target image.
    landmarks : np.ndarray
        (N, 2) pixel-coordinate landmarks.
    feather_amount : int
        Gaussian blur kernel size for edge feathering (must be odd).

    Returns
    -------
    np.ndarray
        Single-channel uint8 mask (0–255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Collect face-oval points, clamping to available landmarks
    n_lm = landmarks.shape[0]
    oval_pts = []
    for idx in FACE_OVAL_INDICES:
        if idx < n_lm:
            oval_pts.append(landmarks[idx].astype(np.int32))

    if len(oval_pts) < 3:
        logger.warning("create_face_mask: insufficient oval points (%d)", len(oval_pts))
        return mask

    hull = cv2.convexHull(np.array(oval_pts, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # Gaussian feathering for smooth edges
    k = feather_amount if feather_amount % 2 == 1 else feather_amount + 1
    k = max(3, k)
    mask = cv2.GaussianBlur(mask, (k, k), k * 0.4)

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# 4. warp_source_to_target
# ──────────────────────────────────────────────────────────────────────────────

def warp_source_to_target(
    source_bgr: np.ndarray,
    source_lm: np.ndarray,
    target_bgr: np.ndarray,
    target_lm: np.ndarray,
) -> np.ndarray:
    """
    Piecewise affine warp of the source face onto the target face geometry.

    Uses Delaunay triangulation on the target landmarks, then warps each
    source triangle to the corresponding target triangle.  Only the face-oval
    subset of landmarks is used for the warp to restrict the swap to the
    inner face region.

    Parameters
    ----------
    source_bgr : np.ndarray
        Source face image (BGR).
    source_lm : np.ndarray
        (N, 2) source landmarks (pixel coords).
    target_bgr : np.ndarray
        Target face image (BGR).
    target_lm : np.ndarray
        (N, 2) target landmarks (pixel coords).

    Returns
    -------
    np.ndarray
        Warped source face image on a canvas the size of target_bgr.
    """
    h, w = target_bgr.shape[:2]
    warped = np.zeros_like(target_bgr)

    # Use only face-oval landmarks + a few interior anchor points for
    # denser triangulation within the face region.
    # Interior anchors: nose, eyes, mouth corners, chin
    interior_indices = [
        # Nose ridge and tip
        1, 2, 4, 5, 6, 19, 94, 168,
        # Left eye
        33, 133, 160, 158, 153, 144, 159, 145,
        # Right eye
        362, 263, 387, 385, 380, 373, 386, 374,
        # Mouth outer
        61, 291, 0, 17, 78, 308, 13, 14, 87, 317,
        # Mouth inner
        82, 312, 311, 310, 415, 324, 318, 402, 95, 88,
        # Cheeks
        205, 425, 50, 280, 117, 346, 118, 347,
        # Under-eye
        111, 340,
    ]

    # Merge face oval + interior, deduplicate, clamp to available landmarks
    n_lm = min(source_lm.shape[0], target_lm.shape[0])
    used_indices = sorted(set(
        [i for i in FACE_OVAL_INDICES if i < n_lm] +
        [i for i in interior_indices if i < n_lm]
    ))

    if len(used_indices) < 10:
        logger.warning("warp_source_to_target: too few landmark matches (%d)",
                        len(used_indices))
        return warped

    src_pts = source_lm[used_indices].astype(np.float32)
    dst_pts = target_lm[used_indices].astype(np.float32)

    # Delaunay on target points
    try:
        tri = Delaunay(dst_pts)
    except Exception as exc:
        logger.error("Delaunay failed in face swap: %s", exc)
        return warped

    for ia, ib, ic in tri.simplices:
        src_tri = np.array(
            [src_pts[ia], src_pts[ib], src_pts[ic]], dtype=np.float32
        ).reshape(3, 2)
        dst_tri = np.array(
            [dst_pts[ia], dst_pts[ib], dst_pts[ic]], dtype=np.float32
        ).reshape(3, 2)

        # Skip degenerate / duplicate-vertex triangles
        if _has_duplicate_vertices(src_tri) or _has_duplicate_vertices(dst_tri):
            continue
        if triangle_area(src_tri) < 1e-3 or triangle_area(dst_tri) < 1e-3:
            continue

        # Bounding rect on destination
        r = cv2.boundingRect(dst_tri)
        bx, by, bw, bh = r
        bx = max(bx, 0)
        by = max(by, 0)
        bw = min(bw, w - bx)
        bh = min(bh, h - by)
        if bw <= 0 or bh <= 0:
            continue

        try:
            # Triangle mask within bounding rect
            mask = np.zeros((bh, bw), dtype=np.uint8)
            dst_crop = (dst_tri - [bx, by]).astype(np.float32).reshape(3, 2)
            src_crop = src_tri.astype(np.float32).reshape(3, 2)
            cv2.fillConvexPoly(mask, np.int32(dst_crop), 255)

            # Affine warp: source triangle → destination triangle
            warp_mat = cv2.getAffineTransform(src_crop, dst_crop)
            warped_patch = cv2.warpAffine(
                source_bgr,
                warp_mat,
                (bw, bh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Composite onto warped canvas
            roi = warped[by: by + bh, bx: bx + bw]
            blended = np.where(mask[..., None] == 255, warped_patch, roi)
            warped[by: by + bh, bx: bx + bw] = blended
        except Exception as tri_exc:
            logger.debug("Face swap triangle warp failed: %s", tri_exc)
            continue

    return warped


# ──────────────────────────────────────────────────────────────────────────────
# 5. blend_face
# ──────────────────────────────────────────────────────────────────────────────

def blend_face(
    warped_source: np.ndarray,
    target_bgr: np.ndarray,
    target_lm: np.ndarray,
    feather_amount: int = 15,
) -> np.ndarray:
    """
    Blend the warped source face onto the target image using
    ``cv2.seamlessClone`` with Gaussian-feathered masks.

    Parameters
    ----------
    warped_source : np.ndarray
        Source face warped to the target geometry (same size as target_bgr).
    target_bgr : np.ndarray
        Original target image.
    target_lm : np.ndarray
        (N, 2) target face landmarks (pixel coords).
    feather_amount : int
        Gaussian blur kernel size for mask feathering.

    Returns
    -------
    np.ndarray
        Final composited image with the swapped face.
    """
    h, w = target_bgr.shape[:2]

    # Build a smooth face mask on the target
    mask = create_face_mask(target_bgr.shape, target_lm, feather_amount)

    # Compute the centroid of the face region for seamlessClone
    n_lm = target_lm.shape[0]
    oval_pts = [target_lm[i] for i in FACE_OVAL_INDICES if i < n_lm]
    if len(oval_pts) < 3:
        # Fallback: simple alpha blend
        logger.warning("blend_face: insufficient oval points; using alpha blend")
        mask_f = mask.astype(np.float32) / 255.0
        mask_3 = mask_f[..., np.newaxis]
        result = (
            warped_source.astype(np.float32) * mask_3
            + target_bgr.astype(np.float32) * (1.0 - mask_3)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    oval_arr = np.array(oval_pts, dtype=np.float32)
    center_x = int(np.mean(oval_arr[:, 0]))
    center_y = int(np.mean(oval_arr[:, 1]))

    # Clamp center to valid range for seamlessClone
    center_x = max(1, min(w - 2, center_x))
    center_y = max(1, min(h - 2, center_y))
    center = (center_x, center_y)

    # The mask for seamlessClone needs to be strictly inside the image
    # and should be a binary-ish mask (seamlessClone handles gradients internally)
    clone_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(oval_pts, dtype=np.int32))
    cv2.fillConvexPoly(clone_mask, hull, 255)

    # Erode slightly to avoid boundary artifacts
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, w // 80), max(3, h // 80)),
    )
    clone_mask = cv2.erode(clone_mask, erode_kernel, iterations=1)

    # Gaussian feather for smooth blending
    k = feather_amount if feather_amount % 2 == 1 else feather_amount + 1
    k = max(3, k)
    clone_mask = cv2.GaussianBlur(clone_mask, (k, k), k * 0.3)

    try:
        # cv2.seamlessClone for Poisson blending — produces the most
        # natural skin-tone matching
        result = cv2.seamlessClone(
            warped_source,
            target_bgr,
            clone_mask,
            center,
            cv2.NORMAL_CLONE,
        )
    except cv2.error as exc:
        # Fallback to manual alpha blending if seamlessClone fails
        logger.warning("seamlessClone failed (%s); falling back to alpha blend", exc)
        mask_f = mask.astype(np.float32) / 255.0
        mask_3 = mask_f[..., np.newaxis]
        result = (
            warped_source.astype(np.float32) * mask_3
            + target_bgr.astype(np.float32) * (1.0 - mask_3)
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. apply_face_swap  — Full pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def apply_face_swap(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    feather_amount: int = 15,
) -> np.ndarray:
    """
    Full face swap pipeline: detect landmarks on both images, warp the source
    face onto the target geometry, and blend with seamless cloning.

    Parameters
    ----------
    source_bgr : np.ndarray
        Source face image (BGR).  Must contain exactly one clearly visible face.
    target_bgr : np.ndarray
        Target image (BGR).  Must contain exactly one clearly visible face.
    feather_amount : int
        Gaussian feathering for mask edges.

    Returns
    -------
    np.ndarray
        Final swapped image (same dimensions as target_bgr).

    Raises
    ------
    FaceSwapError
        If either image is invalid, no face detected, or the swap fails.
    """
    if source_bgr is None or source_bgr.size == 0:
        raise FaceSwapError("Source image is empty or invalid.")
    if target_bgr is None or target_bgr.size == 0:
        raise FaceSwapError("Target image is empty or invalid.")

    # Step 1 & 2: Extract landmarks from both images
    logger.info("Face swap: extracting source landmarks...")
    source_lm = extract_source_landmarks(source_bgr)

    logger.info("Face swap: extracting target landmarks...")
    target_lm = extract_source_landmarks(target_bgr)

    # Step 3: Warp source face to target geometry
    logger.info("Face swap: warping source → target...")
    warped_source = warp_source_to_target(
        source_bgr, source_lm, target_bgr, target_lm
    )

    # Verify warp produced meaningful output
    if np.max(warped_source) < 5:
        raise FaceSwapError(
            "Face warping produced an empty result.  The faces may be at "
            "incompatible angles or sizes."
        )

    # Step 4: Blend with seamless cloning
    logger.info("Face swap: blending faces...")
    result = blend_face(warped_source, target_bgr, target_lm, feather_amount)

    logger.info("Face swap: complete.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7. Realtime Face Swap — Source cache + per-frame warp with pre-smoothed
#    target landmarks (no per-frame source detection).
# ══════════════════════════════════════════════════════════════════════════════

class SourceFaceCache:
    """
    Holds a pre-loaded source face image and its landmarks so they are
    computed only once — NOT on every webcam frame.

    Usage::

        cache = SourceFaceCache()
        cache.load(source_bgr)          # once, when user uploads / selects
        ...
        result = realtime_face_swap(frame, target_lm, cache)  # every frame
    """

    def __init__(self) -> None:
        self.source_bgr: Optional[np.ndarray] = None
        self.source_lm: Optional[np.ndarray] = None
        self.is_loaded: bool = False

    def load(self, source_bgr: np.ndarray) -> None:
        """
        Load and cache the source face image + landmarks.

        Parameters
        ----------
        source_bgr : np.ndarray
            BGR image of the source face.

        Raises
        ------
        FaceSwapError
            If no face is detected in the source image.
        """
        if source_bgr is None or source_bgr.size == 0:
            raise FaceSwapError("Source image is empty or invalid.")

        lm = extract_source_landmarks(source_bgr)
        self.source_bgr = source_bgr.copy()
        self.source_lm = lm.copy()
        self.is_loaded = True
        logger.info(
            "SourceFaceCache: loaded source (%dx%d), %d landmarks",
            source_bgr.shape[1], source_bgr.shape[0], lm.shape[0],
        )

    def load_from_bytes(self, image_bytes: bytes) -> None:
        """
        Convenience: decode bytes and load.

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes (JPEG / PNG / WEBP).
        """
        img = load_source_face(image_bytes)
        self.load(img)

    def clear(self) -> None:
        """Unload the cached source face."""
        self.source_bgr = None
        self.source_lm = None
        self.is_loaded = False
        logger.info("SourceFaceCache: cleared")


def realtime_face_swap(
    frame_bgr: np.ndarray,
    target_landmarks: np.ndarray,
    source_cache: SourceFaceCache,
    feather_amount: int = 11,
) -> np.ndarray:
    """
    Per-frame face swap for the realtime webcam pipeline.

    - Target landmarks come from the live pipeline's EMA smoother (stable).
    - Source landmarks are read from the pre-computed cache (not re-detected).
    - Head-pose is estimated to attenuate the swap during extreme rotations.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Current webcam frame (BGR).
    target_landmarks : np.ndarray
        (N, 2) EMA-smoothed target face landmarks (pixel coords).
    source_cache : SourceFaceCache
        Pre-loaded source face and landmarks.
    feather_amount : int
        Gaussian feathering for mask edges (smaller = sharper blend).

    Returns
    -------
    np.ndarray
        Frame with the source face swapped onto the target.
    """
    if not source_cache.is_loaded:
        return frame_bgr

    if target_landmarks is None or len(target_landmarks) < 100:
        return frame_bgr

    src_bgr = source_cache.source_bgr
    src_lm = source_cache.source_lm

    # ── Pose-aware attenuation ──────────────────────────────────────────
    # Estimate head pose of the target face.  When yaw/pitch is extreme
    # the piecewise affine warp produces visible seams — reduce blend
    # opacity to degrade gracefully rather than showing artifacts.
    h, w = frame_bgr.shape[:2]
    try:
        yaw, pitch, roll = _estimate_head_pose(target_landmarks, w, h)
    except Exception:
        yaw, pitch, roll = 0.0, 0.0, 0.0

    # Attenuation curve: full strength within ±25°, fades to 0 at ±50°
    max_angle = max(abs(yaw), abs(pitch))
    if max_angle > 50.0:
        # Too extreme — skip swap entirely to avoid ugly artifacts
        return frame_bgr
    elif max_angle > 25.0:
        # Gradual fade-out between 25° and 50°
        pose_alpha = 1.0 - (max_angle - 25.0) / 25.0
    else:
        pose_alpha = 1.0

    # ── Warp source face onto target geometry ───────────────────────────
    try:
        warped = warp_source_to_target(src_bgr, src_lm, frame_bgr, target_landmarks)
    except Exception as exc:
        logger.debug("realtime_face_swap warp failed: %s", exc)
        return frame_bgr

    # Verify warp produced meaningful output
    if np.max(warped) < 5:
        return frame_bgr

    # ── Blend via seamlessClone ──────────────────────────────────────────
    try:
        blended = blend_face(warped, frame_bgr, target_landmarks, feather_amount)
    except Exception as exc:
        logger.debug("realtime_face_swap blend failed: %s", exc)
        return frame_bgr

    # ── Apply pose attenuation ──────────────────────────────────────────
    if pose_alpha < 1.0:
        blended = cv2.addWeighted(
            blended, pose_alpha,
            frame_bgr, 1.0 - pose_alpha,
            0,
        )

    return blended
