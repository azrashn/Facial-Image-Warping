"""
warping_module.py — Face Landmark Detection & Delaunay-based Geometric Warping
================================================================================
Production-grade implementation with:
  - MediaPipe FaceLandmarker (Tasks API) 478-point landmark extraction
  - Vectorial smile / thin-face / eyebrow-raise transformations
  - Artifact-free Delaunay triangulation with boundary point injection
  - Per-triangle affine warp with bilinear interpolation & convex-poly blending

Zero placeholders.  Every function contains real linear-algebra.
"""

import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple

import mediapipe as mp

# Access Tasks API classes via attribute path (avoids broken subpackage import on Python 3.14)
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

# ── Model path resolution ──
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ── Singleton FaceLandmarker ──
_face_landmarker = None


def _get_face_landmarker():
    """Singleton FaceLandmarker initializer using the Tasks API."""
    global _face_landmarker
    if _face_landmarker is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"MediaPipe model not found at: {_MODEL_PATH}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', "
                "'face_landmarker.task')\""
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


# ============================================================
# MediaPipe Landmark Index Groups
# ============================================================
# Mouth corners (left / right)
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
# Extended mouth region for smoother smile deformation
MOUTH_OUTER_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
]

# Jawline contour (chin → ear, both sides)
JAWLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Left eyebrow
LEFT_EYEBROW_INDICES = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
# Right eyebrow
RIGHT_EYEBROW_INDICES = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
EYEBROW_INDICES = LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES


# ============================================================
# 1.  Landmark Detection
# ============================================================
def detect_landmarks(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Detects 478 facial landmarks using MediaPipe FaceLandmarker (Tasks API).

    Args:
        image: BGR uint8 numpy array.

    Returns:
        List of (x, y) pixel coordinates.  Empty list if no face found.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image from numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    landmarker = _get_face_landmarker()
    result = landmarker.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return []

    face = result.face_landmarks[0]
    landmarks = []
    for lm in face:
        # Normalized [0,1] → pixel coords, clamped to image bounds
        px = int(min(max(lm.x * w, 0), w - 1))
        py = int(min(max(lm.y * h, 0), h - 1))
        landmarks.append((px, py))

    return landmarks  # len == 478


# ============================================================
# 2.  Transformation Displacement Vectors
# ============================================================
def _compute_face_centroid(landmarks: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Returns the (cx, cy) centroid of all landmark points."""
    pts = np.array(landmarks, dtype=np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _build_displacement_map(
    landmarks: List[Tuple[int, int]],
    operation: str,
    intensity: float,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Builds per-landmark displacement vectors  (dx, dy)  for the requested operation.

    Args:
        landmarks:   468 (x,y) points.
        operation:   'smile' | 'thin_face' | 'eyebrow_raise'
        intensity:   0.0 – 1.0 normalized strength.
        image_shape: (H, W) of the source image.

    Returns:
        np.ndarray of shape (N, 2) — displacement for each landmark.
    """
    n = len(landmarks)
    displacements = np.zeros((n, 2), dtype=np.float64)
    h, w = image_shape

    # Scale factor so deformations are proportional to face size
    pts = np.array(landmarks, dtype=np.float64)
    face_width = pts[:, 0].max() - pts[:, 0].min()
    face_height = pts[:, 1].max() - pts[:, 1].min()
    scale = max(face_width, face_height)

    if operation == "smile":
        # ── Mouth corners → pull outward + upward ──
        cx, cy = _compute_face_centroid(landmarks)

        for idx in MOUTH_OUTER_INDICES:
            if idx >= n:
                continue
            px, py = landmarks[idx]
            # Vector from centroid to point (outward direction)
            vx = px - cx
            vy = py - cy
            length = max(np.sqrt(vx * vx + vy * vy), 1e-6)
            # Normalize
            vx /= length
            vy /= length

            # Outward pull + upward lift
            mag = scale * 0.06 * intensity
            displacements[idx, 0] = vx * mag * 0.5   # slight outward
            displacements[idx, 1] = -mag * 0.7         # upward (negative Y)

        # Extra: pull the two main corners more aggressively
        for corner_idx in [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT]:
            if corner_idx >= n:
                continue
            px, py = landmarks[corner_idx]
            vx = px - cx
            direction = 1.0 if vx >= 0 else -1.0
            mag = scale * 0.10 * intensity
            displacements[corner_idx, 0] = direction * mag  # outward
            displacements[corner_idx, 1] = -mag * 0.6        # upward

    elif operation == "thin_face":
        # ── Jawline points → shrink toward face centroid ──
        cx, cy = _compute_face_centroid(landmarks)

        for idx in JAWLINE_INDICES:
            if idx >= n:
                continue
            px, py = landmarks[idx]
            # Vector from point toward centroid
            vx = cx - px
            vy = cy - py
            length = max(np.sqrt(vx * vx + vy * vy), 1e-6)
            vx /= length
            vy /= length

            mag = scale * 0.08 * intensity
            displacements[idx, 0] = vx * mag
            displacements[idx, 1] = vy * mag * 0.3  # mostly horizontal shrink

    elif operation == "eyebrow_raise":
        # ── Eyebrow points → shift upward (−Y) ──
        mag = scale * 0.07 * intensity
        for idx in EYEBROW_INDICES:
            if idx >= n:
                continue
            displacements[idx, 0] = 0.0
            displacements[idx, 1] = -mag

    return displacements


# ============================================================
# 3.  Delaunay Triangulation with Boundary Injection
# ============================================================
def _add_boundary_points(
    points: np.ndarray, h: int, w: int
) -> np.ndarray:
    """
    Appends 8 boundary points (4 corners + 4 edge midpoints) to prevent
    black-edge artifacts during warping.
    """
    boundary = np.array([
        [0, 0],             # top-left
        [w - 1, 0],         # top-right
        [0, h - 1],         # bottom-left
        [w - 1, h - 1],     # bottom-right
        [w // 2, 0],        # top-mid
        [w - 1, h // 2],    # right-mid
        [w // 2, h - 1],    # bottom-mid
        [0, h // 2],        # left-mid
    ], dtype=np.float64)

    return np.vstack([points, boundary])


def _compute_delaunay(points: np.ndarray) -> np.ndarray:
    """
    Computes Delaunay triangulation.

    Args:
        points: (N, 2) array of (x, y).

    Returns:
        (M, 3) array of triangle vertex indices.
    """
    tri = Delaunay(points)
    return tri.simplices  # shape (M, 3)


# ============================================================
# 4.  Per-Triangle Affine Warp
# ============================================================
def _warp_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
) -> None:
    """
    Warps a single triangle from src_img into dst_img using an affine transform.
    Uses cv2.warpAffine with bilinear interpolation and cv2.fillConvexPoly masking.

    Args:
        src_img: Source image (BGR).
        dst_img: Destination image (BGR), modified in-place.
        src_tri: (3, 2) float64 — source triangle vertices.
        dst_tri: (3, 2) float64 — destination triangle vertices.
    """
    # Bounding rects for both triangles
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))

    sx, sy, sw, sh = sr
    dx, dy, dw, dh = dr

    # Clamp to image boundaries
    img_h, img_w = src_img.shape[:2]
    sx = max(sx, 0); sy = max(sy, 0)
    sw = min(sw, img_w - sx); sh = min(sh, img_h - sy)
    dx = max(dx, 0); dy = max(dy, 0)
    dw = min(dw, img_w - dx); dh = min(dh, img_h - dy)

    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return

    # Triangle coords relative to their bounding rects
    src_tri_local = src_tri - np.array([sx, sy], dtype=np.float64)
    dst_tri_local = dst_tri - np.array([dx, dy], dtype=np.float64)

    # Crop source region
    src_crop = src_img[sy:sy + sh, sx:sx + sw].copy()
    if src_crop.size == 0:
        return

    # Affine transform matrix  (2×3)
    mat = cv2.getAffineTransform(
        np.float32(src_tri_local),
        np.float32(dst_tri_local),
    )

    # Warp with bilinear interpolation
    warped = cv2.warpAffine(
        src_crop,
        mat,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Create mask from destination triangle
    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.int32(dst_tri_local),
        255,
    )

    # Blend into destination using the mask
    mask_3ch = mask[:, :, np.newaxis].astype(np.float64) / 255.0
    dst_region = dst_img[dy:dy + dh, dx:dx + dw].astype(np.float64)

    # Ensure shapes match after clipping
    min_h = min(warped.shape[0], mask_3ch.shape[0], dst_region.shape[0])
    min_w = min(warped.shape[1], mask_3ch.shape[1], dst_region.shape[1])
    if min_h <= 0 or min_w <= 0:
        return

    warped = warped[:min_h, :min_w]
    mask_3ch = mask_3ch[:min_h, :min_w]
    dst_region = dst_region[:min_h, :min_w]

    blended = warped.astype(np.float64) * mask_3ch + dst_region * (1.0 - mask_3ch)
    dst_img[dy:dy + min_h, dx:dx + min_w] = np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# 5.  Main Warping Pipeline
# ============================================================
def apply_warping(
    image: np.ndarray,
    landmarks: List[Tuple[int, int]],
    operation: str,
    intensity: float,
) -> np.ndarray:
    """
    Full warping pipeline:
        1. Compute displacement vectors for the chosen transformation
        2. Build source & destination point sets (with boundary injection)
        3. Triangulate the source points (Delaunay)
        4. Warp each triangle from source → destination

    Args:
        image:      BGR uint8 source image.
        landmarks:  468 (x,y) pixel-coordinate landmarks.
        operation:  'smile' | 'thin_face' | 'eyebrow_raise'
        intensity:  0.0 – 1.0

    Returns:
        Warped BGR uint8 image (same size as input).
    """
    h, w = image.shape[:2]
    n_landmarks = len(landmarks)

    # 1. Displacement vectors
    displacements = _build_displacement_map(landmarks, operation, intensity, (h, w))

    # 2. Source points = original landmarks
    src_points = np.array(landmarks, dtype=np.float64)

    # 3. Destination points = landmarks + displacements
    dst_points = src_points + displacements[:n_landmarks]

    # Clamp destination points to image bounds
    dst_points[:, 0] = np.clip(dst_points[:, 0], 0, w - 1)
    dst_points[:, 1] = np.clip(dst_points[:, 1], 0, h - 1)

    # 4. Add boundary points (identical in src & dst → edges stay fixed)
    src_with_boundary = _add_boundary_points(src_points, h, w)
    # Boundary displacements are zero → same points appended to dst
    boundary_pts = src_with_boundary[n_landmarks:]
    dst_with_boundary = np.vstack([dst_points, boundary_pts])

    # 5. Delaunay on SOURCE points (triangulation topology)
    triangles = _compute_delaunay(src_with_boundary)

    # 6. Warp each triangle
    output = np.zeros_like(image)

    for tri_indices in triangles:
        i0, i1, i2 = tri_indices

        src_tri = np.array([
            src_with_boundary[i0],
            src_with_boundary[i1],
            src_with_boundary[i2],
        ], dtype=np.float64)

        dst_tri = np.array([
            dst_with_boundary[i0],
            dst_with_boundary[i1],
            dst_with_boundary[i2],
        ], dtype=np.float64)

        _warp_triangle(image, output, src_tri, dst_tri)

    return output
