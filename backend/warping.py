"""
warping.py — Geometric Image Warping Engine
============================================
DSP Project — Facial Image Warping (Group 14)

Core Algorithm:
  1. Define per-operation anchor points (source → target control points).
  2. Build a smooth, globally consistent displacement field using
     Radial Basis Function (RBF) interpolation from the sparse control
     points (no high-level cv2 functions used for core logic).
  3. Use INVERSE MAPPING: for every pixel (x, y) in the output canvas,
     compute where it should sample from in the input image.
  4. Sample the input with vectorized BILINEAR INTERPOLATION (pure NumPy)
     to produce sub-pixel-accurate, hole-free output.
  5. Optionally render a deformation-grid visualization.
  6. Compute full image-quality metrics (MSE, PSNR, SSIM) from scratch.
"""

import numpy as np
from PIL import Image
import io
import base64


# ---------------------------------------------------------------------------
# Section 1 — Control Point Definitions
# ---------------------------------------------------------------------------

def _get_control_points(operation: str, intensity: float, w: int, h: int):
    """
    Return (src_pts, dst_pts) as (N, 2) float arrays in **pixel** coordinates.
    All landmark positions are expressed as fractions of (w, h) and scaled.
    intensity ∈ [0, 1].
    """
    # Shortcuts
    cx, cy = w * 0.50, h * 0.50   # image centre
    t = intensity                   # shorthand

    # Anchor corners to keep the image boundary stable
    corners = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5],              [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ], dtype=np.float64) * [w, h]
    corner_dst = corners.copy()

    op = operation.lower()

    if op == "smile":
        # Pull mouth corners up and outward; push lip bottom down slightly
        src = np.array([
            [0.30, 0.68], [0.50, 0.72], [0.70, 0.68],   # mouth corners + centre-bottom
            [0.50, 0.62],                                  # upper lip centre
        ]) * [w, h]
        dst = np.array([
            [0.28, 0.64 - 0.04 * t], [0.50, 0.74 + 0.02 * t], [0.72, 0.64 - 0.04 * t],
            [0.50, 0.62],
        ]) * [w, h]

    elif op == "eyebrow_raise":
        src = np.array([
            [0.25, 0.33], [0.375, 0.31], [0.50, 0.32],   # left brow
            [0.625, 0.31],[0.75, 0.33],                    # right brow
        ]) * [w, h]
        dy = -0.04 * t
        dst = src.copy()
        dst[:, 1] += dy * h

    elif op == "lip_widen":
        src = np.array([
            [0.32, 0.66], [0.68, 0.66],   # mouth corners
            [0.50, 0.70],                  # lip bottom
        ]) * [w, h]
        dst = np.array([
            [0.32 - 0.04 * t, 0.66], [0.68 + 0.04 * t, 0.66],
            [0.50, 0.70],
        ]) * [w, h]

    elif op == "face_slim":
        # Push cheeks inward horizontally
        src = np.array([
            [0.18, 0.45], [0.18, 0.55], [0.18, 0.65],
            [0.82, 0.45], [0.82, 0.55], [0.82, 0.65],
        ]) * [w, h]
        dx = 0.04 * t
        dst = src.copy()
        dst[0:3, 0] += dx * w   # left cheek inward
        dst[3:6, 0] -= dx * w   # right cheek inward

    elif op == "aging":
        # Slight drooping of brows and mouth corners; cheeks sag
        src = np.array([
            [0.25, 0.33], [0.75, 0.33],   # brows
            [0.30, 0.68], [0.70, 0.68],   # mouth corners
            [0.20, 0.55], [0.80, 0.55],   # cheeks
        ]) * [w, h]
        dst = src.copy()
        dst[0:2, 1] += 0.025 * t * h
        dst[2:4, 1] += 0.020 * t * h
        dst[4:6, 1] += 0.030 * t * h

    elif op == "deaging":
        # Lift brows, tighten cheeks, lift mouth corners
        src = np.array([
            [0.25, 0.33], [0.75, 0.33],
            [0.30, 0.68], [0.70, 0.68],
            [0.20, 0.55], [0.80, 0.55],
        ]) * [w, h]
        dst = src.copy()
        dst[0:2, 1] -= 0.025 * t * h
        dst[2:4, 1] -= 0.015 * t * h
        dst[2:4, 0] += np.array([ 0.02 * t * w, -0.02 * t * w])
        dst[4:6, 1] -= 0.020 * t * h

    else:
        # Identity — no warp
        src = corners
        dst = corners

    # Concatenate user landmarks with fixed corner anchors
    src_all = np.vstack([src, corners])
    dst_all = np.vstack([dst, corner_dst])
    return src_all.astype(np.float64), dst_all.astype(np.float64)


# ---------------------------------------------------------------------------
# Section 2 — RBF Interpolation (Thin-Plate Spline variant)
# ---------------------------------------------------------------------------

def _rbf_kernel(r: np.ndarray) -> np.ndarray:
    """
    Thin-Plate Spline radial basis: φ(r) = r² · log(r + ε)
    C∞ everywhere, minimal bending energy → very smooth warp fields.
    """
    eps = 1e-10
    return (r ** 2) * np.log(r + eps)


def _solve_rbf(src_pts: np.ndarray, dst_pts: np.ndarray):
    """
    Solve the TPS system for x- and y-displacement independently.

    Returns (weights_x, weights_y, affine_x, affine_y) where
      affine = [a0, ax, ay]  (affine part of the spline)
    """
    N = len(src_pts)
    # (N×N) pairwise distances
    diff = src_pts[:, None, :] - src_pts[None, :, :]        # (N, N, 2)
    r = np.linalg.norm(diff, axis=2)                         # (N, N)
    K = _rbf_kernel(r)                                        # (N, N)

    # Polynomial part P: [1, x, y]
    P = np.hstack([np.ones((N, 1)), src_pts])                 # (N, 3)

    # Build block system  [ K  P ] [w]   [d]
    #                      [ P' 0 ] [a] = [0]
    top    = np.hstack([K, P])                                 # (N, N+3)
    bottom = np.hstack([P.T, np.zeros((3, 3))])                # (3, N+3)
    A = np.vstack([top, bottom])                               # (N+3, N+3)

    # Displacements (target − source)
    disp = dst_pts - src_pts                                   # (N, 2)

    rhs_x = np.concatenate([disp[:, 0], np.zeros(3)])
    rhs_y = np.concatenate([disp[:, 1], np.zeros(3)])

    # Regularization for numerical stability
    A += np.eye(N + 3) * 1e-6

    sol_x = np.linalg.solve(A, rhs_x)
    sol_y = np.linalg.solve(A, rhs_y)

    return sol_x[:N], sol_x[N:], sol_y[:N], sol_y[N:]


def _evaluate_rbf(query_pts: np.ndarray,
                  src_pts:   np.ndarray,
                  wx: np.ndarray, ax: np.ndarray,
                  wy: np.ndarray, ay: np.ndarray) -> np.ndarray:
    """
    Compute displacement vectors at arbitrary query points.
    Returns (M, 2) array of (dx, dy) displacements.
    """
    # (M, N) pairwise distances from query to control points
    diff = query_pts[:, None, :] - src_pts[None, :, :]  # (M, N, 2)
    r    = np.linalg.norm(diff, axis=2)                  # (M, N)
    phi  = _rbf_kernel(r)                                # (M, N)

    # Weighted RBF sum  +  affine part  [1, x, y] · a
    dx = phi @ wx + ay[0] + ax[1] * query_pts[:, 0] + ax[2] * query_pts[:, 1]
    dy = phi @ wy + ay[0] + ay[1] * query_pts[:, 0] + ay[2] * query_pts[:, 1]

    return np.stack([dx, dy], axis=1)


# ---------------------------------------------------------------------------
# Section 3 — Vectorized Bilinear Interpolation (pure NumPy)
# ---------------------------------------------------------------------------

def _bilinear_interpolate(img: np.ndarray,
                          x:   np.ndarray,
                          y:   np.ndarray) -> np.ndarray:
    """
    Sample `img` at sub-pixel positions (x, y) using bilinear interpolation.

    Parameters
    ----------
    img : (H, W, C) float32 image, values in [0, 1]
    x   : (N,) float array — column (horizontal) positions
    y   : (N,) float array — row    (vertical)   positions

    Returns
    -------
    (N, C) sampled pixel values

    Mathematical derivation
    -----------------------
    For a query point (x, y), let:
        x0, y0 = floor(x), floor(y)          ← top-left pixel
        x1, y1 = x0 + 1,   y0 + 1            ← bottom-right pixel
        α = x − x0,  β = y − y0              ← fractional parts

    The bilinear estimate is:
        f ≈ (1-α)(1-β)·f(x0,y0)
          + (  α)(1-β)·f(x1,y0)
          + (1-α)(  β)·f(x0,y1)
          +    α · β  ·f(x1,y1)
    """
    H, W, C = img.shape

    # Clamp to valid range so boundary pixels don't go out-of-bounds
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    # Fractional parts (weights)
    alpha = (x - x0).astype(np.float32)[:, None]   # (N, 1)
    beta  = (y - y0).astype(np.float32)[:, None]   # (N, 1)

    # Corner pixel values — (N, C)
    f00 = img[y0, x0]   # top-left
    f10 = img[y0, x1]   # top-right
    f01 = img[y1, x0]   # bottom-left
    f11 = img[y1, x1]   # bottom-right

    # Bilinear formula
    return (
        (1 - alpha) * (1 - beta) * f00
      +       alpha  * (1 - beta) * f10
      + (1 - alpha) *       beta  * f01
      +       alpha  *       beta  * f11
    )


# ---------------------------------------------------------------------------
# Section 4 — Deformation Grid Visualizer
# ---------------------------------------------------------------------------

def _draw_deformation_grid(img_rgb:   np.ndarray,
                            src_pts:  np.ndarray,
                            dst_pts:  np.ndarray,
                            wx, ax, wy, ay,
                            grid_step: int = 25) -> np.ndarray:
    """
    Draw a regular grid warped by the displacement field on top of the image.
    Shows how each cell of the grid is distorted by the transformation.
    Returns (H, W, 3) uint8 image.
    """
    H, W = img_rgb.shape[:2]

    # Start from original image
    canvas = img_rgb.copy().astype(np.float32) / 255.0

    # Grid lines — horizontal
    for row_y in range(0, H + 1, grid_step):
        xs = np.linspace(0, W - 1, W * 2).astype(np.float64)
        ys = np.full_like(xs, float(min(row_y, H - 1)))
        pts = np.stack([xs, ys], axis=1)
        disp = _evaluate_rbf(pts, src_pts, wx, ax, wy, ay)
        warped_x = np.clip(xs + disp[:, 0], 0, W - 1).astype(np.int32)
        warped_y = np.clip(ys + disp[:, 1], 0, H - 1).astype(np.int32)
        canvas[warped_y, warped_x] = [0.2, 0.9, 0.4]   # green lines

    # Grid lines — vertical
    for col_x in range(0, W + 1, grid_step):
        ys = np.linspace(0, H - 1, H * 2).astype(np.float64)
        xs = np.full_like(ys, float(min(col_x, W - 1)))
        pts = np.stack([xs, ys], axis=1)
        disp = _evaluate_rbf(pts, src_pts, wx, ax, wy, ay)
        warped_x = np.clip(xs + disp[:, 0], 0, W - 1).astype(np.int32)
        warped_y = np.clip(ys + disp[:, 1], 0, H - 1).astype(np.int32)
        canvas[warped_y, warped_x] = [0.2, 0.9, 0.4]

    # Draw control-point arrows (source → destination)
    for s, d in zip(src_pts, dst_pts):
        sx, sy = int(np.clip(s[0], 0, W-1)), int(np.clip(s[1], 0, H-1))
        dx_, dy_ = int(np.clip(d[0], 0, W-1)), int(np.clip(d[1], 0, H-1))
        # Draw dot at source
        r = 4
        y0_ = max(0, sy - r); y1_ = min(H, sy + r)
        x0_ = max(0, sx - r); x1_ = min(W, sx + r)
        canvas[y0_:y1_, x0_:x1_] = [1.0, 0.2, 0.2]   # red dot (source)
        # Draw dot at destination
        y0_ = max(0, dy_ - r); y1_ = min(H, dy_ + r)
        x0_ = max(0, dx_ - r); x1_ = min(W, dx_ + r)
        canvas[y0_:y1_, x0_:x1_] = [0.1, 0.4, 1.0]   # blue dot (destination)

    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Section 5 — Image Quality Metrics (from scratch, no scikit-image)
# ---------------------------------------------------------------------------

def _compute_metrics(original: np.ndarray, warped: np.ndarray) -> dict:
    """
    Compute MSE, PSNR, and SSIM between two uint8 RGB images.

    MSE  = mean((A - B)²)
    PSNR = 10 · log10(MAX² / MSE)         where MAX = 255
    SSIM = structural similarity index (simplified single-scale)
    """
    A = original.astype(np.float64)
    B = warped.astype(np.float64)

    # --- MSE ---
    mse = float(np.mean((A - B) ** 2))

    # --- PSNR ---
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = float(10 * np.log10(255.0 ** 2 / mse))

    # --- SSIM (luminance-only, window-free global version) ---
    # Convert to float [0,1]
    Af = A / 255.0
    Bf = B / 255.0

    mu_a  = np.mean(Af)
    mu_b  = np.mean(Bf)
    sig_a = np.std(Af)
    sig_b = np.std(Bf)
    sig_ab = float(np.mean((Af - mu_a) * (Bf - mu_b)))

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim = float(
        ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2))
        / ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))
    )

    return {
        "mse":  round(mse,  4),
        "psnr": round(psnr, 4),
        "ssim": round(ssim, 4),
    }


# ---------------------------------------------------------------------------
# Section 6 — Main Public API
# ---------------------------------------------------------------------------

def warp_image(
    image_bytes: bytes,
    operation:   str   = "smile",
    intensity:   int   = 50,
    show_grid:   bool  = False,
) -> dict:
    """
    Apply geometric warping to `image_bytes` and return a result dict.

    Parameters
    ----------
    image_bytes : raw image file bytes (JPEG / PNG / etc.)
    operation   : one of smile | eyebrow_raise | lip_widen | face_slim | aging | deaging
    intensity   : 0–100 (mapped linearly to warp strength)
    show_grid   : if True, also return a deformation-grid visualization

    Returns
    -------
    {
      "processed_image"  : "data:image/png;base64,...",
      "grid_image"       : "data:image/png;base64,..." | None,
      "metrics"          : { mse, psnr, ssim },
      "algorithm_info"   : { ... }
    }
    """
    # ---- Load image --------------------------------------------------------
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to at most 1024 on the longest side to keep processing fast
    MAX_DIM = 1024
    W, H = pil_img.size
    scale = min(MAX_DIM / W, MAX_DIM / H, 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize(
            (int(W * scale), int(H * scale)),
            Image.LANCZOS
        )
    img_np = np.array(pil_img, dtype=np.uint8)   # (H, W, 3)
    H, W   = img_np.shape[:2]

    # ---- Build control points ----------------------------------------------
    t = intensity / 100.0
    src_pts, dst_pts = _get_control_points(operation, t, W, H)

    # ---- Fit RBF (TPS) -----------------------------------------------------
    wx, ax, wy, ay = _solve_rbf(src_pts, dst_pts)

    # ---- Build full inverse displacement field (vectorized) ----------------
    # For INVERSE MAPPING we need: where in SOURCE does each OUTPUT pixel map?
    # The forward map moves src → dst.  For the inverse map we evaluate the
    # displacement at each OUTPUT (destination) pixel and subtract it,
    # which gives a first-order approximation of the inverse TPS field.
    # This avoids holes entirely since every output pixel is filled.

    out_xs = np.tile(np.arange(W, dtype=np.float64), H)   # (H*W,)
    out_ys = np.repeat(np.arange(H, dtype=np.float64), W)  # (H*W,)
    query_pts = np.stack([out_xs, out_ys], axis=1)         # (H*W, 2)

    # Evaluate displacement at output positions (inverse = -displacement)
    disp = _evaluate_rbf(query_pts, src_pts, wx, ax, wy, ay)  # (H*W, 2)

    src_x = out_xs - disp[:, 0]  # sample x in the input image
    src_y = out_ys - disp[:, 1]  # sample y in the input image

    # ---- Bilinear interpolation (pure NumPy) --------------------------------
    img_float = img_np.astype(np.float32) / 255.0                  # (H, W, 3)
    sampled   = _bilinear_interpolate(img_float, src_x, src_y)     # (H*W, 3)
    warped_np = np.clip(sampled * 255, 0, 255).astype(np.uint8).reshape(H, W, 3)

    # ---- Encode output image to base64 PNG ---------------------------------
    def _to_b64_png(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    result_b64 = _to_b64_png(warped_np)

    # ---- Deformation grid --------------------------------------------------
    grid_b64 = None
    if show_grid:
        grid_np  = _draw_deformation_grid(img_np, src_pts, dst_pts, wx, ax, wy, ay)
        grid_b64 = _to_b64_png(grid_np)

    # ---- Quality metrics ---------------------------------------------------
    metrics = _compute_metrics(img_np, warped_np)

    # ---- Algorithm info (for UI / educational display) --------------------
    algo_info = {
        "method":          "Thin-Plate Spline (TPS) RBF Warping",
        "interpolation":   "Vectorized Bilinear (pure NumPy)",
        "mapping_strategy":"Inverse Mapping — no holes possible",
        "control_points":  len(src_pts),
        "image_size":      f"{W}×{H}",
        "operation":       operation,
        "intensity_pct":   intensity,
    }

    return {
        "processed_image": result_b64,
        "grid_image":      grid_b64,
        "metrics":         metrics,
        "algorithm_info":  algo_info,
    }
