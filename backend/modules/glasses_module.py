"""
Advanced Procedural 3D Glasses Rendering Module
================================================
Photorealistic, face-adaptive parametric glasses using OpenCV/NumPy.
Three models: Metal Aviator, Acetate Wayfarer, Minimalist Round.
"""

import cv2
import numpy as np
import math

# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def _lm_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm["x"] * w, lm["y"] * h], dtype=np.float64)


def _eye_center(landmarks, indices, w, h):
    pts = np.array([_lm_px(landmarks, i, w, h) for i in indices])
    return pts.mean(axis=0)


def _face_geometry(landmarks, w, h):
    """Extract geometric parameters from landmarks."""
    L_EYE = [33, 133, 160, 158, 153, 144, 159, 145]
    R_EYE = [362, 263, 387, 385, 380, 373, 386, 374]

    lc = _eye_center(landmarks, L_EYE, w, h)
    rc = _eye_center(landmarks, R_EYE, w, h)

    lo = _lm_px(landmarks, 33, w, h)
    li = _lm_px(landmarks, 133, w, h)
    ri = _lm_px(landmarks, 362, w, h)
    ro = _lm_px(landmarks, 263, w, h)

    nose = _lm_px(landmarks, 6, w, h)
    nose_tip = _lm_px(landmarks, 4, w, h)

    lt = _lm_px(landmarks, 234, w, h)
    rt = _lm_px(landmarks, 454, w, h)
    face_w = np.linalg.norm(rt - lt)

    angle = math.atan2(rc[1] - lc[1], rc[0] - lc[0])
    eye_dist = np.linalg.norm(rc - lc)
    bridge_w = np.linalg.norm(ri - li)
    l_eye_w = np.linalg.norm(li - lo)
    r_eye_w = np.linalg.norm(ro - ri)

    l_ear = _lm_px(landmarks, 234, w, h)
    r_ear = _lm_px(landmarks, 454, w, h)
    l_nose = _lm_px(landmarks, 198, w, h)
    r_nose = _lm_px(landmarks, 420, w, h)

    return {
        "lc": lc, "rc": rc, "lo": lo, "li": li, "ri": ri, "ro": ro,
        "nose": nose, "nose_tip": nose_tip, "face_w": face_w,
        "angle": angle, "eye_dist": eye_dist, "bridge_w": bridge_w,
        "l_eye_w": l_eye_w, "r_eye_w": r_eye_w,
        "l_ear": l_ear, "r_ear": r_ear,
        "l_nose": l_nose, "r_nose": r_nose,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rot(pt, center, angle):
    c, s = math.cos(angle), math.sin(angle)
    d = pt - center
    return center + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _rot_pts(pts, center, angle):
    return np.array([_rot(p, center, angle) for p in pts])


def _bezier_quad(p0, p1, p2, n=30):
    """Quadratic Bezier curve from p0 through control p1 to p2."""
    ts = np.linspace(0, 1, n)
    pts = []
    for t in ts:
        pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
        pts.append(pt)
    return np.array(pts, dtype=np.float64)


def _draw_smooth_curve(canvas, mask, pts, color, thickness, alpha=255):
    ipts = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [ipts], False, color, thickness, cv2.LINE_AA)
    cv2.polylines(mask, [ipts], False, alpha, thickness, cv2.LINE_AA)


def _line(canvas, mask, p1, p2, color, thickness, alpha=255):
    cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             color, thickness, cv2.LINE_AA)
    cv2.line(mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             alpha, thickness, cv2.LINE_AA)


def _fill_poly(canvas, mask, pts, color, alpha=200):
    ipts = pts.astype(np.int32)
    cv2.fillPoly(canvas, [ipts], color, cv2.LINE_AA)
    cv2.fillPoly(mask, [ipts], alpha, cv2.LINE_AA)


def _draw_poly(canvas, mask, pts, color, thickness, closed=True, alpha=255):
    ipts = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [ipts], closed, color, thickness, cv2.LINE_AA)
    cv2.polylines(mask, [ipts], closed, alpha, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Polygon shape generators (local coords, then rotated+translated)
# ---------------------------------------------------------------------------

def _teardrop_contour(cx, cy, half_w, half_h, n=60):
    """Aviator teardrop: wider top, narrowing toward bottom-inside."""
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        # Base ellipse
        x = math.cos(t)
        y = math.sin(t)
        # Squash bottom inward (teardrop sag)
        if y > 0:
            # Bottom half: narrow horizontally, extend vertically
            squeeze = 1.0 - 0.25 * y  # narrows at bottom
            x *= squeeze
            y *= 1.15  # extends downward
        else:
            # Top half: flatter
            y *= 0.85
        pts.append([cx + x * half_w, cy + y * half_h])
    return np.array(pts, dtype=np.float64)


def _wayfarer_contour(cx, cy, half_w, half_h, n=60):
    """Wayfarer: trapezoidal rounded-rectangle, wider at top, angled at bottom."""
    pts = []
    corner_r = half_w * 0.30  # corner rounding radius
    # Define the 4 corners of a trapezoid
    top_w = half_w * 1.0
    bot_w = half_w * 0.85  # narrower at bottom
    top_y = -half_h
    bot_y = half_h * 0.95

    # Build path: top-left → top-right → bottom-right → bottom-left
    # with rounded corners using small arcs
    segments = [
        # top edge (flat, slight upward bow)
        (-top_w, top_y), (top_w, top_y),
        # right side (angled inward)
        (bot_w, bot_y),
        # bottom edge (slight upward curve)
        (-bot_w, bot_y),
    ]

    # Generate rounded polygon
    for i in range(n):
        t = 2.0 * math.pi * i / n
        # Superellipse with n=3 for squarish shape
        ct = math.cos(t)
        st = math.sin(t)
        # Squarish superellipse
        exp = 2.8
        sx = abs(ct) ** (2.0/exp) * np.sign(ct)
        sy = abs(st) ** (2.0/exp) * np.sign(st)
        # Adjust width: wider at top, narrower at bottom
        w_factor = top_w if sy <= 0 else top_w - (top_w - bot_w) * sy
        pts.append([cx + sx * w_factor, cy + sy * half_h])

    return np.array(pts, dtype=np.float64)


def _round_contour(cx, cy, radius, n=60):
    """Perfect circle contour."""
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        pts.append([cx + math.cos(t) * radius, cy + math.sin(t) * radius])
    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Lens tint/fill for polygon shapes
# ---------------------------------------------------------------------------

def _fill_lens_poly(image, overlay, mask, contour, tint_color, tint_alpha):
    """Fill a polygon-shaped lens with tint and specular highlight."""
    h, w = image.shape[:2]
    ipts = contour.astype(np.int32)

    # Fill tint
    cv2.fillPoly(overlay, [ipts], tint_color, cv2.LINE_AA)
    cv2.fillPoly(mask, [ipts], tint_alpha, cv2.LINE_AA)

    # Specular highlight in upper-left area
    cx = contour[:, 0].mean()
    cy = contour[:, 1].mean()
    rx = (contour[:, 0].max() - contour[:, 0].min()) / 2
    ry = (contour[:, 1].max() - contour[:, 1].min()) / 2
    hx = int(cx - rx * 0.2)
    hy = int(cy - ry * 0.3)
    hr = max(3, int(min(rx, ry) * 0.3))

    h_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(h_mask, (hx, hy), hr, 255, -1, cv2.LINE_AA)
    h_mask = cv2.GaussianBlur(h_mask, (0, 0), max(1, hr * 0.4))

    # Clip highlight to lens area
    lens_region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_region, [ipts], 255, cv2.LINE_AA)
    h_mask = cv2.bitwise_and(h_mask, lens_region)

    h_f = (h_mask.astype(np.float32) / 255.0 * 0.2).reshape(h, w, 1)
    white = np.full_like(overlay, 255, dtype=np.uint8)
    overlay[:] = np.clip(
        overlay.astype(np.float32) + white.astype(np.float32) * h_f,
        0, 255
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Smooth arc bridge
# ---------------------------------------------------------------------------

def _draw_arc_bridge(overlay, mask, p_left, p_right, sag, color, thickness, alpha=255):
    """Draw a smooth upward-arcing bridge using a quadratic Bezier curve."""
    mid = (p_left + p_right) / 2.0
    control = mid + np.array([0, -sag])  # negative Y = upward arc
    curve = _bezier_quad(p_left, control, p_right, n=40)
    _draw_smooth_curve(overlay, mask, curve, color, thickness, alpha)
    return curve


# ---------------------------------------------------------------------------
# MODEL 1: Metal Aviator (teardrop shape + double bridge)
# ---------------------------------------------------------------------------

def _draw_aviator(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    angle = g["angle"]
    mid = (g["lc"] + g["rc"]) / 2.0

    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    ed = g["eye_dist"]
    frame_t = max(2, int(ed * 0.02))

    # Teardrop lens dimensions
    l_hw = int(g["l_eye_w"] * 0.68)
    l_hh = int(l_hw * 0.82)
    r_hw = int(g["r_eye_w"] * 0.68)
    r_hh = int(r_hw * 0.82)

    # Generate teardrop contours (local) then rotate for face tilt
    l_contour = _teardrop_contour(g["lc"][0], g["lc"][1], l_hw, l_hh)
    r_contour = _teardrop_contour(g["rc"][0], g["rc"][1], r_hw, r_hh)
    l_contour = _rot_pts(l_contour, g["lc"], angle)
    r_contour = _rot_pts(r_contour, g["rc"], angle)

    # Lens fill (green-grey aviator tint)
    _fill_lens_poly(image, overlay, mask, l_contour, (45, 65, 35), 140)
    _fill_lens_poly(image, overlay, mask, r_contour, (45, 65, 35), 140)

    # Thin metal frame outline
    frame_color = (160, 160, 170)
    _draw_poly(overlay, mask, l_contour, frame_color, frame_t)
    _draw_poly(overlay, mask, r_contour, frame_color, frame_t)

    # === Smooth arc bridge ===
    # Connect inner edges of lenses with an upward arc
    l_inner = g["li"].copy()
    r_inner = g["ri"].copy()
    bridge_sag = ed * 0.06  # arc height
    _draw_arc_bridge(overlay, mask, l_inner, r_inner, bridge_sag,
                     frame_color, frame_t)

    # Double bridge: top bar (straight, slightly above the arc)
    top_offset = np.array([0, -ed * 0.07])
    top_l = _rot(l_inner + top_offset, mid, angle)
    top_r = _rot(r_inner + top_offset, mid, angle)
    _line(overlay, mask, top_l, top_r, frame_color, max(1, frame_t - 1))

    # Nose pads
    pad_r = max(2, int(ed * 0.018))
    pad_color = (180, 180, 185)
    bridge_mid = (l_inner + r_inner) / 2.0
    for np_pos in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   pad_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   200, -1, cv2.LINE_AA)
        pad_arm = _bezier_quad(np_pos, (np_pos + bridge_mid) / 2 + np.array([0, -ed*0.02]),
                               bridge_mid, n=20)
        _draw_smooth_curve(overlay, mask, pad_arm, frame_color,
                           max(1, frame_t - 1), 180)

    # Temple arms
    arm_t = max(1, int(ed * 0.015))
    l_arm_end = g["l_ear"] + np.array([0, -int(ed * 0.05)])
    r_arm_end = g["r_ear"] + np.array([0, -int(ed * 0.05)])
    _line(overlay, mask, g["lo"], l_arm_end, frame_color, arm_t, 220)
    _line(overlay, mask, g["ro"], r_arm_end, frame_color, arm_t, 220)

    # Rubber arm tips
    tip_color = (60, 60, 65)
    l_tip = l_arm_end + np.array([-ed * 0.04, ed * 0.06])
    r_tip = r_arm_end + np.array([ed * 0.04, ed * 0.06])
    _line(overlay, mask, l_arm_end, l_tip, tip_color, arm_t + 1, 240)
    _line(overlay, mask, r_arm_end, r_tip, tip_color, arm_t + 1, 240)

    return image, overlay, mask


# ---------------------------------------------------------------------------
# MODEL 2: Acetate Wayfarer (trapezoidal + thick frames)
# ---------------------------------------------------------------------------

def _draw_wayfarer(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    angle = g["angle"]
    mid = (g["lc"] + g["rc"]) / 2.0

    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    ed = g["eye_dist"]
    frame_t = max(4, int(ed * 0.05))

    # Wayfarer lens dimensions (wide, not very tall)
    l_hw = int(g["l_eye_w"] * 0.78)
    l_hh = int(l_hw * 0.62)
    r_hw = int(g["r_eye_w"] * 0.78)
    r_hh = int(r_hw * 0.62)

    # Generate wayfarer contours then rotate
    l_contour = _wayfarer_contour(g["lc"][0], g["lc"][1], l_hw, l_hh)
    r_contour = _wayfarer_contour(g["rc"][0], g["rc"][1], r_hw, r_hh)
    l_contour = _rot_pts(l_contour, g["lc"], angle)
    r_contour = _rot_pts(r_contour, g["rc"], angle)

    # Dark lens fill
    _fill_lens_poly(image, overlay, mask, l_contour, (25, 30, 35), 170)
    _fill_lens_poly(image, overlay, mask, r_contour, (25, 30, 35), 170)

    # Thick black acetate frame
    frame_outer = (15, 15, 18)
    frame_inner = (50, 45, 42)
    _draw_poly(overlay, mask, l_contour, frame_outer, frame_t)
    _draw_poly(overlay, mask, r_contour, frame_outer, frame_t)

    # Inner bevel highlight
    inner_t = max(1, frame_t // 3)
    # Shrink contours slightly for inner line
    def _shrink(contour, center, factor=0.92):
        return center + (contour - center) * factor
    l_inner_c = _shrink(l_contour, g["lc"])
    r_inner_c = _shrink(r_contour, g["rc"])
    _draw_poly(overlay, mask, l_inner_c, frame_inner, inner_t, alpha=100)
    _draw_poly(overlay, mask, r_inner_c, frame_inner, inner_t, alpha=100)

    # === Smooth arc bridge (keyhole style) ===
    l_inner = g["li"].copy()
    r_inner = g["ri"].copy()
    bridge_sag = ed * 0.05
    _draw_arc_bridge(overlay, mask, l_inner, r_inner, bridge_sag,
                     frame_outer, max(3, int(ed * 0.035)))

    # Keyhole notch: small downward arc below the bridge
    kh_sag = -ed * 0.03  # negative = downward
    kh_l = l_inner + np.array([ed * 0.03, 0])
    kh_r = r_inner - np.array([ed * 0.03, 0])
    _draw_arc_bridge(overlay, mask, kh_l, kh_r, kh_sag,
                     frame_outer, max(2, int(ed * 0.025)))

    # Hinge pins (silver dots at upper outer corners)
    pin_r = max(2, int(ed * 0.014))
    pin_color = (160, 155, 140)
    for pos in [g["lo"], g["ro"]]:
        pin_pos = pos + np.array([0, -l_hh * 0.15])  # slightly above center
        cv2.circle(overlay, (int(pin_pos[0]), int(pin_pos[1])), pin_r,
                   pin_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(pin_pos[0]), int(pin_pos[1])), pin_r,
                   255, -1, cv2.LINE_AA)
        # Cross detail
        cv2.line(overlay,
                 (int(pin_pos[0] - pin_r + 1), int(pin_pos[1])),
                 (int(pin_pos[0] + pin_r - 1), int(pin_pos[1])),
                 (110, 105, 95), 1, cv2.LINE_AA)

    # Thick acetate temple arms
    arm_t = max(3, int(ed * 0.035))
    l_arm_end = g["l_ear"] + np.array([0, -int(ed * 0.03)])
    r_arm_end = g["r_ear"] + np.array([0, -int(ed * 0.03)])
    _line(overlay, mask, g["lo"], l_arm_end, frame_outer, arm_t, 240)
    _line(overlay, mask, g["ro"], r_arm_end, frame_outer, arm_t, 240)
    # Highlight stripe on arms
    _line(overlay, mask, g["lo"], l_arm_end, frame_inner,
          max(1, arm_t // 3), 70)
    _line(overlay, mask, g["ro"], r_arm_end, frame_inner,
          max(1, arm_t // 3), 70)

    # Arm tips curve down
    l_tip = l_arm_end + np.array([-ed * 0.03, ed * 0.09])
    r_tip = r_arm_end + np.array([ed * 0.03, ed * 0.09])
    _line(overlay, mask, l_arm_end, l_tip, frame_outer, arm_t, 220)
    _line(overlay, mask, r_arm_end, r_tip, frame_outer, arm_t, 220)

    return image, overlay, mask


# ---------------------------------------------------------------------------
# MODEL 3: Minimalist Round (circle + smooth arc bridge)
# ---------------------------------------------------------------------------

def _draw_round(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    angle = g["angle"]
    mid = (g["lc"] + g["rc"]) / 2.0

    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    ed = g["eye_dist"]
    frame_t = max(1, int(ed * 0.018))

    # Round lenses
    l_r = int(g["l_eye_w"] * 0.48)
    r_r = int(g["r_eye_w"] * 0.48)

    l_contour = _round_contour(g["lc"][0], g["lc"][1], l_r)
    r_contour = _round_contour(g["rc"][0], g["rc"][1], r_r)
    l_contour = _rot_pts(l_contour, g["lc"], angle)
    r_contour = _rot_pts(r_contour, g["rc"], angle)

    # Subtle clear tint
    _fill_lens_poly(image, overlay, mask, l_contour, (220, 215, 210), 35)
    _fill_lens_poly(image, overlay, mask, r_contour, (220, 215, 210), 35)

    # Rose-gold wire frame
    frame_color = (130, 170, 210)
    _draw_poly(overlay, mask, l_contour, frame_color, frame_t)
    _draw_poly(overlay, mask, r_contour, frame_color, frame_t)

    # === Smooth arc bridge ===
    l_inner = g["li"].copy()
    r_inner = g["ri"].copy()
    bridge_sag = ed * 0.045
    _draw_arc_bridge(overlay, mask, l_inner, r_inner, bridge_sag,
                     frame_color, frame_t)

    # Nose pads
    pad_r = max(1, int(ed * 0.015))
    bridge_mid = (l_inner + r_inner) / 2.0
    for np_pos in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   frame_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   180, -1, cv2.LINE_AA)
        pad_arm = _bezier_quad(np_pos, (np_pos + bridge_mid)/2 + np.array([0, -ed*0.015]),
                               bridge_mid, n=15)
        _draw_smooth_curve(overlay, mask, pad_arm, frame_color,
                           max(1, frame_t - 1), 150)

    # Ultra-thin temple arms
    arm_t = max(1, int(ed * 0.013))
    l_arm_end = g["l_ear"] + np.array([0, -int(ed * 0.04)])
    r_arm_end = g["r_ear"] + np.array([0, -int(ed * 0.04)])
    _line(overlay, mask, g["lo"], l_arm_end, frame_color, arm_t, 200)
    _line(overlay, mask, g["ro"], r_arm_end, frame_color, arm_t, 200)

    l_tip = l_arm_end + np.array([-ed * 0.025, ed * 0.05])
    r_tip = r_arm_end + np.array([ed * 0.025, ed * 0.05])
    _line(overlay, mask, l_arm_end, l_tip, frame_color, arm_t, 180)
    _line(overlay, mask, r_arm_end, r_tip, frame_color, arm_t, 180)

    return image, overlay, mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MODEL_DISPATCH = {
    "aviator": _draw_aviator,
    "wayfarer": _draw_wayfarer,
    "round": _draw_round,
    "sunglasses": _draw_wayfarer,
    "reading": _draw_round,
}


def apply_glasses(image, landmarks, model_id="aviator"):
    """
    Render procedural 3D-modeled glasses onto a face image.

    Parameters
    ----------
    image : np.ndarray  - BGR input image
    landmarks : list    - MediaPipe 468 landmark dicts with x,y keys
    model_id : str      - one of: aviator, wayfarer, round

    Returns
    -------
    np.ndarray - BGR image with glasses composited
    """
    h, w = image.shape[:2]
    draw_fn = MODEL_DISPATCH.get(model_id.lower(), _draw_aviator)

    processed = image.copy()
    processed, overlay, mask = draw_fn(processed, landmarks, w, h)

    # Soften mask edges
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 1.5)
    mask_f = mask_blur.astype(np.float32) / 255.0
    mask_3 = np.stack([mask_f] * 3, axis=2)

    # Pre-multiplied alpha composite
    result = (processed.astype(np.float32) * (1.0 - mask_3) +
              overlay.astype(np.float32) * mask_3).astype(np.uint8)

    return result
