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

    # Face width from temples
    lt = _lm_px(landmarks, 234, w, h)
    rt = _lm_px(landmarks, 454, w, h)
    face_w = np.linalg.norm(rt - lt)

    # Tilt angle
    angle = math.atan2(rc[1] - lc[1], rc[0] - lc[0])
    eye_dist = np.linalg.norm(rc - lc)

    # Bridge width
    bridge_w = np.linalg.norm(ri - li)

    # Eye dimensions
    l_eye_w = np.linalg.norm(li - lo)
    r_eye_w = np.linalg.norm(ro - ri)

    # Cheek landmarks for temple arm endpoints
    l_ear = _lm_px(landmarks, 234, w, h)
    r_ear = _lm_px(landmarks, 454, w, h)

    # Nose pad positions
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
# Rotation helper
# ---------------------------------------------------------------------------

def _rot(pt, center, angle):
    c, s = math.cos(angle), math.sin(angle)
    d = pt - center
    return center + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _rot_pts(pts, center, angle):
    return np.array([_rot(p, center, angle) for p in pts])


# ---------------------------------------------------------------------------
# Drawing primitives with anti-aliasing
# ---------------------------------------------------------------------------

def _thick_ellipse(canvas, mask, center, axes, angle_deg, color, thickness, alpha=255):
    cx, cy = int(center[0]), int(center[1])
    ax, ay = int(axes[0]), int(axes[1])
    cv2.ellipse(canvas, (cx, cy), (ax, ay), angle_deg, 0, 360, color, thickness, cv2.LINE_AA)
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle_deg, 0, 360, alpha, thickness, cv2.LINE_AA)


def _filled_ellipse(canvas, mask, center, axes, angle_deg, color, alpha=200):
    cx, cy = int(center[0]), int(center[1])
    ax, ay = int(axes[0]), int(axes[1])
    cv2.ellipse(canvas, (cx, cy), (ax, ay), angle_deg, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle_deg, 0, 360, alpha, -1, cv2.LINE_AA)


def _polyline(canvas, mask, pts, color, thickness, alpha=255):
    ipts = pts.astype(np.int32)
    cv2.polylines(canvas, [ipts], False, color, thickness, cv2.LINE_AA)
    cv2.polylines(mask, [ipts], False, alpha, thickness, cv2.LINE_AA)


def _line(canvas, mask, p1, p2, color, thickness, alpha=255):
    cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             color, thickness, cv2.LINE_AA)
    cv2.line(mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             alpha, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Lens rendering with refraction + tint + reflection
# ---------------------------------------------------------------------------

def _render_lens(image, overlay, mask, center, axes, angle_deg,
                 tint_color, tint_alpha, refraction=0.0):
    """Render a photorealistic lens with tint, refraction distortion, and reflection."""
    h, w = image.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    ax, ay = int(axes[0]), int(axes[1])

    # Create lens mask
    lens_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(lens_mask, (cx, cy), (ax, ay), angle_deg, 0, 360, 255, -1, cv2.LINE_AA)

    # Refraction: subtle barrel distortion inside lens area
    if abs(refraction) > 0.001:
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for yy in range(max(0, cy - ay - 5), min(h, cy + ay + 5)):
            for xx in range(max(0, cx - ax - 5), min(w, cx + ax + 5)):
                dx = (xx - cx) / max(ax, 1)
                dy = (yy - cy) / max(ay, 1)
                r2 = dx * dx + dy * dy
                if r2 < 1.0:
                    factor = 1.0 + refraction * r2
                    map_x[yy, xx] = cx + dx * ax * factor
                    map_y[yy, xx] = cy + dy * ay * factor
                else:
                    map_x[yy, xx] = xx
                    map_y[yy, xx] = yy
            # Fill remaining
        # Only remap the lens region
        refracted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        lens_f = (lens_mask / 255.0).reshape(h, w, 1)
        image_region = image.copy()
        image_region = (image_region * (1 - lens_f) + refracted * lens_f).astype(np.uint8)
    else:
        image_region = image

    # Tint overlay
    tint_layer = np.full_like(overlay, tint_color, dtype=np.uint8)
    lens_f = lens_mask.astype(np.float32) / 255.0
    lens_3 = np.stack([lens_f] * 3, axis=2)
    alpha_f = tint_alpha / 255.0
    overlay[:] = (overlay.astype(np.float32) * (1 - lens_3 * alpha_f) +
                  tint_layer.astype(np.float32) * lens_3 * alpha_f).astype(np.uint8)

    # Update mask
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle_deg, 0, 360, tint_alpha, -1, cv2.LINE_AA)

    # Specular highlight (crescent reflection)
    highlight_cx = cx - int(ax * 0.25)
    highlight_cy = cy - int(ay * 0.25)
    highlight_ax = max(2, int(ax * 0.35))
    highlight_ay = max(2, int(ay * 0.20))
    h_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(h_mask, (highlight_cx, highlight_cy),
                (highlight_ax, highlight_ay), angle_deg - 15,
                0, 360, 255, -1, cv2.LINE_AA)
    h_mask = cv2.GaussianBlur(h_mask, (0, 0), max(1, ax * 0.08))
    h_f = (h_mask.astype(np.float32) / 255.0 * 0.25).reshape(h, w, 1)
    # Only apply within lens
    h_f = h_f * lens_3
    white = np.full_like(overlay, (255, 255, 255), dtype=np.uint8)
    overlay[:] = np.clip(overlay.astype(np.float32) + white.astype(np.float32) * h_f,
                         0, 255).astype(np.uint8)

    return image_region


# ---------------------------------------------------------------------------
# Gradient frame rendering
# ---------------------------------------------------------------------------

def _gradient_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ---------------------------------------------------------------------------
# MODEL 1: Metal Aviator
# ---------------------------------------------------------------------------

def _draw_aviator(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    angle_deg = math.degrees(g["angle"])
    mid = (g["lc"] + g["rc"]) / 2.0

    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    ed = g["eye_dist"]
    frame_t = max(1, int(ed * 0.018))

    # Teardrop-shaped aviator lenses (taller than wide at bottom)
    l_ax = int(g["l_eye_w"] * 0.72)
    l_ay = int(l_ax * 0.85)
    r_ax = int(g["r_eye_w"] * 0.72)
    r_ay = int(r_ax * 0.85)

    # Lens tint - classic aviator green-grey gradient
    lens_tint = (45, 65, 35)
    image = _render_lens(image, overlay, mask, g["lc"], (l_ax, l_ay),
                         angle_deg, lens_tint, 140, refraction=0.008)
    image = _render_lens(image, overlay, mask, g["rc"], (r_ax, r_ay),
                         angle_deg, lens_tint, 140, refraction=0.008)

    # Thin metal frames - silver/gunmetal
    frame_color = (160, 160, 170)
    _thick_ellipse(overlay, mask, g["lc"], (l_ax, l_ay), angle_deg,
                   frame_color, frame_t)
    _thick_ellipse(overlay, mask, g["rc"], (r_ax, r_ay), angle_deg,
                   frame_color, frame_t)

    # Bridge - double bar aviator style
    bridge_y_off = int(ed * 0.03)
    b1_l = _rot(g["li"] + np.array([0, -bridge_y_off]), mid, 0)
    b1_r = _rot(g["ri"] + np.array([0, -bridge_y_off]), mid, 0)
    b2_l = _rot(g["li"] + np.array([0, bridge_y_off]), mid, 0)
    b2_r = _rot(g["ri"] + np.array([0, bridge_y_off]), mid, 0)

    _line(overlay, mask, b1_l, b1_r, frame_color, max(1, frame_t - 1))
    _line(overlay, mask, b2_l, b2_r, frame_color, max(1, frame_t - 1))

    # Nose pads - small circles
    pad_r = max(2, int(ed * 0.02))
    pad_color = (180, 180, 185)
    for np_pos in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   pad_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   200, -1, cv2.LINE_AA)
        # Pad arm
        bridge_mid = (g["li"] + g["ri"]) / 2.0
        _line(overlay, mask, np_pos, bridge_mid, frame_color,
              max(1, frame_t - 1), 180)

    # Temple arms - thin metal
    arm_len = int(ed * 0.55)
    arm_t = max(1, int(ed * 0.015))
    l_arm_end = g["l_ear"] + np.array([0, -int(ed * 0.05)])
    r_arm_end = g["r_ear"] + np.array([0, -int(ed * 0.05)])

    # Arm with slight curve (3 control points)
    l_hinge = g["lo"] + np.array([-int(ed * 0.02), 0])
    r_hinge = g["ro"] + np.array([int(ed * 0.02), 0])

    _line(overlay, mask, l_hinge, l_arm_end, frame_color, arm_t, 220)
    _line(overlay, mask, r_hinge, r_arm_end, frame_color, arm_t, 220)

    # Arm tips (acetate/rubber ends)
    tip_len = int(ed * 0.12)
    tip_color = (60, 60, 65)
    l_tip = l_arm_end + np.array([-tip_len * 0.3, tip_len * 0.5])
    r_tip = r_arm_end + np.array([tip_len * 0.3, tip_len * 0.5])
    _line(overlay, mask, l_arm_end, l_tip, tip_color, arm_t + 1, 240)
    _line(overlay, mask, r_arm_end, r_tip, tip_color, arm_t + 1, 240)

    return image, overlay, mask


# ---------------------------------------------------------------------------
# MODEL 2: Acetate Wayfarer
# ---------------------------------------------------------------------------

def _draw_wayfarer(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    angle_deg = math.degrees(g["angle"])
    mid = (g["lc"] + g["rc"]) / 2.0

    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    ed = g["eye_dist"]
    frame_t = max(3, int(ed * 0.045))

    # Wayfarer: trapezoidal-ish, wider at top
    l_ax = int(g["l_eye_w"] * 0.75)
    l_ay = int(l_ax * 0.65)
    r_ax = int(g["r_eye_w"] * 0.75)
    r_ay = int(r_ax * 0.65)

    # Dark tinted lenses
    lens_tint = (25, 30, 35)
    image = _render_lens(image, overlay, mask, g["lc"], (l_ax, l_ay),
                         angle_deg, lens_tint, 170, refraction=0.005)
    image = _render_lens(image, overlay, mask, g["rc"], (r_ax, r_ay),
                         angle_deg, lens_tint, 170, refraction=0.005)

    # Thick acetate frames - glossy black with gradient
    frame_outer = (15, 15, 18)
    frame_inner = (45, 40, 38)

    # Outer thick frame
    _thick_ellipse(overlay, mask, g["lc"], (l_ax, l_ay), angle_deg,
                   frame_outer, frame_t)
    _thick_ellipse(overlay, mask, g["rc"], (r_ax, r_ay), angle_deg,
                   frame_outer, frame_t)

    # Inner highlight line for depth
    inner_t = max(1, frame_t // 3)
    _thick_ellipse(overlay, mask, g["lc"],
                   (l_ax - frame_t // 2, l_ay - frame_t // 2),
                   angle_deg, frame_inner, inner_t, 120)
    _thick_ellipse(overlay, mask, g["rc"],
                   (r_ax - frame_t // 2, r_ay - frame_t // 2),
                   angle_deg, frame_inner, inner_t, 120)

    # Bridge - thick keyhole style
    bridge_t = max(3, int(ed * 0.035))
    _line(overlay, mask, g["li"], g["ri"], frame_outer, bridge_t)

    # Keyhole bridge detail
    bridge_mid = (g["li"] + g["ri"]) / 2.0
    kh_r = int(g["bridge_w"] * 0.18)
    cv2.circle(overlay, (int(bridge_mid[0]), int(bridge_mid[1] + kh_r * 0.5)),
               kh_r, frame_outer, max(1, bridge_t // 2), cv2.LINE_AA)
    cv2.circle(mask, (int(bridge_mid[0]), int(bridge_mid[1] + kh_r * 0.5)),
               kh_r, 200, max(1, bridge_t // 2), cv2.LINE_AA)

    # Screw details at corners
    screw_r = max(1, int(ed * 0.012))
    screw_color = (140, 135, 120)
    for pos in [g["lo"], g["ro"]]:
        cv2.circle(overlay, (int(pos[0]), int(pos[1])), screw_r,
                   screw_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(pos[0]), int(pos[1])), screw_r,
                   255, -1, cv2.LINE_AA)
        # Screw cross
        cv2.line(overlay,
                 (int(pos[0] - screw_r), int(pos[1])),
                 (int(pos[0] + screw_r), int(pos[1])),
                 (100, 95, 85), 1, cv2.LINE_AA)

    # Temple arms - thick acetate
    arm_t = max(2, int(ed * 0.03))
    l_arm_end = g["l_ear"] + np.array([0, -int(ed * 0.03)])
    r_arm_end = g["r_ear"] + np.array([0, -int(ed * 0.03)])

    _line(overlay, mask, g["lo"], l_arm_end, frame_outer, arm_t, 240)
    _line(overlay, mask, g["ro"], r_arm_end, frame_outer, arm_t, 240)

    # Gradient highlight on arms
    _line(overlay, mask, g["lo"], l_arm_end, frame_inner,
          max(1, arm_t // 3), 80)
    _line(overlay, mask, g["ro"], r_arm_end, frame_inner,
          max(1, arm_t // 3), 80)

    # Arm tips curve down
    tip_len = int(ed * 0.15)
    l_tip = l_arm_end + np.array([-tip_len * 0.2, tip_len * 0.6])
    r_tip = r_arm_end + np.array([tip_len * 0.2, tip_len * 0.6])
    _line(overlay, mask, l_arm_end, l_tip, frame_outer, arm_t, 220)
    _line(overlay, mask, r_arm_end, r_tip, frame_outer, arm_t, 220)

    return image, overlay, mask


# ---------------------------------------------------------------------------
# MODEL 3: Minimalist Round
# ---------------------------------------------------------------------------

def _draw_round(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    angle_deg = math.degrees(g["angle"])
    mid = (g["lc"] + g["rc"]) / 2.0

    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    ed = g["eye_dist"]
    frame_t = max(1, int(ed * 0.015))

    # Perfectly round lenses
    l_r = int(g["l_eye_w"] * 0.52)
    r_r = int(g["r_eye_w"] * 0.52)

    # Very subtle clear/light tint
    lens_tint = (220, 215, 210)
    image = _render_lens(image, overlay, mask, g["lc"], (l_r, l_r),
                         angle_deg, lens_tint, 35, refraction=0.012)
    image = _render_lens(image, overlay, mask, g["rc"], (r_r, r_r),
                         angle_deg, lens_tint, 35, refraction=0.012)

    # Ultra-thin metal wire frame - gold/rose-gold
    frame_color = (130, 170, 210)  # Rose-gold in BGR
    _thick_ellipse(overlay, mask, g["lc"], (l_r, l_r), angle_deg,
                   frame_color, frame_t)
    _thick_ellipse(overlay, mask, g["rc"], (r_r, r_r), angle_deg,
                   frame_color, frame_t)

    # Minimal bridge
    _line(overlay, mask, g["li"], g["ri"], frame_color, frame_t)

    # Nose pads
    pad_r = max(1, int(ed * 0.015))
    for np_pos in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   frame_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_pos[0]), int(np_pos[1])), pad_r,
                   180, -1, cv2.LINE_AA)
        # Thin pad arm
        bridge_mid = (g["li"] + g["ri"]) / 2.0
        _line(overlay, mask, np_pos, bridge_mid, frame_color,
              max(1, frame_t - 1), 150)

    # Temple arms - ultra thin wire
    arm_t = max(1, int(ed * 0.012))
    l_arm_end = g["l_ear"] + np.array([0, -int(ed * 0.04)])
    r_arm_end = g["r_ear"] + np.array([0, -int(ed * 0.04)])

    _line(overlay, mask, g["lo"], l_arm_end, frame_color, arm_t, 200)
    _line(overlay, mask, g["ro"], r_arm_end, frame_color, arm_t, 200)

    # Subtle arm tips
    tip_len = int(ed * 0.10)
    l_tip = l_arm_end + np.array([-tip_len * 0.2, tip_len * 0.5])
    r_tip = r_arm_end + np.array([tip_len * 0.2, tip_len * 0.5])
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
    # Legacy compat
    "sunglasses": _draw_wayfarer,
    "reading": _draw_round,
}


def apply_glasses(image, landmarks, model_id="aviator"):
    """
    Render procedural 3D-modeled glasses onto a face image.

    Parameters
    ----------
    image : np.ndarray  – BGR input image
    landmarks : list    – MediaPipe 468 landmark dicts with x,y keys
    model_id : str      – one of: aviator, wayfarer, round

    Returns
    -------
    np.ndarray – BGR image with glasses composited
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
