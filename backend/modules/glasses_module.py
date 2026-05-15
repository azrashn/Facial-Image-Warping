"""
Advanced Procedural & sprite-based glasses overlay for Face AR.
---------------------------------------------------------------
Procedural models (metal aviator, acetate wayfarer, minimalist round, etc.)
draw straight temple segments from outer-eye hinges to temporal anchors.

PNG assets resolve from ``backend/assets/glasses/<model>/<model>_model.png`` when
present; missing files gracefully fall back to matched procedural shaders.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

_GLASSES_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "glasses")
)


def _lm_px(landmarks, idx: int, w: int, h: int):
    lm = landmarks[idx]
    return np.array([lm["x"] * w, lm["y"] * h], dtype=np.float64)


def _eye_center(landmarks, indices, w, h):
    pts = np.array([_lm_px(landmarks, i, w, h) for i in indices])
    return pts.mean(axis=0)


def _face_geometry(landmarks, w, h):
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
    l_temple = _lm_px(landmarks, 162, w, h)
    r_temple = _lm_px(landmarks, 389, w, h)
    l_nose = _lm_px(landmarks, 198, w, h)
    r_nose = _lm_px(landmarks, 420, w, h)
    return {
        "lc": lc,
        "rc": rc,
        "lo": lo,
        "li": li,
        "ri": ri,
        "ro": ro,
        "nose": nose,
        "nose_tip": nose_tip,
        "face_w": face_w,
        "angle": angle,
        "eye_dist": eye_dist,
        "bridge_w": bridge_w,
        "l_eye_w": l_eye_w,
        "r_eye_w": r_eye_w,
        "l_temple": l_temple,
        "r_temple": r_temple,
        "l_nose": l_nose,
        "r_nose": r_nose,
        "l_tragion": lt,
        "r_tragion": rt,
    }


# --- geometry helpers ---
def _rot(pt, center, angle):
    c, s = math.cos(angle), math.sin(angle)
    d = pt - center
    return center + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _rot_pts(pts, center, angle):
    return np.array([_rot(p, center, angle) for p in pts])


def _bezier_quad(p0, p1, p2, n=30):
    ts = np.linspace(0, 1, n)
    return np.array([(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in ts])


def _shrink(contour, center, factor):
    return center + (contour - center) * factor


# --- drawing helpers ---
def _draw_curve(canvas, mask, pts, color, thick, alpha=255):
    ip = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [ip], False, color, thick, cv2.LINE_AA)
    cv2.polylines(mask, [ip], False, alpha, thick, cv2.LINE_AA)


def _line(canvas, mask, p1, p2, color, thick, alpha=255):
    cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick, cv2.LINE_AA)
    cv2.line(mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), alpha, thick, cv2.LINE_AA)


def _fill_poly(canvas, mask, pts, color, alpha=255):
    ip = pts.astype(np.int32)
    cv2.fillPoly(canvas, [ip], color, cv2.LINE_AA)
    cv2.fillPoly(mask, [ip], alpha, cv2.LINE_AA)


def _draw_poly(canvas, mask, pts, color, thick, closed=True, alpha=255):
    ip = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [ip], closed, color, thick, cv2.LINE_AA)
    cv2.polylines(mask, [ip], closed, alpha, thick, cv2.LINE_AA)


# --- contour generators ---
def _teardrop_contour(cx, cy, hw, hh, n=60):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = math.cos(t), math.sin(t)
        if y > 0:
            x *= 1.0 - 0.25 * y
            y *= 1.15
        else:
            y *= 0.85
        pts.append([cx + x * hw, cy + y * hh])
    return np.array(pts, dtype=np.float64)


def _wayfarer_contour(cx, cy, hw, hh, n=60):
    top_w, bot_w = hw, hw * 0.85
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        ct, st = math.cos(t), math.sin(t)
        exp = 2.8
        sx = abs(ct) ** (2.0 / exp) * np.sign(ct)
        sy = abs(st) ** (2.0 / exp) * np.sign(st)
        wf = top_w if sy <= 0 else top_w - (top_w - bot_w) * sy
        pts.append([cx + sx * wf, cy + sy * hh])
    return np.array(pts, dtype=np.float64)


def _round_contour(cx, cy, r, n=60):
    return np.array(
        [
            [
                cx + math.cos(2 * math.pi * i / n) * r,
                cy + math.sin(2 * math.pi * i / n) * r,
            ]
            for i in range(n)
        ],
        dtype=np.float64,
    )


def _cat_eye_contour(cx, cy, hw, hh, n=72):
    """Up-swept outer wings (outer upper quadrant stretched)."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        ct, st = math.cos(t), math.sin(t)
        bx = hw * (abs(ct) ** 0.9) * np.sign(ct)
        by = hh * st
        if ct > 0.25 and by < -0.12 * hh:
            bx *= 1.22
            by *= 0.94
        if ct > 0.45 and abs(by) < 0.55 * hh:
            bx *= 1.08
        pts.append([cx + bx, cy + by])
    return np.array(pts, dtype=np.float64)


def _sport_wrap_contour(cx, cy, hw, hh, n=64):
    """Wraparound sporty lens: wide horizontal ellipse with softened vertical."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        ct, st = math.cos(t), math.sin(t)
        xs = ct * hw * (1.0 + 0.12 * abs(st))
        ys = st * hh * (0.88 + 0.12 * abs(ct))
        pts.append([cx + xs, cy + ys])
    return np.array(pts, dtype=np.float64)


# --- Hollow-frame + lens compositing ---
def _composite_frame_and_lens(
    overlay, mask, contour, center,
    frame_color, frame_thick,
    lens_tint, lens_alpha,
    inner_factor=0.88,
):
    h, w = overlay.shape[:2]
    inner = _shrink(contour, center, inner_factor)
    ipts_out = contour.astype(np.int32)
    ipts_in = inner.astype(np.int32)

    frame_layer = np.zeros_like(overlay)
    frame_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(frame_layer, [ipts_out], frame_color, cv2.LINE_AA)
    cv2.fillPoly(frame_mask, [ipts_out], 255, cv2.LINE_AA)
    cv2.fillPoly(frame_layer, [ipts_in], (0, 0, 0), cv2.LINE_AA)
    cv2.fillPoly(frame_mask, [ipts_in], 0, cv2.LINE_AA)

    lens_layer = np.zeros_like(overlay)
    lens_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_layer, [ipts_in], lens_tint, cv2.LINE_AA)
    cv2.fillPoly(lens_mask, [ipts_in], lens_alpha, cv2.LINE_AA)

    cx_f = inner[:, 0].mean()
    cy_f = inner[:, 1].mean()
    rx = (inner[:, 0].max() - inner[:, 0].min()) / 2
    ry = (inner[:, 1].max() - inner[:, 1].min()) / 2
    hx, hy = int(cx_f - rx * 0.2), int(cy_f - ry * 0.3)
    hr = max(3, int(min(rx, ry) * 0.3))
    h_m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(h_m, (hx, hy), hr, 255, -1, cv2.LINE_AA)
    h_m = cv2.GaussianBlur(h_m, (0, 0), max(1, hr * 0.4))
    lens_rgn = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_rgn, [ipts_in], 255, cv2.LINE_AA)
    h_m = cv2.bitwise_and(h_m, lens_rgn)
    h_f = (h_m.astype(np.float32) / 255.0 * 0.2).reshape(h, w, 1)
    lens_layer = np.clip(lens_layer.astype(np.float32) + 255.0 * h_f, 0, 255).astype(np.uint8)

    lf = (lens_mask.astype(np.float32) / 255.0).reshape(h, w, 1)
    overlay[:] = np.clip(overlay * (1 - lf) + lens_layer * lf, 0, 255).astype(np.uint8)
    mask[:] = np.maximum(mask, lens_mask)

    ff = (frame_mask.astype(np.float32) / 255.0).reshape(h, w, 1)
    overlay[:] = np.clip(overlay * (1 - ff) + frame_layer * ff, 0, 255).astype(np.uint8)
    mask[:] = np.maximum(mask, frame_mask)

    return inner


def _draw_arc_bridge(overlay, mask, p_left, p_right, sag, color, thick, alpha=255):
    mid = (p_left + p_right) / 2.0
    ctrl = mid + np.array([0, -sag])
    curve = _bezier_quad(p_left, ctrl, p_right, n=40)
    _draw_curve(overlay, mask, curve, color, thick, alpha)


def _draw_temples(overlay, mask, g, landmarks, w, h, frame_color, arm_thick, tip_color=None):
    """Hinge (outer eye 33 / 263) → temporal anchor (162 / 389); single straight segment per side."""
    thick = max(1, int(arm_thick))
    p_lo = (int(g["lo"][0]), int(g["lo"][1]))
    p_162 = (int(g["l_temple"][0]), int(g["l_temple"][1]))
    p_ro = (int(g["ro"][0]), int(g["ro"][1]))
    p_389 = (int(g["r_temple"][0]), int(g["r_temple"][1]))
    cv2.line(overlay, p_lo, p_162, frame_color, thick, cv2.LINE_AA)
    cv2.line(overlay, p_ro, p_389, frame_color, thick, cv2.LINE_AA)
    cv2.line(mask, p_lo, p_162, 255, thick, cv2.LINE_AA)
    cv2.line(mask, p_ro, p_389, 255, thick, cv2.LINE_AA)


def _sprite_model_path(kind: str) -> str:
    return os.path.join(_GLASSES_ROOT, kind, f"{kind}_model.png")


def _load_rgba_png(path: str) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        bgra[:, :, 3] = 255
    elif img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
    elif img.shape[2] == 4:
        bgra = img
    else:
        return None
    return bgra


def _affine_sprite_to_face_canvas(png_bgra: np.ndarray, g, cw: int, ch: int):
    """
    Warp sprite with a 3-point affine (stable vs. perspective): left eye outer, right eye
    outer, nose-bridge midpoint in image space. Horizontal template span is scaled from
    inter-eye-center distance relative to outer-eye span so overall size tracks eye_dist.
    """
    ph, pw = png_bgra.shape[:2]
    lens_y = ph * 0.48
    bridge_y = lens_y - ph * 0.18
    ref_outer = float(np.linalg.norm(g["ro"] - g["lo"]))
    ed = float(g["eye_dist"])
    bx = float(pw * 0.17) * (ed / max(ref_outer, 1e-6))
    bx = float(np.clip(bx, pw * 0.08, pw * 0.46))

    src = np.float32([[pw * 0.5 - bx, lens_y], [pw * 0.5 + bx, lens_y], [pw * 0.5, bridge_y]])
    bridge_dst = ((g["li"] + g["ri"]) * 0.5).astype(np.float32)
    dst = np.stack([g["lo"], g["ro"], bridge_dst]).astype(np.float32)

    m2 = cv2.getAffineTransform(src, dst)
    mx32 = m2.astype(np.float32)

    bgr_dst = png_bgra[:, :, :3]
    if png_bgra.shape[2] == 4:
        alpha_src = png_bgra[:, :, 3]
        warped_bgr = cv2.warpAffine(
            bgr_dst,
            mx32,
            (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_a = cv2.warpAffine(
            alpha_src,
            mx32,
            (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped_bgr, warped_a
    warped_bgr = cv2.warpAffine(
        bgr_dst,
        mx32,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_a = np.full((ch, cw), 255, dtype=np.uint8)
    return warped_bgr, warped_a


def _draw_from_sprite(
    image: np.ndarray,
    landmarks,
    w: int,
    h: int,
    kind: str,
    procedural_fallback,
):
    path = _sprite_model_path(kind)
    rgba = _load_rgba_png(path)
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    if rgba is None:
        return procedural_fallback(image, landmarks, w, h)

    g = _face_geometry(landmarks, w, h)
    warped_bgr, warped_a = _affine_sprite_to_face_canvas(rgba, g, w, h)
    if int(warped_a.max()) <= 4:
        return procedural_fallback(image, landmarks, w, h)

    overlay[:] = warped_bgr
    mask[:] = warped_a
    return image, overlay, mask


# ===================================================================
# MODEL 1: Metal Aviator
# ===================================================================
def _draw_aviator(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    mid = (g["lc"] + g["rc"]) / 2
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(2, int(ed * 0.02))
    frame_color = (160, 160, 170)

    lhw, lhh = int(g["l_eye_w"] * 0.68), int(g["l_eye_w"] * 0.68 * 0.82)
    rhw, rhh = int(g["r_eye_w"] * 0.68), int(g["r_eye_w"] * 0.68 * 0.82)
    lc = _rot_pts(_teardrop_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_teardrop_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    _composite_frame_and_lens(overlay, mask, lc, g["lc"], frame_color, frame_t, (45, 65, 35), 140, inner_factor=0.92)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"], frame_color, frame_t, (45, 65, 35), 140, inner_factor=0.92)

    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.06, frame_color, frame_t)
    off = np.array([0, -ed * 0.07])
    _line(overlay, mask, _rot(li + off, mid, ang), _rot(ri + off, mid, ang), frame_color, max(1, frame_t - 1))

    pr = max(2, int(ed * 0.018))
    bmid = (li + ri) / 2
    for np_ in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_[0]), int(np_[1])), pr, (180, 180, 185), -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_[0]), int(np_[1])), pr, 200, -1, cv2.LINE_AA)
        arm = _bezier_quad(np_, (np_ + bmid) / 2 + np.array([0, -ed * 0.02]), bmid, 20)
        _draw_curve(overlay, mask, arm, frame_color, max(1, frame_t - 1), 180)

    _draw_temples(overlay, mask, g, landmarks, w, h, frame_color, max(1, int(ed * 0.015)), (60, 60, 65))
    return image, overlay, mask


# ===================================================================
# MODEL 2: Acetate Wayfarer
# ===================================================================
def _draw_wayfarer(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(4, int(ed * 0.05))
    frame_outer = (15, 15, 18)
    frame_inner = (50, 45, 42)

    lhw, lhh = int(g["l_eye_w"] * 0.78), int(g["l_eye_w"] * 0.78 * 0.62)
    rhw, rhh = int(g["r_eye_w"] * 0.78), int(g["r_eye_w"] * 0.78 * 0.62)
    lc = _rot_pts(_wayfarer_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_wayfarer_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    li_c = _composite_frame_and_lens(
        overlay, mask, lc, g["lc"], frame_outer, frame_t, (25, 30, 35), 170, inner_factor=0.85
    )
    ri_c = _composite_frame_and_lens(
        overlay, mask, rc, g["rc"], frame_outer, frame_t, (25, 30, 35), 170, inner_factor=0.85
    )
    _draw_poly(overlay, mask, li_c, frame_inner, max(1, frame_t // 3), alpha=100)
    _draw_poly(overlay, mask, ri_c, frame_inner, max(1, frame_t // 3), alpha=100)

    li, ri = g["li"].copy(), g["ri"].copy()
    bt = max(3, int(ed * 0.035))
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.05, frame_outer, bt)
    kl = li + np.array([ed * 0.03, 0])
    kr = ri - np.array([ed * 0.03, 0])
    _draw_arc_bridge(overlay, mask, kl, kr, -ed * 0.03, frame_outer, max(2, int(ed * 0.025)))

    pr = max(2, int(ed * 0.014))
    for pos in [g["lo"], g["ro"]]:
        pp = pos + np.array([0, -lhh * 0.15])
        cv2.circle(overlay, (int(pp[0]), int(pp[1])), pr, (160, 155, 140), -1, cv2.LINE_AA)
        cv2.circle(mask, (int(pp[0]), int(pp[1])), pr, 255, -1, cv2.LINE_AA)

    arm_t = max(3, int(ed * 0.035))
    _draw_temples(overlay, mask, g, landmarks, w, h, frame_outer, arm_t)

    return image, overlay, mask


def _draw_square(image, landmarks, w, h):
    """More angular wayfarer-like frame (distinct square silhouette)."""
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(4, int(ed * 0.052))
    frame_outer = (22, 22, 28)
    frame_inner = (55, 52, 48)

    lhw, lhh = int(g["l_eye_w"] * 0.82), int(g["l_eye_w"] * 0.82 * 0.58)
    rhw, rhh = int(g["r_eye_w"] * 0.82), int(g["r_eye_w"] * 0.82 * 0.58)
    lc = _rot_pts(_wayfarer_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_wayfarer_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    li_c = _composite_frame_and_lens(
        overlay, mask, lc, g["lc"], frame_outer, frame_t, (30, 35, 45), 165, inner_factor=0.86
    )
    ri_c = _composite_frame_and_lens(
        overlay, mask, rc, g["rc"], frame_outer, frame_t, (30, 35, 45), 165, inner_factor=0.86
    )
    _draw_poly(overlay, mask, li_c, frame_inner, max(1, frame_t // 3), alpha=95)
    _draw_poly(overlay, mask, ri_c, frame_inner, max(1, frame_t // 3), alpha=95)

    li, ri = g["li"].copy(), g["ri"].copy()
    bt = max(3, int(ed * 0.038))
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.04, frame_outer, bt)

    arm_t = max(3, int(ed * 0.034))
    _draw_temples(overlay, mask, g, landmarks, w, h, frame_outer, arm_t)
    return image, overlay, mask


def _draw_retro(image, landmarks, w, h):
    """Thick acetate browline + soft round lower rim."""
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    mid = (g["lc"] + g["rc"]) / 2
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(5, int(ed * 0.058))
    frame_outer = (12, 10, 8)
    frame_bar = (40, 32, 24)

    lr = int(g["l_eye_w"] * 0.54)
    rr = int(g["r_eye_w"] * 0.54)
    lc = _rot_pts(_round_contour(g["lc"][0], g["lc"][1], lr), g["lc"], ang)
    rc = _rot_pts(_round_contour(g["rc"][0], g["rc"][1], rr), g["rc"], ang)

    _composite_frame_and_lens(overlay, mask, lc, g["lc"], frame_outer, frame_t, (40, 38, 35), 160, inner_factor=0.86)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"], frame_outer, frame_t, (40, 38, 35), 160, inner_factor=0.86)

    li, ri = g["li"].copy(), g["ri"].copy()
    brow_l = _rot(li + np.array([0, -ed * 0.07]), mid, ang)
    brow_r = _rot(ri + np.array([0, -ed * 0.07]), mid, ang)
    _draw_curve(overlay, mask, _bezier_quad(brow_l, (brow_l + brow_r) / 2 + np.array([0, -ed * 0.02]), brow_r, 40), frame_bar, frame_t, 220)
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.035, frame_outer, max(2, frame_t // 2))

    _draw_temples(overlay, mask, g, landmarks, w, h, frame_outer, max(3, int(ed * 0.036)))
    return image, overlay, mask


def _draw_cat_eye(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(3, int(ed * 0.042))
    frame_color = (18, 18, 22)

    lhw, lhh = int(g["l_eye_w"] * 0.74), int(g["l_eye_w"] * 0.74 * 0.58)
    rhw, rhh = int(g["r_eye_w"] * 0.74), int(g["r_eye_w"] * 0.74 * 0.58)
    lc = _rot_pts(_cat_eye_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_cat_eye_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    _composite_frame_and_lens(overlay, mask, lc, g["lc"], frame_color, frame_t, (55, 40, 75), 120, inner_factor=0.88)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"], frame_color, frame_t, (55, 40, 75), 120, inner_factor=0.88)

    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.048, frame_color, max(2, frame_t - 1))

    _draw_temples(overlay, mask, g, landmarks, w, h, frame_color, max(2, int(ed * 0.028)))
    return image, overlay, mask


def _futuristic_shield_lens(cx: float, cy: float, hw: float, hh: float, n: int = 70):
    """Wide wrap shield: pronounced upper brow shelf, compact lower rim."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        ct, st = math.cos(t), math.sin(t)
        brow = float(max(-st, 0.0)) ** 1.25
        xs = ct * hw * (1.0 + 0.24 * brow + 0.1 * abs(st))
        ys = st * hh * (0.86 + 0.2 * ct * ct + 0.08 * brow)
        pts.append([cx + xs, cy + ys])
    return np.array(pts, dtype=np.float64)


def _draw_futuristic(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    mid_ip = (g["lc"] + g["rc"]) / 2
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(7, int(ed * 0.072))
    frame_color = (0, 0, 0)

    lhw, lhh = int(g["l_eye_w"] * 0.93), int(g["l_eye_w"] * 0.93 * 0.51)
    rhw, rhh = int(g["r_eye_w"] * 0.93), int(g["r_eye_w"] * 0.93 * 0.51)
    lc = _rot_pts(_futuristic_shield_lens(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_futuristic_shield_lens(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    _composite_frame_and_lens(
        overlay, mask, lc, g["lc"], frame_color, frame_t,
        (5, 5, 5), 90, inner_factor=0.86,
    )
    _composite_frame_and_lens(
        overlay, mask, rc, g["rc"], frame_color, frame_t,
        (5, 5, 5), 90, inner_factor=0.86,
    )

    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.018, frame_color, max(6, frame_t - 4))

    arm_t = max(4, int(ed * 0.026))
    _draw_temples(overlay, mask, g, landmarks, w, h, frame_color, arm_t)
    return image, overlay, mask


def _draw_sport(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(3, int(ed * 0.026))
    frame_color = (50, 85, 110)

    lhw, lhh = int(g["l_eye_w"] * 0.82), int(g["l_eye_w"] * 0.82 * 0.55)
    rhw, rhh = int(g["r_eye_w"] * 0.82), int(g["r_eye_w"] * 0.82 * 0.55)
    lc = _rot_pts(_sport_wrap_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_sport_wrap_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    _composite_frame_and_lens(overlay, mask, lc, g["lc"], frame_color, frame_t, (20, 45, 60), 135, inner_factor=0.90)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"], frame_color, frame_t, (20, 45, 60), 135, inner_factor=0.90)

    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.03, frame_color, frame_t)
    off = np.array([0, -ed * 0.05])
    mid = (g["lc"] + g["rc"]) / 2
    _line(overlay, mask, _rot(li + off, mid, ang), _rot(ri + off, mid, ang), (25, 55, 75), max(1, frame_t - 1))

    _draw_temples(overlay, mask, g, landmarks, w, h, frame_color, max(2, int(ed * 0.02)), (15, 30, 40))
    return image, overlay, mask


def _sprite_round(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "round", _draw_round)


def _sprite_aviator(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "aviator", _draw_aviator)


def _sprite_wayfarer(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "wayfarer", _draw_wayfarer)


def _sprite_square(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "square", _draw_square)


def _sprite_retro(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "retro", _draw_retro)





def _sprite_sport(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "sport", _draw_sport)


def _sprite_futuristic(im, lm, w, h):
    return _draw_from_sprite(im, lm, w, h, "futuristic", _draw_futuristic)


# ===================================================================
# MODEL 3: Minimalist Round
# ===================================================================
def _draw_round(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]
    mid = (g["lc"] + g["rc"]) / 2
    ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(1, int(ed * 0.018))
    frame_color = (130, 170, 210)

    lr = int(g["l_eye_w"] * 0.48)
    rr = int(g["r_eye_w"] * 0.48)
    lc = _rot_pts(_round_contour(g["lc"][0], g["lc"][1], lr), g["lc"], ang)
    rc = _rot_pts(_round_contour(g["rc"][0], g["rc"][1], rr), g["rc"], ang)

    _composite_frame_and_lens(overlay, mask, lc, g["lc"], frame_color, frame_t, (220, 215, 210), 35, inner_factor=0.90)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"], frame_color, frame_t, (220, 215, 210), 35, inner_factor=0.90)

    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed * 0.045, frame_color, frame_t)

    pr = max(1, int(ed * 0.015))
    bmid = (li + ri) / 2
    for np_ in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_[0]), int(np_[1])), pr, frame_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_[0]), int(np_[1])), pr, 180, -1, cv2.LINE_AA)
        arm = _bezier_quad(np_, (np_ + bmid) / 2 + np.array([0, -ed * 0.015]), bmid, 15)
        _draw_curve(overlay, mask, arm, frame_color, max(1, frame_t - 1), 150)

    _draw_temples(overlay, mask, g, landmarks, w, h, frame_color, max(1, int(ed * 0.013)))
    return image, overlay, mask


MODEL_ALIASES = {
    "classic": "aviator",
    "horn_rimmed": "wayfarer",
    "kemik": "wayfarer",
    "sunglasses": "wayfarer",
    "reading": "round",
}


MODEL_DISPATCH = {
    "aviator": _sprite_aviator,
    "wayfarer": _sprite_wayfarer,
    "round": _sprite_round,
    "square": _sprite_square,
    "retro": _sprite_retro,
    "sport": _sprite_sport,
    "futuristic": _sprite_futuristic,
}


def _normalize_model_id(model_id: str) -> str:
    key = (model_id or "aviator").strip().lower()
    key = key.replace(" ", "_").replace("-", "_")
    return MODEL_ALIASES.get(key, key)


def apply_glasses(image, landmarks, model_id="aviator"):
    """
    Rasterize procedural or sprite-backed glasses overlay with correct alpha.

    Frame regions (mask >= 250) are composited at FULL opacity so solid frames
    (black, metal, acetate) never show the face bleeding through.
    Lens regions (mask < 250) use their actual alpha for tinted transparency.
    """
    h, w = image.shape[:2]
    if image.size == 0 or not landmarks or len(landmarks) < 468:
        return image.copy()

    key = _normalize_model_id(model_id)
    draw_fn = MODEL_DISPATCH.get(key, _sprite_aviator)
    frame = image.copy()
    _, overlay, mask = draw_fn(frame, landmarks, w, h)

    # Build alpha channel:
    #   - Frame pixels (mask >= 250) → alpha = 1.0 (fully opaque, no face bleed)
    #   - Lens/semi-transparent pixels → preserve original alpha ratio
    alpha = mask.astype(np.float32) / 255.0
    # Enforce full opacity on solid frame regions
    alpha[mask >= 250] = 1.0
    alpha = np.clip(alpha, 0.0, 1.0)

    a3 = alpha[:, :, np.newaxis]  # (h, w, 1) broadcasts to (h, w, 3)
    frame_f = frame.astype(np.float32)
    over_f = overlay.astype(np.float32)
    blended = frame_f * (1.0 - a3) + over_f * a3
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
