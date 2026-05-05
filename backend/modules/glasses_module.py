"""
Advanced Procedural 3D Glasses Rendering Module
================================================
Photorealistic, face-adaptive parametric glasses using OpenCV/NumPy.
Three models: Metal Aviator, Acetate Wayfarer, Minimalist Round.
"""
import cv2
import numpy as np
import math

def _lm_px(landmarks, idx, w, h):
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
    # Temple arm endpoints: use landmark 162/389 (upper temple, where
    # ear helix meets skull) NOT 234/454 (mid-ear tragion).
    l_temple = _lm_px(landmarks, 162, w, h)
    r_temple = _lm_px(landmarks, 389, w, h)
    l_nose = _lm_px(landmarks, 198, w, h)
    r_nose = _lm_px(landmarks, 420, w, h)
    return {
        "lc": lc, "rc": rc, "lo": lo, "li": li, "ri": ri, "ro": ro,
        "nose": nose, "nose_tip": nose_tip, "face_w": face_w,
        "angle": angle, "eye_dist": eye_dist, "bridge_w": bridge_w,
        "l_eye_w": l_eye_w, "r_eye_w": r_eye_w,
        "l_temple": l_temple, "r_temple": r_temple,
        "l_nose": l_nose, "r_nose": r_nose,
    }

# --- geometry helpers ---
def _rot(pt, center, angle):
    c, s = math.cos(angle), math.sin(angle)
    d = pt - center
    return center + np.array([c*d[0] - s*d[1], s*d[0] + c*d[1]])

def _rot_pts(pts, center, angle):
    return np.array([_rot(p, center, angle) for p in pts])

def _bezier_quad(p0, p1, p2, n=30):
    ts = np.linspace(0, 1, n)
    return np.array([(1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2 for t in ts])

def _shrink(contour, center, factor):
    return center + (contour - center) * factor

# --- drawing helpers ---
def _draw_curve(canvas, mask, pts, color, thick, alpha=255):
    ip = pts.astype(np.int32).reshape((-1,1,2))
    cv2.polylines(canvas, [ip], False, color, thick, cv2.LINE_AA)
    cv2.polylines(mask, [ip], False, alpha, thick, cv2.LINE_AA)

def _line(canvas, mask, p1, p2, color, thick, alpha=255):
    cv2.line(canvas, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),
             color, thick, cv2.LINE_AA)
    cv2.line(mask, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),
             alpha, thick, cv2.LINE_AA)

def _fill_poly(canvas, mask, pts, color, alpha=200):
    ip = pts.astype(np.int32)
    cv2.fillPoly(canvas, [ip], color, cv2.LINE_AA)
    cv2.fillPoly(mask, [ip], alpha, cv2.LINE_AA)

def _draw_poly(canvas, mask, pts, color, thick, closed=True, alpha=255):
    ip = pts.astype(np.int32).reshape((-1,1,2))
    cv2.polylines(canvas, [ip], closed, color, thick, cv2.LINE_AA)
    cv2.polylines(mask, [ip], closed, alpha, thick, cv2.LINE_AA)

# --- contour generators ---
def _teardrop_contour(cx, cy, hw, hh, n=60):
    pts = []
    for i in range(n):
        t = 2*math.pi*i/n
        x, y = math.cos(t), math.sin(t)
        if y > 0:
            x *= (1.0 - 0.25*y)
            y *= 1.15
        else:
            y *= 0.85
        pts.append([cx + x*hw, cy + y*hh])
    return np.array(pts, dtype=np.float64)

def _wayfarer_contour(cx, cy, hw, hh, n=60):
    top_w, bot_w = hw, hw*0.85
    pts = []
    for i in range(n):
        t = 2*math.pi*i/n
        ct, st = math.cos(t), math.sin(t)
        exp = 2.8
        sx = abs(ct)**(2.0/exp) * np.sign(ct)
        sy = abs(st)**(2.0/exp) * np.sign(st)
        wf = top_w if sy <= 0 else top_w - (top_w - bot_w)*sy
        pts.append([cx + sx*wf, cy + sy*hh])
    return np.array(pts, dtype=np.float64)

def _round_contour(cx, cy, r, n=60):
    return np.array([[cx+math.cos(2*math.pi*i/n)*r,
                      cy+math.sin(2*math.pi*i/n)*r] for i in range(n)],
                    dtype=np.float64)

# --- Proper hollow-frame + lens compositing ---
def _composite_frame_and_lens(overlay, mask, contour, center,
                              frame_color, frame_thick,
                              lens_tint, lens_alpha,
                              inner_factor=0.88):
    """
    Boolean-subtraction compositing:
    1. Draw solid frame band (outer contour filled, then inner socket erased)
    2. Fill lens tint into the inner socket only
    3. Add specular highlight clipped to lens
    """
    h, w = overlay.shape[:2]
    inner = _shrink(contour, center, inner_factor)
    ipts_out = contour.astype(np.int32)
    ipts_in = inner.astype(np.int32)

    # --- Frame layer: fill outer, erase inner (boolean subtract) ---
    frame_layer = np.zeros_like(overlay)
    frame_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(frame_layer, [ipts_out], frame_color, cv2.LINE_AA)
    cv2.fillPoly(frame_mask, [ipts_out], 255, cv2.LINE_AA)
    # Erase the inner lens socket from the frame
    cv2.fillPoly(frame_layer, [ipts_in], (0,0,0), cv2.LINE_AA)
    cv2.fillPoly(frame_mask, [ipts_in], 0, cv2.LINE_AA)

    # --- Lens layer: fill inner socket with tint ---
    lens_layer = np.zeros_like(overlay)
    lens_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_layer, [ipts_in], lens_tint, cv2.LINE_AA)
    cv2.fillPoly(lens_mask, [ipts_in], lens_alpha, cv2.LINE_AA)

    # Specular highlight (upper-left crescent, clipped to lens)
    cx_f = inner[:, 0].mean()
    cy_f = inner[:, 1].mean()
    rx = (inner[:, 0].max() - inner[:, 0].min()) / 2
    ry = (inner[:, 1].max() - inner[:, 1].min()) / 2
    hx, hy = int(cx_f - rx*0.2), int(cy_f - ry*0.3)
    hr = max(3, int(min(rx, ry)*0.3))
    h_m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(h_m, (hx, hy), hr, 255, -1, cv2.LINE_AA)
    h_m = cv2.GaussianBlur(h_m, (0,0), max(1, hr*0.4))
    lens_rgn = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_rgn, [ipts_in], 255, cv2.LINE_AA)
    h_m = cv2.bitwise_and(h_m, lens_rgn)
    h_f = (h_m.astype(np.float32)/255.0*0.2).reshape(h, w, 1)
    lens_layer = np.clip(lens_layer.astype(np.float32) +
                         255.0*h_f, 0, 255).astype(np.uint8)

    # Composite: lens first, then frame on top
    # Lens onto overlay
    lf = (lens_mask.astype(np.float32)/255.0).reshape(h, w, 1)
    overlay[:] = np.clip(overlay*(1-lf) + lens_layer*lf, 0, 255).astype(np.uint8)
    mask[:] = np.maximum(mask, lens_mask)
    # Frame onto overlay
    ff = (frame_mask.astype(np.float32)/255.0).reshape(h, w, 1)
    overlay[:] = np.clip(overlay*(1-ff) + frame_layer*ff, 0, 255).astype(np.uint8)
    mask[:] = np.maximum(mask, frame_mask)

    return inner  # return inner contour for optional bevel


def _draw_arc_bridge(overlay, mask, p_left, p_right, sag, color, thick, alpha=255):
    mid = (p_left + p_right) / 2.0
    ctrl = mid + np.array([0, -sag])
    curve = _bezier_quad(p_left, ctrl, p_right, n=40)
    _draw_curve(overlay, mask, curve, color, thick, alpha)

# --- temple arm drawing (biologically correct above-ear placement) ---
def _draw_temples(overlay, mask, g, frame_color, arm_thick, tip_color=None):
    ed = g["eye_dist"]
    # Arm endpoints: upper temple landmarks (162/389) = ear helix
    # junction, where glasses actually rest above the ear
    l_end = g["l_temple"]
    r_end = g["r_temple"]
    _line(overlay, mask, g["lo"], l_end, frame_color, arm_thick, 220)
    _line(overlay, mask, g["ro"], r_end, frame_color, arm_thick, 220)
    # Arm tips curve down behind the ear
    tc = tip_color or frame_color
    l_tip = l_end + np.array([-ed*0.04, ed*0.08])
    r_tip = r_end + np.array([ ed*0.04, ed*0.08])
    _line(overlay, mask, l_end, l_tip, tc, arm_thick, 200)
    _line(overlay, mask, r_end, r_tip, tc, arm_thick, 200)

# ===================================================================
# MODEL 1: Metal Aviator
# ===================================================================
def _draw_aviator(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]; mid = (g["lc"]+g["rc"])/2; ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(2, int(ed*0.02))
    frame_color = (160, 160, 170)

    # Teardrop contours
    lhw, lhh = int(g["l_eye_w"]*0.68), int(g["l_eye_w"]*0.68*0.82)
    rhw, rhh = int(g["r_eye_w"]*0.68), int(g["r_eye_w"]*0.68*0.82)
    lc = _rot_pts(_teardrop_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_teardrop_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    # Hollow frame + lens (proper boolean subtraction)
    _composite_frame_and_lens(overlay, mask, lc, g["lc"],
                              frame_color, frame_t, (45,65,35), 140,
                              inner_factor=0.92)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"],
                              frame_color, frame_t, (45,65,35), 140,
                              inner_factor=0.92)

    # Smooth arc bridge
    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed*0.06, frame_color, frame_t)
    # Double bridge top bar
    off = np.array([0, -ed*0.07])
    _line(overlay, mask, _rot(li+off, mid, ang), _rot(ri+off, mid, ang),
          frame_color, max(1, frame_t-1))

    # Nose pads
    pr = max(2, int(ed*0.018)); bmid = (li+ri)/2
    for np_ in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_[0]),int(np_[1])), pr, (180,180,185), -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_[0]),int(np_[1])), pr, 200, -1, cv2.LINE_AA)
        arm = _bezier_quad(np_, (np_+bmid)/2+np.array([0,-ed*0.02]), bmid, 20)
        _draw_curve(overlay, mask, arm, frame_color, max(1,frame_t-1), 180)

    # Temple arms (above-ear anchoring)
    _draw_temples(overlay, mask, g, frame_color, max(1,int(ed*0.015)), (60,60,65))
    return image, overlay, mask

# ===================================================================
# MODEL 2: Acetate Wayfarer
# ===================================================================
def _draw_wayfarer(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]; mid = (g["lc"]+g["rc"])/2; ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(4, int(ed*0.05))
    frame_outer = (15, 15, 18)
    frame_inner = (50, 45, 42)

    # Wayfarer contours
    lhw, lhh = int(g["l_eye_w"]*0.78), int(g["l_eye_w"]*0.78*0.62)
    rhw, rhh = int(g["r_eye_w"]*0.78), int(g["r_eye_w"]*0.78*0.62)
    lc = _rot_pts(_wayfarer_contour(g["lc"][0], g["lc"][1], lhw, lhh), g["lc"], ang)
    rc = _rot_pts(_wayfarer_contour(g["rc"][0], g["rc"][1], rhw, rhh), g["rc"], ang)

    # Hollow frame + lens
    li_c = _composite_frame_and_lens(overlay, mask, lc, g["lc"],
                                     frame_outer, frame_t, (25,30,35), 170,
                                     inner_factor=0.85)
    ri_c = _composite_frame_and_lens(overlay, mask, rc, g["rc"],
                                     frame_outer, frame_t, (25,30,35), 170,
                                     inner_factor=0.85)
    # Inner bevel highlight
    _draw_poly(overlay, mask, li_c, frame_inner, max(1,frame_t//3), alpha=100)
    _draw_poly(overlay, mask, ri_c, frame_inner, max(1,frame_t//3), alpha=100)

    # Smooth keyhole bridge
    li, ri = g["li"].copy(), g["ri"].copy()
    bt = max(3, int(ed*0.035))
    _draw_arc_bridge(overlay, mask, li, ri, ed*0.05, frame_outer, bt)
    kl = li + np.array([ed*0.03, 0]); kr = ri - np.array([ed*0.03, 0])
    _draw_arc_bridge(overlay, mask, kl, kr, -ed*0.03, frame_outer, max(2,int(ed*0.025)))

    # Hinge pins
    pr = max(2, int(ed*0.014))
    for pos in [g["lo"], g["ro"]]:
        pp = pos + np.array([0, -lhh*0.15])
        cv2.circle(overlay, (int(pp[0]),int(pp[1])), pr, (160,155,140), -1, cv2.LINE_AA)
        cv2.circle(mask, (int(pp[0]),int(pp[1])), pr, 255, -1, cv2.LINE_AA)

    # Temple arms (above-ear)
    arm_t = max(3, int(ed*0.035))
    _draw_temples(overlay, mask, g, frame_outer, arm_t)
    # Highlight stripe
    l_end, r_end = g["l_temple"], g["r_temple"]
    _line(overlay, mask, g["lo"], l_end, frame_inner, max(1,arm_t//3), 70)
    _line(overlay, mask, g["ro"], r_end, frame_inner, max(1,arm_t//3), 70)

    return image, overlay, mask

# ===================================================================
# MODEL 3: Minimalist Round
# ===================================================================
def _draw_round(image, landmarks, w, h):
    g = _face_geometry(landmarks, w, h)
    ang = g["angle"]; mid = (g["lc"]+g["rc"])/2; ed = g["eye_dist"]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_t = max(1, int(ed*0.018))
    frame_color = (130, 170, 210)

    # Round contours
    lr = int(g["l_eye_w"]*0.48); rr = int(g["r_eye_w"]*0.48)
    lc = _rot_pts(_round_contour(g["lc"][0], g["lc"][1], lr), g["lc"], ang)
    rc = _rot_pts(_round_contour(g["rc"][0], g["rc"][1], rr), g["rc"], ang)

    # Hollow frame + lens
    _composite_frame_and_lens(overlay, mask, lc, g["lc"],
                              frame_color, frame_t, (220,215,210), 35,
                              inner_factor=0.90)
    _composite_frame_and_lens(overlay, mask, rc, g["rc"],
                              frame_color, frame_t, (220,215,210), 35,
                              inner_factor=0.90)

    # Smooth arc bridge
    li, ri = g["li"].copy(), g["ri"].copy()
    _draw_arc_bridge(overlay, mask, li, ri, ed*0.045, frame_color, frame_t)

    # Nose pads
    pr = max(1, int(ed*0.015)); bmid = (li+ri)/2
    for np_ in [g["l_nose"], g["r_nose"]]:
        cv2.circle(overlay, (int(np_[0]),int(np_[1])), pr, frame_color, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(np_[0]),int(np_[1])), pr, 180, -1, cv2.LINE_AA)
        arm = _bezier_quad(np_, (np_+bmid)/2+np.array([0,-ed*0.015]), bmid, 15)
        _draw_curve(overlay, mask, arm, frame_color, max(1,frame_t-1), 150)

    # Temple arms (above-ear)
    _draw_temples(overlay, mask, g, frame_color, max(1,int(ed*0.013)))
    return image, overlay, mask

# ===================================================================
# Public API
# ===================================================================
MODEL_DISPATCH = {
    "aviator": _draw_aviator,
    "wayfarer": _draw_wayfarer,
    "round": _draw_round,
    "sunglasses": _draw_wayfarer,
    "reading": _draw_round,
}

def apply_glasses(image, landmarks, model_id="aviator"):
    h, w = image.shape[:2]
    draw_fn = MODEL_DISPATCH.get(model_id.lower(), _draw_aviator)
    processed = image.copy()
    processed, overlay, mask = draw_fn(processed, landmarks, w, h)
    mask_blur = cv2.GaussianBlur(mask, (5,5), 1.5)
    mf = mask_blur.astype(np.float32)/255.0
    m3 = np.stack([mf]*3, axis=2)
    return (processed*(1-m3) + overlay*m3).astype(np.uint8)
