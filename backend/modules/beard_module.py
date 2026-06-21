import os
import logging
from typing import Optional
import math

import cv2
import numpy as np

logger = logging.getLogger("facial_pipeline.beard_module")

_ASSET_CACHE = {}

def _load_asset(filename: str) -> Optional[np.ndarray]:
    if filename in _ASSET_CACHE:
        return _ASSET_CACHE[filename]
        
    path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", filename)
    if not os.path.exists(path):
        logger.error("Beard asset not found: %s", path)
        return None
        
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        logger.error("Beard asset must be a valid PNG with 4 channels (RGBA): %s", path)
        return None
        
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    _ASSET_CACHE[filename] = rgba
    return rgba

def _sample_eyebrow_color(image_bgr: np.ndarray, landmarks: np.ndarray) -> tuple:
    """Samples the average BGR color of the eyebrows to use for tinting."""
    h, w = image_bgr.shape[:2]
    left_eb = [52, 53, 65, 55, 70, 63, 105, 66, 107, 282, 283, 295, 285, 300, 293, 334, 296, 336]
    
    pts = []
    for idx in left_eb:
        if idx < len(landmarks):
            px, py = int(landmarks[idx][0]), int(landmarks[idx][1])
            if 0 <= px < w and 0 <= py < h:
                pts.append(image_bgr[py, px])
                
    if len(pts) == 0:
        return (40, 40, 40)
        
    pts = np.array(pts, dtype=np.float32)
    avg_color = np.mean(pts, axis=0)
    return tuple(avg_color)

def _get_target_points(landmarks: np.ndarray, face_h: float) -> np.ndarray:
    points = []
    lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    for idx in lip_indices:
        if idx < len(landmarks): points.append(landmarks[idx][:2])
        
    sb_l_indices = [234, 93, 137]
    for idx in sb_l_indices:
        if idx < len(landmarks): points.append(landmarks[idx][:2])
        
    sb_r_indices = [454, 323, 366]
    for idx in sb_r_indices:
        if idx < len(landmarks): points.append(landmarks[idx][:2])
        
    jaw_indices = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
    
    offset_y = face_h * 0.15 
    center_x = landmarks[152][0] if 152 < len(landmarks) else 0
    
    for idx in jaw_indices:
        if idx < len(landmarks):
            px, py = landmarks[idx][:2]
            outward_x = (px - center_x) * 0.1
            points.append([px + outward_x, py + offset_y])
            
    # Anchor the top of the mustache under the nose
    if 164 < len(landmarks):
        points.append(landmarks[164][:2])
            
    return np.array(points, dtype=np.float32)

def _get_source_points(asset_rgba: np.ndarray) -> np.ndarray:
    alpha = asset_rgba[..., 3]
    _, bin_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = cv2.boundingRect(bin_mask)
    if w == 0 or h == 0:
        return np.zeros((43, 2), dtype=np.float32)
        
    mouth_bbox = None
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                mx, my, mw, mh = cv2.boundingRect(cnt)
                if mw > w * 0.1 and mh > h * 0.05:
                    mouth_bbox = (mx, my, mw, mh)
                    break
    
    if not mouth_bbox:
        mx, my = x + int(w * 0.3), y + int(h * 0.2)
        mw, mh = int(w * 0.4), int(h * 0.2)
    else:
        mx, my, mw, mh = mouth_bbox
        
    points = []
    
    cx, cy = mx + mw/2.0, my + mh/2.0
    rx, ry = mw/2.0, mh/2.0
    for i in range(20):
        angle = math.pi - (i * 2 * math.pi / 20)
        px = cx + rx * math.cos(angle)
        py = cy - ry * math.sin(angle)
        points.append([px, py])
        
    points.append([x, y])
    points.append([x, y + h * 0.1])
    points.append([x, y + h * 0.2])
    
    points.append([x + w, y])
    points.append([x + w, y + h * 0.1])
    points.append([x + w, y + h * 0.2])
    
    for i in range(17):
        t = i / 16.0 
        px = x + w * t
        py = y + h * 0.3 + (h * 0.7) * (1.0 - 4 * (t - 0.5)**2)
        points.append([px, py])
        
    # Find the top of the mustache by raycasting upwards from the mouth center
    mustache_top_y = int(my)
    cx_int = int(cx)
    if 0 <= cx_int < bin_mask.shape[1]:
        while mustache_top_y > y and bin_mask[mustache_top_y, cx_int] > 0:
            mustache_top_y -= 1
            
    points.append([cx, mustache_top_y])
        
    return np.array(points, dtype=np.float32)

def _get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    
    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []
    
    for t in triangle_list:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        ind = []
        for j in range(3):
            for k in range(len(points)):
                if abs(pt[j][0] - points[k][0]) < 2.0 and abs(pt[j][1] - points[k][1]) < 2.0:
                    ind.append(k)
                    break
        if len(ind) == 3:
            delaunay_tri.append((ind[0], ind[1], ind[2]))
            
    return delaunay_tri

def _apply_piecewise_affine(image_bgr: np.ndarray, landmarks: np.ndarray, asset_filename: str) -> np.ndarray:
    asset_rgba = _load_asset(asset_filename)
    if asset_rgba is None:
        return image_bgr.copy()
        
    ih, iw = image_bgr.shape[:2]
    if len(landmarks) <= 152:
        return image_bgr.copy()
        
    face_h = landmarks[152][1] - landmarks[10][1]
    
    # 1. Dynamic Color Tinting (Preserve Luminance)
    eb_bgr = _sample_eyebrow_color(image_bgr, landmarks)
    eb_hsv = cv2.cvtColor(np.uint8([[eb_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    asset_rgb = asset_rgba[..., :3].copy()
    asset_alpha = asset_rgba[..., 3].copy()
    
    asset_hsv = cv2.cvtColor(asset_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    asset_hsv[..., 0] = eb_hsv[0]        # Match Hue
    asset_hsv[..., 1] = eb_hsv[1] * 0.8    # Match Saturation
    # STRICT CONSTRAINT: Do NOT touch asset_hsv[..., 2] (Value/Luminance). Preserve deep blacks!
    
    asset_rgb_tinted = cv2.cvtColor(asset_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    asset_rgba_tinted = np.zeros_like(asset_rgba)
    asset_rgba_tinted[..., :3] = asset_rgb_tinted
    asset_rgba_tinted[..., 3] = asset_alpha
    
    # 2. Meshing
    src_pts = _get_source_points(asset_rgba)
    dst_pts = _get_target_points(landmarks, face_h)
    
    if len(src_pts) != len(dst_pts) or len(src_pts) < 3:
        return image_bgr.copy()
        
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, iw-1)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, ih-1)
    
    rect = (0, 0, iw, ih)
    dt = _get_delaunay_triangles(rect, dst_pts)
    
    if len(dt) == 0:
        return image_bgr.copy()
        
    warped_rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
    
    for i in range(len(dt)):
        t_src = [src_pts[dt[i][0]], src_pts[dt[i][1]], src_pts[dt[i][2]]]
        t_dst = [dst_pts[dt[i][0]], dst_pts[dt[i][1]], dst_pts[dt[i][2]]]
        
        r_src = cv2.boundingRect(np.float32([t_src]))
        r_dst = cv2.boundingRect(np.float32([t_dst]))
        
        r_src_x, r_src_y, r_src_w, r_src_h = r_src
        r_dst_x, r_dst_y, r_dst_w, r_dst_h = r_dst
        
        if r_src_w <= 0 or r_src_h <= 0 or r_dst_w <= 0 or r_dst_h <= 0:
            continue
            
        t_src_rect = []
        t_dst_rect = []
        
        for j in range(3):
            t_src_rect.append(((t_src[j][0] - r_src_x), (t_src[j][1] - r_src_y)))
            t_dst_rect.append(((t_dst[j][0] - r_dst_x), (t_dst[j][1] - r_dst_y)))
            
        warp_mat = cv2.getAffineTransform(np.float32(t_src_rect), np.float32(t_dst_rect))
        
        src_y_start, src_y_end = max(0, r_src_y), min(asset_rgba_tinted.shape[0], r_src_y + r_src_h)
        src_x_start, src_x_end = max(0, r_src_x), min(asset_rgba_tinted.shape[1], r_src_x + r_src_w)
        
        src_crop = np.zeros((r_src_h, r_src_w, 4), dtype=np.uint8)
        crop_h = src_y_end - src_y_start
        crop_w = src_x_end - src_x_start
        if crop_h <= 0 or crop_w <= 0:
            continue
            
        src_crop[:crop_h, :crop_w] = asset_rgba_tinted[src_y_start:src_y_end, src_x_start:src_x_end]
            
        dst_crop = cv2.warpAffine(src_crop, warp_mat, (r_dst_w, r_dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        mask = np.zeros((r_dst_h, r_dst_w, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)
        
        mask4 = np.zeros((r_dst_h, r_dst_w, 4), dtype=np.float32)
        mask4[:,:,:3] = mask
        mask4[:,:,3] = mask[:,:,0]
        
        dst_crop_float = dst_crop.astype(np.float32) * mask4
        
        dst_y_start, dst_y_end = max(0, r_dst_y), min(ih, r_dst_y + r_dst_h)
        dst_x_start, dst_x_end = max(0, r_dst_x), min(iw, r_dst_x + r_dst_w)
        
        roi_h = dst_y_end - dst_y_start
        roi_w = dst_x_end - dst_x_start
        if roi_h <= 0 or roi_w <= 0:
            continue
            
        warped_roi = warped_rgba[dst_y_start:dst_y_end, dst_x_start:dst_x_end].astype(np.float32)
        dst_crop_clp = dst_crop_float[:roi_h, :roi_w]
        mask4_clp = mask4[:roi_h, :roi_w]
        
        warped_roi = warped_roi * (1 - mask4_clp) + dst_crop_clp
        warped_rgba[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = warped_roi.astype(np.uint8)
        
    warped_rgb = warped_rgba[..., :3]
    warped_alpha = warped_rgba[..., 3].astype(np.float32) / 255.0
        
    # 3. Mouth Cutout Dilation & Feathering (Applied ONLY to Alpha)
    lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    lip_points = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in lip_indices if idx < len(landmarks)], dtype=np.int32)
    
    if len(lip_points) > 0:
        mouth_mask = np.zeros((ih, iw), dtype=np.float32)
        cv2.fillPoly(mouth_mask, [lip_points], 1.0)
        
        dilate_size = int(face_h * 0.04)
        if dilate_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=1)
            
        blur_size = int(face_h * 0.08) | 1
        if blur_size > 1:
            mouth_mask = cv2.GaussianBlur(mouth_mask, (blur_size, blur_size), 0)
            
        warped_alpha = warped_alpha * (1.0 - mouth_mask)

    # 4. Global Feathering (Applied ONLY to Alpha to keep RGB core solid/punchy)
    blur_kernel = int(face_h * 0.03) | 1
    if blur_kernel > 1:
        warped_alpha = cv2.GaussianBlur(warped_alpha, (blur_kernel, blur_kernel), 0)

    # 5. Strict Alpha Compositing
    asset_alpha_3d = warped_alpha[..., None]
    image_bgr_float = image_bgr.astype(np.float32)
    warped_bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    
    result_bgr = (warped_bgr * asset_alpha_3d) + (image_bgr_float * (1.0 - asset_alpha_3d))
    
    return np.clip(result_bgr, 0, 255).astype(np.uint8)

def _apply_mustache_affine(image_bgr: np.ndarray, landmarks: np.ndarray, asset_filename: str) -> np.ndarray:
    asset_rgba = _load_asset(asset_filename)
    if asset_rgba is None:
        return image_bgr.copy()
        
    ih, iw = image_bgr.shape[:2]
    if len(landmarks) < 292:
        return image_bgr.copy()
        
    face_h = landmarks[152][1] - landmarks[10][1]
    
    # 1. Dynamic Color Tinting
    eb_bgr = _sample_eyebrow_color(image_bgr, landmarks)
    eb_hsv = cv2.cvtColor(np.uint8([[eb_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    asset_rgb = asset_rgba[..., :3].copy()
    asset_alpha = asset_rgba[..., 3].copy()
    
    asset_hsv = cv2.cvtColor(asset_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    asset_hsv[..., 0] = eb_hsv[0]
    asset_hsv[..., 1] = eb_hsv[1] * 0.8
    asset_rgb_tinted = cv2.cvtColor(asset_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    asset_rgba_tinted = np.zeros_like(asset_rgba)
    asset_rgba_tinted[..., :3] = asset_rgb_tinted
    asset_rgba_tinted[..., 3] = asset_alpha
    
    # 2. Extract Source Bounding Box Points
    pts_y, pts_x = np.where(asset_alpha > 0)
    if len(pts_x) == 0: return image_bgr.copy()
    
    min_x, max_x = np.min(pts_x), np.max(pts_x)
    min_y, max_y = np.min(pts_y), np.max(pts_y)
    
    # 3. Define Target Points on Face (Orthogonal Center-Anchored Paste)
    pt_61 = np.array(landmarks[61][:2], dtype=np.float32)
    pt_291 = np.array(landmarks[291][:2], dtype=np.float32)
    
    # Guarantee left-to-right orientation for robust vectors
    if pt_61[0] < pt_291[0]:
        pt_left, pt_right = pt_61, pt_291
    else:
        pt_left, pt_right = pt_291, pt_61
        
    vec_h = pt_right - pt_left
    mouth_w = np.linalg.norm(vec_h)
    if mouth_w <= 0: mouth_w = 1.0
    vec_h = vec_h / mouth_w
    
    # vec_v points perpendicular down (positive Y in OpenCV)
    vec_v = np.array([-vec_h[1], vec_h[0]], dtype=np.float32)
    
    # The user explicitly wants the EXACT CENTER of the asset on the 3 landmarks in the middle of the philtrum
    pt_164 = np.array(landmarks[164][:2], dtype=np.float32)
    pt_0 = np.array(landmarks[0][:2], dtype=np.float32)
    pt_target_center = (pt_164 + pt_0) / 2.0
    
    # Enlarge width to 130% of mouth width for a full, glorious mustache
    target_w = mouth_w * 1.7
    
    src_w = max_x - min_x
    if src_w <= 0: src_w = 1.0
    scale = target_w / src_w
    target_h = (max_y - min_y) * scale
    
    # Construct an orthogonal target bounding box centered exactly on pt_target_center
    dst_pt1 = pt_target_center - vec_h * (target_w / 2) - vec_v * (target_h / 2) # Top-Left
    dst_pt2 = pt_target_center + vec_h * (target_w / 2) - vec_v * (target_h / 2) # Top-Right
    dst_pt3 = pt_target_center - vec_h * (target_w / 2) + vec_v * (target_h / 2) # Bottom-Left
    
    dst_pts = np.float32([dst_pt1, dst_pt2, dst_pt3])
    
    # Map from the exact corners of the source bounding box
    src_pt1 = [min_x, min_y] # Top-Left
    src_pt2 = [max_x, min_y] # Top-Right
    src_pt3 = [min_x, max_y] # Bottom-Left
    
    src_pts = np.float32([src_pt1, src_pt2, src_pt3])
    
    # 4. Single Unified Affine Warp (Perfect Centered Sticker Paste)
    warp_mat = cv2.getAffineTransform(src_pts, dst_pts)
    warped_rgba = cv2.warpAffine(asset_rgba_tinted, warp_mat, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    warped_rgb = warped_rgba[..., :3]
    warped_alpha = warped_rgba[..., 3].astype(np.float32) / 255.0
    
    # The user specifically requested: "dudağın önüne asset geçebilir" 
    # Therefore, the Mouth Cutout mask logic has been intentionally removed from the mustache.
    
    # 5. Global Feathering
    blur_kernel = int(face_h * 0.02) | 1
    if blur_kernel > 1:
        warped_alpha = cv2.GaussianBlur(warped_alpha, (blur_kernel, blur_kernel), 0)
        
    # 6. Strict Alpha Compositing
    asset_alpha_3d = warped_alpha[..., None]
    image_bgr_float = image_bgr.astype(np.float32)
    warped_bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    
    result_bgr = (warped_bgr * asset_alpha_3d) + (image_bgr_float * (1.0 - asset_alpha_3d))
    return np.clip(result_bgr, 0, 255).astype(np.uint8)

# Public API wrappers
def apply_beard(image_bgr: np.ndarray, *args, **kwargs) -> np.ndarray:
    try:
        landmarks = kwargs.get("landmarks", args[0] if len(args) > 0 else None)
        if landmarks is None: return image_bgr.copy()
        if getattr(landmarks, "ndim", 0) == 2 and landmarks.shape[1] >= 3:
            landmarks = landmarks[:, :2].copy()
        return _apply_piecewise_affine(image_bgr, landmarks, "beard.png")
    except Exception as exc:
        logger.error("apply_beard failed: %s", exc)
        return image_bgr.copy()

def apply_mustache(image_bgr: np.ndarray, *args, **kwargs) -> np.ndarray:
    try:
        landmarks = kwargs.get("landmarks", args[0] if len(args) > 0 else None)
        if landmarks is None: return image_bgr.copy()
        if getattr(landmarks, "ndim", 0) == 2 and landmarks.shape[1] >= 3:
            landmarks = landmarks[:, :2].copy()
        return _apply_mustache_affine(image_bgr, landmarks, "mustache.png")
    except Exception as exc:
        logger.error("apply_mustache failed: %s", exc)
        return image_bgr.copy()

def apply_goatee(image_bgr: np.ndarray, *args, **kwargs) -> np.ndarray:
    try:
        landmarks = kwargs.get("landmarks", args[0] if len(args) > 0 else None)
        if landmarks is None: return image_bgr.copy()
        if getattr(landmarks, "ndim", 0) == 2 and landmarks.shape[1] >= 3:
            landmarks = landmarks[:, :2].copy()
        return _apply_piecewise_affine(image_bgr, landmarks, "goatee.png")
    except Exception as exc:
        logger.error("apply_goatee failed: %s", exc)
        return image_bgr.copy()
