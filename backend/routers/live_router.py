"""
Live Router -- WebSocket-based realtime facial filter pipeline.

Architecture:
  Browser webcam (getUserMedia) --> base64 JPEG via WebSocket -->
  Backend warping pipeline (PersistentFaceMesh + geometric warp) -->
  base64 JPEG result via WebSocket --> Browser display

Endpoints:
  WS   /live/ws          Realtime frame processing WebSocket
  GET  /live/status       Pipeline health check
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"])


def _hex_to_opencv_hue(hex_str: str) -> int:
    """Parse '#RRGGBB' hex color to OpenCV HSV hue (0-179)."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) < 6:
        return 0
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    d = max_c - min_c
    if d < 1e-6:
        h = 0.0
    elif max_c == r:
        h = 60.0 * (((g - b) / d) % 6)
    elif max_c == g:
        h = 60.0 * ((b - r) / d + 2)
    else:
        h = 60.0 * ((r - g) / d + 4)
    if h < 0:
        h += 360.0
    return int(round(h / 2.0))  # OpenCV hue range 0-179

# -- Import warping module (dual-path) --
try:
    from modules.warping_module import (
        PersistentFaceMesh,
        apply_smile,
        apply_eyebrow_raise,
        apply_lip_widen,
        apply_face_slim,
        apply_eye_scaling,
        apply_beard,
        apply_emoji_preset,
    )
    from modules.frequency_module import (
        apply_aging,
        apply_deaging,
        apply_fft_filter,
        apply_cartoon_filter,
        apply_virtual_makeup,
        apply_virtual_makeup_fallback,
    )
    from modules.input_module import get_landmarks, preprocess_image
    from modules.face_swap_module import face_swap_engine, FaceSwapError
except ModuleNotFoundError:
    from backend.modules.warping_module import (
        PersistentFaceMesh,
        apply_smile,
        apply_eyebrow_raise,
        apply_lip_widen,
        apply_face_slim,
        apply_eye_scaling,
        apply_beard,
        apply_emoji_preset,
    )
    from backend.modules.frequency_module import (
        apply_aging,
        apply_deaging,
        apply_fft_filter,
        apply_cartoon_filter,
        apply_virtual_makeup,
        apply_virtual_makeup_fallback,
    )
    from backend.modules.input_module import get_landmarks, preprocess_image
    from backend.modules.face_swap_module import face_swap_engine, FaceSwapError

# Emoji presets from process.py (lazy import to avoid circular)
def _get_emoji_presets_map():
    try:
        from routers.process import _EMOJI_PRESETS_MAP
    except (ModuleNotFoundError, ImportError):
        from backend.routers.process import _EMOJI_PRESETS_MAP
    return _EMOJI_PRESETS_MAP

# Glasses (lazy import)
def _get_apply_glasses():
    try:
        from modules.glasses_module import apply_glasses
    except ModuleNotFoundError:
        from backend.modules.glasses_module import apply_glasses
    return apply_glasses

# Hair color (lazy import)
def _get_apply_hair_color():
    try:
        from modules.hair_module import apply_hair_color
    except ModuleNotFoundError:
        from backend.modules.hair_module import apply_hair_color
    return apply_hair_color


# -- Shared persistent face mesh (one per server lifetime) --
_face_mesh: Optional[PersistentFaceMesh] = None
_mesh_lock = asyncio.Lock()


async def _get_face_mesh() -> PersistentFaceMesh:
    """Lazily create and return a single PersistentFaceMesh instance."""
    global _face_mesh
    async with _mesh_lock:
        if _face_mesh is None:
            _face_mesh = PersistentFaceMesh()
            logger.info("PersistentFaceMesh created for live router")
    return _face_mesh


# -- EMA smoother for browser-based pipeline --
class _BrowserSmoother:
    """Simple per-connection EMA landmark smoother."""

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def smooth(self, pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pts is None:
            return self.prev  # hold last good landmarks briefly
        if self.prev is None or self.prev.shape != pts.shape:
            self.prev = pts.copy()
            return pts
        smoothed = self.alpha * pts + (1.0 - self.alpha) * self.prev
        self.prev = smoothed.copy()
        return smoothed


def _decode_frame(data_url: str) -> Optional[np.ndarray]:
    """Decode a data:image/jpeg;base64,... string into a BGR ndarray."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.debug("Frame decode failed: %s", exc)
        return None


def _encode_frame(img: np.ndarray, quality: int = 70) -> str:
    """Encode a BGR frame as a data:image/jpeg;base64,... string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _apply_filter(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    config: dict,
) -> np.ndarray:
    """Apply the requested filter using existing warping/processing functions."""
    filter_name = config.get("filter", "none")
    intensity = config.get("intensity", 50)

    if filter_name == "none":
        return frame

    try:
        # ── Geometric Warps (use persistent landmarks) ──
        if filter_name == "smile":
            return apply_smile(frame, intensity, landmarks=landmarks)
        elif filter_name == "eyebrow_raise":
            return apply_eyebrow_raise(frame, intensity, landmarks=landmarks)
        elif filter_name == "lip_widen":
            return apply_lip_widen(frame, intensity, landmarks=landmarks)
        elif filter_name == "face_slim":
            return apply_face_slim(frame, intensity, landmarks=landmarks)
        elif filter_name == "eye_scaling":
            return apply_eye_scaling(frame, intensity, landmarks=landmarks)
        elif filter_name == "beard":
            return apply_beard(frame, intensity, landmarks=landmarks)
        elif filter_name.startswith("emoji_"):
            emoji_name = filter_name.split("_", 1)[1]
            return apply_emoji_preset(frame, emoji_name, landmarks=landmarks)

        # ── Emoji Presets from process.py (alien, robot, clown etc.) ──
        # All presets receive the pre-smoothed landmarks from the live
        # pipeline so they skip redundant MediaPipe detection and track
        # with zero jitter — same architecture as _apply_robot.
        elif filter_name in ("alien", "robot", "clown", "star_eyes", "heart_eyes", "crying"):
            presets_map = _get_emoji_presets_map()
            fn = presets_map.get(filter_name)
            if fn:
                return fn(frame, landmarks=landmarks)

        # ── Makeup ──
        elif filter_name.startswith("makeup_"):
            region = filter_name.split("_", 1)[1]  # lips, eyeshadow, blush
            hue = config.get("makeup_hue", 0)
            # Use makeup_color hex string (more reliable) if provided
            makeup_color_hex = config.get("makeup_color", "")
            if makeup_color_hex and isinstance(makeup_color_hex, str) and len(makeup_color_hex) >= 4:
                hue = _hex_to_opencv_hue(makeup_color_hex)
            opacity = config.get("makeup_opacity", 0.5)
            # Realtime pipeline: smoothed pixel landmarks → normalize to [0,1]
            try:
                if landmarks is not None:
                    h_f, w_f = frame.shape[:2]
                    lm_list = [[float(pt[0]) / w_f, float(pt[1]) / h_f] for pt in landmarks]
                else:
                    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lm_list = get_landmarks(preprocess_image(rgb_img))
                return apply_virtual_makeup(
                    image=frame, landmarks=lm_list,
                    region=region, hue=int(hue), opacity=float(opacity),
                )
            except Exception as exc:
                logger.debug("Live makeup landmark path failed; using fallback: %s", exc)
                return apply_virtual_makeup_fallback(
                    image=frame,
                    region=region,
                    hue=int(hue),
                    opacity=float(opacity),
                )

        # ── Glasses ──
        elif filter_name == "glasses":
            glasses_type = config.get("glasses_type", "aviator")
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm_list = get_landmarks(preprocess_image(rgb_img))
            apply_glasses = _get_apply_glasses()
            return apply_glasses(frame, lm_list, glasses_type)

        # ── Hair Color ──
        elif filter_name == "hair_color":
            hair_color = config.get("hair_color", "255,0,0")
            hair_intensity = config.get("hair_intensity", 0.6)
            apply_hair = _get_apply_hair_color()
            return apply_hair(frame, hair_color, hair_intensity)

        # ── Aging / De-aging ──
        elif filter_name == "aging":
            return apply_aging(frame, intensity)
        elif filter_name == "deaging":
            return apply_deaging(frame, intensity)

        # ── Cartoon ──
        elif filter_name == "cartoon":
            return apply_cartoon_filter(frame)

        # ── Face Swap ──
        elif filter_name == "face_swap":
            if face_swap_engine.is_loaded:
                return face_swap_engine.apply_face_swap(frame, landmarks)
            else:
                logger.debug("Face swap: face_swap_engine not loaded")

        else:
            logger.debug("Unknown live filter: %s", filter_name)

    except Exception as exc:
        logger.warning("Filter '%s' failed: %s", filter_name, exc)

    return frame


# -- WebSocket endpoint --

@router.websocket("/ws")
async def live_websocket(ws: WebSocket):
    """
    Realtime facial filter pipeline via WebSocket.

    Client protocol (JSON messages):
      -> { "type": "frame", "data": "<base64 jpeg>" }
      -> { "type": "config", "filter": "smile", "intensity": 50, ... }
      -> { "action": "update_live_state", "active_states": { "glasses": {...}, "smile": {...} } }

      <- { "type": "frame", "data": "<base64 jpeg>", "fps": 25.3, "face_detected": true }
      <- { "type": "status", "message": "..." }
      <- { "type": "state_ack", "active_states": [...] }
    """
    await ws.accept()
    logger.info("Live WebSocket connected")

    mesh = await _get_face_mesh()
    smoother = _BrowserSmoother(alpha=0.7)

    # ── Stacked active_states dictionary ──
    # Keys are feature names (e.g. "glasses", "smile", "makeup_lips")
    # Values are parameter dicts for that feature
    active_states: dict[str, dict] = {}

    # SourceFaceCache is now handled globally by face_swap_engine

    # Legacy single-filter config (kept for backward compatibility)
    current_config = {
        "filter": "none",
        "intensity": 50,
        "makeup_hue": 0,
        "makeup_opacity": 0.5,
        "glasses_type": "aviator",
        "hair_color": "255,0,0",
        "hair_intensity": 0.6,
        "beard_type": "beard",
        "beard_darkness": 60,
    }
    frame_count = 0
    fps = 0.0
    last_fps_time = time.perf_counter()
    fps_frame_count = 0

    try:
        while True:
            raw_msg = await ws.receive_text()

            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            msg_action = msg.get("action", "")

            # ── NEW: Stacked state update ──
            if msg_action == "update_live_state":
                incoming = msg.get("active_states", {})
                # Merge: null/None values → remove, otherwise upsert
                for feature, params in incoming.items():
                    if params is None:
                        active_states.pop(feature, None)
                    else:
                        active_states[feature] = params if isinstance(params, dict) else {"value": params}
                logger.info("Live active_states updated: %s", list(active_states.keys()))
                await ws.send_json({
                    "type": "state_ack",
                    "active_states": list(active_states.keys()),
                    "source_face_loaded": face_swap_engine.is_loaded,
                })
                continue

            # ── Face swap source upload ──
            # Handled via HTTP POST now, but we keep clear_source_face fallback if used
            if msg_action == "clear_source_face":
                face_swap_engine.is_loaded = False
                active_states.pop("face_swap", None)
                await ws.send_json({
                    "type": "status",
                    "message": "Source face cleared",
                    "source_face_loaded": False,
                })
                continue

            # ── Legacy config update (backward compat) ──
            if msg_type == "config":
                changed = False
                for key in current_config:
                    if key in msg and msg[key] != current_config[key]:
                        current_config[key] = msg[key]
                        changed = True
                # Auto-migrate legacy config into active_states
                filt = current_config.get("filter", "none")
                if filt and filt != "none":
                    active_states = {filt: dict(current_config)}
                elif not filt or filt == "none":
                    active_states.clear()
                if changed:
                    logger.info("Live config updated (legacy): %s", current_config)
                    await ws.send_json({
                        "type": "status",
                        "message": f"Config: {current_config['filter']}, Intensity: {current_config['intensity']}%",
                    })
                continue

            # ── Frame processing ──
            if msg_type == "frame":
                frame_data = msg.get("data", "")
                frame = _decode_frame(frame_data)
                if frame is None:
                    continue

                # Run heavy processing in a thread
                result, face_detected = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _process_frame_sync,
                    frame, mesh, smoother, active_states,
                )

                # FPS calculation
                frame_count += 1
                fps_frame_count += 1
                now = time.perf_counter()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    last_fps_time = now

                encoded = _encode_frame(result, quality=94)
                await ws.send_json({
                    "type": "frame",
                    "data": encoded,
                    "fps": round(fps, 1),
                    "face_detected": face_detected,
                })

    except WebSocketDisconnect:
        logger.info("Live WebSocket disconnected")
    except Exception as exc:
        logger.error("Live WebSocket error: %s", exc)
    finally:
        logger.info("Live WebSocket session ended (frames: %d)", frame_count)


def _feature_to_config(feature: str, params: dict) -> dict:
    """Convert a feature name + params into a config dict for _apply_filter."""
    config = dict(params)  # copy
    # Map feature name → filter name expected by _apply_filter
    _FEATURE_TO_FILTER = {
        "smile": "smile", "eyebrow": "eyebrow_raise", "lip": "lip_widen",
        "slim": "face_slim", "eye_scale": "eye_scaling",
        "beard": "beard",
        "alien": "alien", "robot": "robot", "clown": "clown",
        "star_eyes": "star_eyes", "heart_eyes": "heart_eyes", "crying": "crying",
        "makeup_lips": "makeup_lips", "makeup_eyeshadow": "makeup_eyeshadow",
        "makeup_blush": "makeup_blush", "makeup_eyeliner": "makeup_eyeliner",
        "makeup_mascara": "makeup_mascara",
        "glasses": "glasses", "hair_color": "hair_color",
        "aging": "aging", "deaging": "deaging", "cartoon": "cartoon",
        "face_swap": "face_swap",
    }
    config["filter"] = _FEATURE_TO_FILTER.get(feature, feature)
    config.setdefault("intensity", 50)
    return config


def _draw_landmarks_overlay(
    frame: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    """Draw 468 face mesh landmark points on the frame."""
    out = frame.copy()
    color = (0, 255, 200)  # bright cyan-green
    radius = max(1, int(min(out.shape[:2]) * 0.003))
    for pt in landmarks:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return out


def _process_frame_sync(
    frame: np.ndarray,
    mesh: PersistentFaceMesh,
    smoother: _BrowserSmoother,
    active_states: dict,
) -> tuple[np.ndarray, bool]:
    """Synchronous frame processing — applies ALL active states sequentially."""
    # 0. Extract show_landmarks — independent toggle, NOT a filter.
    show_landmarks = "show_landmarks" in active_states
    # Build filter-only states (exclude show_landmarks)
    filter_states = {k: v for k, v in active_states.items() if k != "show_landmarks"}

    # 1. Detect landmarks
    raw_landmarks = mesh.detect(frame)

    # 2. Smooth
    smoothed = smoother.smooth(raw_landmarks)
    face_detected = smoothed is not None

    # 3. Apply all stacked filters sequentially. Makeup has an approximate
    # fallback, so it should not flash off during brief landmark misses.
    has_makeup = any(str(feature).startswith("makeup_") for feature in filter_states)
    if not filter_states and not show_landmarks:
        if not face_detected:
            return frame, face_detected
        # No filters but show_landmarks is on → skip to step 4
        if show_landmarks and smoothed is not None:
            return _draw_landmarks_overlay(frame, smoothed), face_detected
        return frame, face_detected

    if not face_detected and not has_makeup and not show_landmarks:
        return frame, face_detected

    result = frame
    for feature, params in filter_states.items():
        try:
            config = _feature_to_config(feature, params)
            filter_landmarks = smoothed
            if raw_landmarks is None and str(feature).startswith("makeup_"):
                filter_landmarks = None
            result = _apply_filter(result, filter_landmarks, config)
        except Exception as exc:
            logger.warning("Stacked filter '%s' failed: %s", feature, exc)

    # 4. Draw landmarks AFTER all filters (on top of everything)
    if show_landmarks and smoothed is not None:
        result = _draw_landmarks_overlay(result, smoothed)

    return result, face_detected


# -- Health check --

@router.get("/status")
async def live_status():
    """Check live pipeline readiness."""
    return {
        "ready": _face_mesh is not None,
        "backend": _face_mesh._backend if _face_mesh else None,
    }
