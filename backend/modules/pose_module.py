from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PoseEstimate:
    yaw: float
    pitch: float
    roll: float
    confidence: float


def estimate_head_pose(landmarks: np.ndarray, width: int, height: int) -> PoseEstimate:
    """Estimate head pose and a conservative confidence score."""
    if landmarks is None or landmarks.shape[0] < 468:
        return PoseEstimate(0.0, 0.0, 0.0, 0.0)

    model_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -63.0, -12.0],
            [-45.0, 32.0, -24.0],
            [45.0, 32.0, -24.0],
            [-34.0, -28.0, -20.0],
            [34.0, -28.0, -20.0],
        ],
        dtype=np.float32,
    )
    image_points = np.array(
        [landmarks[1], landmarks[152], landmarks[33], landmarks[263], landmarks[61], landmarks[291]],
        dtype=np.float32,
    )

    focal = float(max(width, height))
    cam = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    ok, rvec, _tvec = cv2.solvePnP(model_points, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return PoseEstimate(0.0, 0.0, 0.0, 0.0)

    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = 0.0

    abs_yaw = abs(float(yaw))
    abs_pitch = abs(float(pitch))
    abs_roll = abs(float(roll))
    extreme = max(abs_yaw / 65.0, abs_pitch / 55.0, abs_roll / 45.0)
    confidence = float(np.clip(1.0 - extreme, 0.0, 1.0))
    return PoseEstimate(float(yaw), float(pitch), float(roll), confidence)


def landmark_stability_confidence(
    landmarks: np.ndarray,
    prev_landmarks: Optional[np.ndarray],
    frame_diag: float,
) -> float:
    if landmarks is None:
        return 0.0
    if prev_landmarks is None or prev_landmarks.shape != landmarks.shape:
        return 1.0
    motion = float(np.mean(np.linalg.norm(landmarks - prev_landmarks, axis=1)))
    normalized = motion / max(frame_diag * 0.035, 1e-6)
    return float(np.clip(1.0 - normalized, 0.0, 1.0))
