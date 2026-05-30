from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from modules.face_swap_module import face_swap_engine
    from modules.warping_module import PersistentFaceMesh
except ModuleNotFoundError:
    from backend.modules.face_swap_module import face_swap_engine
    from backend.modules.warping_module import PersistentFaceMesh


class LandmarkEma:
    def __init__(self, alpha: float = 0.72):
        self.alpha = alpha
        self.prev = None

    def smooth(self, pts):
        if pts is None:
            return self.prev
        if self.prev is None or self.prev.shape != pts.shape:
            self.prev = pts.copy()
            return pts
        smoothed = self.alpha * pts + (1.0 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed


def main():
    parser = argparse.ArgumentParser(description="Benchmark live face swap pipeline.")
    parser.add_argument("--source", required=True, help="Source face image path")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--output-dir", default="backend/assets/benchmark_results")
    args = parser.parse_args()

    source = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot read source image: {args.source}")
    ok, src_buf = cv2.imencode(".png", source)
    if not ok:
        raise RuntimeError("Could not encode source image.")
    face_swap_engine.process_source_image(src_buf.tobytes())

    mesh = PersistentFaceMesh()
    smoother = LandmarkEma(alpha=0.72)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    latencies = []
    jitter = []
    consistency = []
    tracking_loss = 0
    total = 0
    prev_lm = None
    prev_out = None
    face_roi = None

    t0 = time.perf_counter()
    while total < args.max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        total += 1
        start = time.perf_counter()
        lm = mesh.detect(frame)
        lm = smoother.smooth(lm)
        if lm is None:
            tracking_loss += 1
            out = frame
        else:
            out = face_swap_engine.apply_face_swap(frame, lm)
            if prev_lm is not None and prev_lm.shape == lm.shape:
                jitter.append(float(np.mean(np.linalg.norm(lm - prev_lm, axis=1))))
            prev_lm = lm.copy()
            oval = np.array([lm[i] for i in range(min(len(lm), 468)) if i in set([10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109])], dtype=np.float32)
            if len(oval) >= 3:
                x, y, w, h = cv2.boundingRect(np.int32(oval))
                face_roi = (x, y, w, h)
        if prev_out is not None and face_roi is not None:
            x, y, w, h = face_roi
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            if x2 > x and y2 > y:
                diff = cv2.absdiff(out[y:y2, x:x2], prev_out[y:y2, x:x2])
                consistency.append(float(np.mean(diff)))
        prev_out = out.copy()
        latencies.append((time.perf_counter() - start) * 1000.0)
    elapsed = time.perf_counter() - t0
    cap.release()
    mesh.close()

    fps = total / elapsed if elapsed > 0 else 0.0
    result = {
        "frames": total,
        "fps_avg": fps,
        "fps_std_est": float(np.std([1000.0 / max(v, 1e-6) for v in latencies])) if latencies else 0.0,
        "latency_ms_avg": float(np.mean(latencies)) if latencies else 0.0,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "landmark_jitter_avg": float(np.mean(jitter)) if jitter else 0.0,
        "swap_consistency_diff_avg": float(np.mean(consistency)) if consistency else 0.0,
        "tracking_loss_rate": (tracking_loss / total) if total else 0.0,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"face_swap_benchmark_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved={os.fspath(out_path)}")


if __name__ == "__main__":
    main()
