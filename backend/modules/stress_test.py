"""
Stress Testing Infrastructure for the Live Face Swap Pipeline.

Provides automated tests for difficult real-world conditions:
  - Rapid head motion
  - Low-light frames
  - Partial face occlusion
  - Extreme pose angles
  - Sustained high-load processing

Usage (standalone):
    python -m backend.modules.stress_test --source path/to/face.jpg

Usage (programmatic):
    tester = FaceSwapStressTester()
    tester.load_source("path/to/face.jpg")
    report = tester.run_all()
    print(report)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

try:
    from modules.face_swap_module import face_swap_engine, FaceSwapError
    from modules.warping_module import (
        PersistentFaceMesh,
        detect_face_landmarks,
        estimate_head_pose,
        validate_landmarks,
    )
except ModuleNotFoundError:
    from backend.modules.face_swap_module import face_swap_engine, FaceSwapError
    from backend.modules.warping_module import (
        PersistentFaceMesh,
        detect_face_landmarks,
        estimate_head_pose,
        validate_landmarks,
    )

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single stress test scenario."""
    test_name: str
    frames_total: int = 0
    frames_success: int = 0
    frames_failed: int = 0
    fps_values: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    pose_angles: List[dict] = field(default_factory=list)
    detection_confidence: List[bool] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def fps_min(self) -> float:
        return min(self.fps_values) if self.fps_values else 0.0

    @property
    def fps_max(self) -> float:
        return max(self.fps_values) if self.fps_values else 0.0

    @property
    def fps_avg(self) -> float:
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0.0

    @property
    def fps_p95(self) -> float:
        if not self.fps_values:
            return 0.0
        sorted_fps = sorted(self.fps_values)
        idx = int(len(sorted_fps) * 0.05)  # p5 of FPS = worst 5%
        return sorted_fps[idx]

    @property
    def latency_avg_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def latency_p95_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        sorted_lat = sorted(self.latency_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def success_rate(self) -> float:
        return self.frames_success / max(1, self.frames_total)

    @property
    def detection_rate(self) -> float:
        if not self.detection_confidence:
            return 0.0
        return sum(self.detection_confidence) / len(self.detection_confidence)

    def summary(self) -> dict:
        return {
            "test_name": self.test_name,
            "frames_total": self.frames_total,
            "frames_success": self.frames_success,
            "frames_failed": self.frames_failed,
            "success_rate": f"{self.success_rate:.1%}",
            "detection_rate": f"{self.detection_rate:.1%}",
            "fps": {
                "min": round(self.fps_min, 1),
                "max": round(self.fps_max, 1),
                "avg": round(self.fps_avg, 1),
                "p95": round(self.fps_p95, 1),
            },
            "latency_ms": {
                "avg": round(self.latency_avg_ms, 1),
                "p95": round(self.latency_p95_ms, 1),
            },
            "duration_s": round(self.duration_s, 2),
            "errors_count": len(self.errors),
        }


class FaceSwapStressTester:
    """Automated stress testing for the live face swap pipeline.

    Each test generates synthetic frames that simulate difficult conditions,
    runs them through the full swap pipeline, and collects metrics.
    """

    def __init__(self) -> None:
        self._mesh: Optional[PersistentFaceMesh] = None
        self._source_loaded: bool = False

    def load_source(self, source_path: str) -> bool:
        """Load a source face image for testing."""
        img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Cannot read source image: %s", source_path)
            return False
        try:
            ok, buf = cv2.imencode(".png", img)
            if ok:
                face_swap_engine.process_source_image(buf.tobytes())
                self._source_loaded = True
                logger.info("Stress test source loaded: %s", source_path)
                return True
        except FaceSwapError as e:
            logger.error("Failed to load source for stress test: %s", e)
        return False

    def _get_mesh(self) -> PersistentFaceMesh:
        if self._mesh is None:
            self._mesh = PersistentFaceMesh()
        return self._mesh

    def _process_test_frame(
        self, frame: np.ndarray, result: TestResult
    ) -> None:
        """Run a single frame through the pipeline and record metrics."""
        result.frames_total += 1
        t_start = time.perf_counter()

        try:
            mesh = self._get_mesh()
            landmarks = mesh.detect(frame)
            detected = validate_landmarks(landmarks)
            result.detection_confidence.append(detected)

            if not detected:
                result.frames_failed += 1
                return

            # Get pose
            h, w = frame.shape[:2]
            yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
            result.pose_angles.append({"yaw": yaw, "pitch": pitch, "roll": roll})

            # Apply face swap
            if self._source_loaded and face_swap_engine.is_loaded:
                swapped = face_swap_engine.apply_face_swap(frame, landmarks)
                if swapped is not None and swapped is not frame:
                    result.frames_success += 1
                else:
                    result.frames_failed += 1
            else:
                result.frames_failed += 1

        except Exception as exc:
            result.frames_failed += 1
            result.errors.append(str(exc))

        finally:
            t_end = time.perf_counter()
            dt = t_end - t_start
            result.latency_ms.append(dt * 1000.0)
            if dt > 0:
                result.fps_values.append(1.0 / dt)

    # ── Test Scenarios ────────────────────────────────────────────────────

    def test_rapid_motion(
        self,
        base_frame: np.ndarray,
        n_frames: int = 60,
    ) -> TestResult:
        """Simulate rapid head motion via translation + rotation jitter.

        Tests recovery from sudden movements and motion blur.
        """
        result = TestResult(test_name="rapid_motion")
        t_start = time.perf_counter()
        h, w = base_frame.shape[:2]

        for i in range(n_frames):
            # Random affine transform simulating head motion
            angle = np.random.uniform(-15, 15)
            tx = np.random.uniform(-w * 0.1, w * 0.1)
            ty = np.random.uniform(-h * 0.1, h * 0.1)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            frame = cv2.warpAffine(base_frame, M, (w, h))

            # Add motion blur on some frames
            if i % 3 == 0:
                ksize = np.random.choice([5, 9, 13])
                kernel = np.zeros((ksize, ksize))
                kernel[ksize // 2, :] = np.ones(ksize) / ksize
                frame = cv2.filter2D(frame, -1, kernel)

            self._process_test_frame(frame, result)

        result.duration_s = time.perf_counter() - t_start
        logger.info("test_rapid_motion: %s", result.summary())
        return result

    def test_low_light(
        self,
        base_frame: np.ndarray,
        n_frames: int = 40,
    ) -> TestResult:
        """Simulate low-light by darkening and adding noise."""
        result = TestResult(test_name="low_light")
        t_start = time.perf_counter()

        for i in range(n_frames):
            # Progressive darkening
            darkness = 0.15 + 0.6 * (i / max(1, n_frames - 1))
            dark = (base_frame.astype(np.float32) * darkness).astype(np.uint8)

            # Add Gaussian noise (camera sensor noise in low light)
            noise = np.random.normal(0, 15, dark.shape).astype(np.float32)
            noisy = np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            self._process_test_frame(noisy, result)

        result.duration_s = time.perf_counter() - t_start
        logger.info("test_low_light: %s", result.summary())
        return result

    def test_partial_occlusion(
        self,
        base_frame: np.ndarray,
        n_frames: int = 40,
    ) -> TestResult:
        """Simulate partial face visibility via cropping and occlusion bars."""
        result = TestResult(test_name="partial_occlusion")
        t_start = time.perf_counter()
        h, w = base_frame.shape[:2]

        for i in range(n_frames):
            frame = base_frame.copy()

            # Random occlusion: black rectangle over part of the face
            occ_type = i % 4
            if occ_type == 0:
                # Left side occluded
                frame[:, :w // 3] = 0
            elif occ_type == 1:
                # Right side occluded
                frame[:, 2 * w // 3:] = 0
            elif occ_type == 2:
                # Bottom half occluded (chin/mouth)
                frame[2 * h // 3:, :] = 0
            else:
                # Random rectangle occlusion
                rx = np.random.randint(0, w // 2)
                ry = np.random.randint(0, h // 2)
                rw = np.random.randint(w // 6, w // 3)
                rh = np.random.randint(h // 6, h // 3)
                frame[ry:ry + rh, rx:rx + rw] = 0

            self._process_test_frame(frame, result)

        result.duration_s = time.perf_counter() - t_start
        logger.info("test_partial_occlusion: %s", result.summary())
        return result

    def test_extreme_pose(
        self,
        base_frame: np.ndarray,
        n_frames: int = 50,
    ) -> TestResult:
        """Simulate extreme head rotations via perspective transforms."""
        result = TestResult(test_name="extreme_pose")
        t_start = time.perf_counter()
        h, w = base_frame.shape[:2]

        for i in range(n_frames):
            # Simulate perspective distortion (like head turning)
            strength = (i / max(1, n_frames - 1)) * 0.4  # 0 → 0.4

            pts_src = np.float32([
                [0, 0], [w, 0], [w, h], [0, h]
            ])

            # Alternate left/right turn
            if i % 2 == 0:
                offset = int(w * strength * 0.3)
                pts_dst = np.float32([
                    [offset, int(h * strength * 0.15)],
                    [w, 0],
                    [w, h],
                    [offset, h - int(h * strength * 0.15)],
                ])
            else:
                offset = int(w * strength * 0.3)
                pts_dst = np.float32([
                    [0, 0],
                    [w - offset, int(h * strength * 0.15)],
                    [w - offset, h - int(h * strength * 0.15)],
                    [0, h],
                ])

            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            frame = cv2.warpPerspective(base_frame, M, (w, h))

            self._process_test_frame(frame, result)

        result.duration_s = time.perf_counter() - t_start
        logger.info("test_extreme_pose: %s", result.summary())
        return result

    # ── Run All ───────────────────────────────────────────────────────────

    def run_all(
        self,
        base_frame: Optional[np.ndarray] = None,
        webcam_src: int = 0,
    ) -> dict:
        """Run all stress tests and return a combined report.

        Parameters
        ----------
        base_frame : np.ndarray | None
            A reference frame with a visible face. If None, captures from webcam.
        webcam_src : int
            Webcam index (used only if base_frame is None).

        Returns
        -------
        dict
            Combined report with per-test summaries and overall statistics.
        """
        if base_frame is None:
            cap = cv2.VideoCapture(webcam_src)
            if not cap.isOpened():
                return {"error": "Cannot open webcam"}
            for _ in range(10):
                ok, base_frame = cap.read()
                if ok and base_frame is not None:
                    break
            cap.release()
            if base_frame is None:
                return {"error": "Failed to capture reference frame"}

        if not self._source_loaded:
            return {"error": "No source face loaded. Call load_source() first."}

        results = {}
        logger.info("=== STRESS TEST SUITE START ===")

        for test_fn in [
            self.test_rapid_motion,
            self.test_low_light,
            self.test_partial_occlusion,
            self.test_extreme_pose,
        ]:
            try:
                r = test_fn(base_frame)
                results[r.test_name] = r.summary()
            except Exception as exc:
                results[test_fn.__name__] = {"error": str(exc)}
                logger.error("Stress test %s failed: %s", test_fn.__name__, exc)

        # Overall summary
        total_frames = sum(
            r.get("frames_total", 0) for r in results.values() if isinstance(r, dict)
        )
        total_success = sum(
            r.get("frames_success", 0) for r in results.values() if isinstance(r, dict)
        )

        report = {
            "tests": results,
            "overall": {
                "total_frames": total_frames,
                "total_success": total_success,
                "overall_success_rate": f"{total_success / max(1, total_frames):.1%}",
                "engine_stats": face_swap_engine.stats,
            },
        }

        logger.info("=== STRESS TEST SUITE COMPLETE ===")
        logger.info("Overall: %d/%d frames succeeded (%.1f%%)",
                     total_success, total_frames,
                     total_success / max(1, total_frames) * 100)

        return report

    def close(self) -> None:
        """Release resources."""
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Face Swap Stress Tester")
    parser.add_argument("--source", required=True, help="Path to source face image")
    parser.add_argument("--target", default=None, help="Path to target/reference frame (or webcam if omitted)")
    args = parser.parse_args()

    tester = FaceSwapStressTester()
    if not tester.load_source(args.source):
        print("ERROR: Failed to load source face")
        exit(1)

    target = None
    if args.target:
        target = cv2.imread(args.target, cv2.IMREAD_COLOR)
        if target is None:
            print(f"ERROR: Cannot read target image: {args.target}")
            exit(1)

    report = tester.run_all(base_frame=target)
    print("\n" + "=" * 60)
    print("STRESS TEST REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))

    tester.close()
