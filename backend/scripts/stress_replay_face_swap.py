from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

try:
    from modules.face_swap_module import face_swap_engine
    from modules.warping_module import PersistentFaceMesh
except ModuleNotFoundError:
    from backend.modules.face_swap_module import face_swap_engine
    from backend.modules.warping_module import PersistentFaceMesh


def degrade_frame(frame: np.ndarray, idx: int) -> np.ndarray:
    out = frame.copy()
    mode = idx % 5
    if mode == 0:
        out = cv2.GaussianBlur(out, (9, 9), 2.0)
    elif mode == 1:
        out = cv2.convertScaleAbs(out, alpha=0.75, beta=-25)
    elif mode == 2:
        noise = np.random.normal(0, 12, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif mode == 3:
        h, w = out.shape[:2]
        x0 = int(w * 0.35)
        y0 = int(h * 0.2)
        cv2.rectangle(out, (x0, y0), (x0 + int(w * 0.3), y0 + int(h * 0.28)), (0, 0, 0), -1)
    else:
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 8.0, 1.0)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stress replay for live face swap.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    src = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if src is None:
        raise RuntimeError("Cannot read source image.")
    ok, src_buf = cv2.imencode(".png", src)
    if not ok:
        raise RuntimeError("Cannot encode source image.")
    face_swap_engine.process_source_image(src_buf.tobytes())

    mesh = PersistentFaceMesh()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    recovered = 0
    lost = 0
    total = 0
    t0 = time.perf_counter()
    while total < args.max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        total += 1
        stressed = degrade_frame(frame, total)
        lm = mesh.detect(stressed)
        if lm is None:
            lost += 1
            out = stressed
        else:
            out = face_swap_engine.apply_face_swap(stressed, lm, runtime_hints={"degraded_mode": True})
            recovered += 1
        if args.preview:
            cv2.imshow("stress_input", stressed)
            cv2.imshow("stress_output", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    elapsed = max(time.perf_counter() - t0, 1e-6)
    cap.release()
    mesh.close()
    cv2.destroyAllWindows()
    print(f"frames={total}")
    print(f"fps={total/elapsed:.2f}")
    print(f"tracking_loss_rate={(lost / total) if total else 0.0:.3f}")
    print(f"recovery_rate={(recovered / total) if total else 0.0:.3f}")


if __name__ == "__main__":
    main()
