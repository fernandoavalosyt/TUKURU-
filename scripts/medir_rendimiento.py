"""Mide pose sobre el mismo video en 1-3 camaras simuladas, sin Gemini.

Ejemplo: python scripts/medir_rendimiento.py --device cpu --cameras 3
Los modos predict-lock y predict-batch miden solo inferencia; no preservan IDs.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tukuru.optimizer import FrameOptimizer  # noqa: E402
from tukuru.vision import PoseTracker, ensure_model, fp16_arguments  # noqa: E402


def frames_from_video(path: Path, fps: float, max_frames: int, max_side: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {path}")
    stride = max(1, round((capture.get(cv2.CAP_PROP_FPS) or 25.0) / fps))
    optimizer = FrameOptimizer(max_side)
    frames = []
    index = 0
    while len(frames) < max_frames:
        ok, image = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(optimizer.process(image)[0])
        index += 1
    capture.release()
    return frames


def wrist_counts(kpts: np.ndarray) -> tuple[int, int]:
    if len(kpts) == 0:
        return 0, 0
    wrists = kpts[:, [9, 10], 2]
    return int((wrists >= 0.40).sum()), int(wrists.size)


def run_trackers(frames: list[np.ndarray], model_path: str, device: str, max_side: int,
                 cameras: int) -> tuple[float, list[float], int, int]:
    trackers = [PoseTracker(model_path, device, 0.40, max_side) for _ in range(cameras)]
    durations: list[float] = []
    visible = possible = 0

    def worker(tracker: PoseTracker) -> tuple[list[float], int, int]:
        own_times = []
        own_visible = own_possible = 0
        for frame in frames:
            started = time.perf_counter()
            people = tracker.infer(frame, max_side)
            own_times.append((time.perf_counter() - started) * 1000)
            if people:
                kpts = np.stack([p.kpts for p in people])
                a, b = wrist_counts(kpts)
                own_visible += a
                own_possible += b
        return own_times, own_visible, own_possible

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cameras) as pool:
        for times, a, b in pool.map(worker, trackers):
            durations.extend(times)
            visible += a
            possible += b
    return time.perf_counter() - started, durations, visible, possible


def run_predict(frames: list[np.ndarray], model_path: str, device: str, max_side: int,
                cameras: int, mode: str) -> tuple[float, list[float], int, int]:
    from ultralytics import YOLO
    import torch

    resolved = "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    precision = fp16_arguments() if resolved.startswith("cuda") else {}
    model = YOLO(model_path)
    model.predict(np.zeros((max_side, max_side, 3), np.uint8), device=resolved, imgsz=max_side,
                  verbose=False, **precision)
    durations: list[float] = []
    visible = possible = 0
    lock = threading.Lock()

    def predict_one(frame: np.ndarray) -> tuple[float, int, int]:
        started = time.perf_counter()
        with lock:
            result = model.predict(frame, device=resolved, imgsz=max_side, conf=0.40,
                                   classes=[0], verbose=False, **precision)[0]
        elapsed = (time.perf_counter() - started) * 1000
        kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0, 17, 3))
        a, b = wrist_counts(kpts)
        return elapsed, a, b

    started = time.perf_counter()
    if mode == "predict-lock":
        with ThreadPoolExecutor(max_workers=cameras) as pool:
            for elapsed, a, b in pool.map(predict_one, [f for _ in range(cameras) for f in frames]):
                durations.append(elapsed)
                visible += a
                possible += b
    else:
        for frame in frames:
            tick = time.perf_counter()
            results = model.predict([frame] * cameras, device=resolved, imgsz=max_side, conf=0.40,
                                    classes=[0], verbose=False, **precision)
            durations.extend([(time.perf_counter() - tick) * 1000 / cameras] * cameras)
            for result in results:
                kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0, 17, 3))
                a, b = wrist_counts(kpts)
                visible += a
                possible += b
    return time.perf_counter() - started, durations, visible, possible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "video/video2.mp4")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument("--cameras", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--mode", choices=("trackers", "predict-lock", "predict-batch"), default="trackers")
    args = parser.parse_args()
    frames = frames_from_video(args.video, args.fps, args.frames, args.imgsz)
    if not frames:
        raise SystemExit("No se leyeron cuadros")
    model_path = ensure_model(ROOT / "models", args.model)
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.mode == "trackers":
        seconds, times, visible, possible = run_trackers(frames, model_path, args.device, args.imgsz, args.cameras)
    else:
        seconds, times, visible, possible = run_predict(frames, model_path, args.device, args.imgsz,
                                                        args.cameras, args.mode)
    count = len(frames) * args.cameras
    ordered = sorted(times)
    print(f"modo={args.mode} dispositivo={args.device} modelo={args.model} imgsz={args.imgsz} "
          f"camaras={args.cameras} cuadros_por_camara={len(frames)}")
    print(f"tiempo_s={seconds:.2f} fps_total={count / seconds:.2f} "
          f"fps_por_camara={len(frames) / seconds:.2f} ms_por_cuadro={statistics.mean(times):.1f} "
          f"p95_ms={ordered[int(0.95 * (len(ordered) - 1))]:.1f} "
          f"munecas_visibles={visible}/{possible}")
    if torch.cuda.is_available():
        print(f"vram_asignada_mib={torch.cuda.memory_allocated() / 1048576:.0f} "
              f"vram_pico_mib={torch.cuda.max_memory_allocated() / 1048576:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
