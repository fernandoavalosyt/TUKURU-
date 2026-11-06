"""Muestrea cuadros localmente para preparar un detector cenital de cabezas.

No sube imagenes. Para camaras usa el endpoint local de cuadro sin overlays de Tukuru.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "head_dataset"
SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
CAMERA_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def camera_frame(camera_id: str):
    status_url = "http://127.0.0.1:5000/api/status"
    with urllib.request.urlopen(status_url, timeout=4) as response:
        status = json.load(response)
    camera = next((item for item in status.get("cameras", [])
                   if item.get("camera", {}).get("id") == camera_id), None)
    if camera is None:
        raise RuntimeError(f"No existe {camera_id} en la instancia local de Tukuru")
    if camera.get("stream", {}).get("state") != "connected":
        raise RuntimeError(f"{camera_id} no esta conectada en Tukuru")

    url = "http://127.0.0.1:5000/api/cameras/{}/zone-frame".format(camera_id)
    with urllib.request.urlopen(url, timeout=6) as response:
        payload = json.load(response)
    preview = payload.get("preview", "")
    prefix = "data:image/jpeg;base64,"
    if not preview.startswith(prefix):
        raise RuntimeError("Tukuru no devolvio un cuadro JPEG local valido")
    raw = base64.b64decode(preview[len(prefix):], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("No se pudo decodificar el cuadro local")
    return image


def save_frame(image, session_dir: Path, index: int, seconds: float) -> dict:
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"frame_{index:06d}.jpg"
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("No se pudo codificar un cuadro JPEG")
    data = encoded.tobytes()
    (images_dir / filename).write_bytes(data)
    return {"filename": filename, "seconds": round(seconds, 3),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "sha256": hashlib.sha256(data).hexdigest()}


def sample_video(path: Path, session_dir: Path, interval: float, duration: float, limit: int) -> list[dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("No se pudo abrir el video local")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0
    frames, next_sample, index = [], 0.0, 0
    try:
        while len(frames) < limit:
            ok, image = capture.read()
            if not ok:
                break
            seconds = index / fps
            if seconds > duration:
                break
            if seconds + 1e-6 >= next_sample:
                frames.append(save_frame(image, session_dir, len(frames), seconds))
                next_sample += interval
            index += 1
    finally:
        capture.release()
    return frames


def sample_camera(camera_id: str, session_dir: Path, interval: float, duration: float, limit: int) -> list[dict]:
    frames, started, next_sample = [], time.monotonic(), 0.0
    while len(frames) < limit:
        elapsed = time.monotonic() - started
        if elapsed > duration:
            break
        if elapsed + 1e-6 >= next_sample:
            image = camera_frame(camera_id)
            frames.append(save_frame(image, session_dir, len(frames), elapsed))
            next_sample += interval
        else:
            time.sleep(min(0.1, next_sample - elapsed))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Muestrea cuadros locales para etiquetar cabezas cenitales")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", help="ID de camara conectada en Tukuru local, por ejemplo CAM-01")
    source.add_argument("--video", type=Path, help="Video local grabado desde la camara cenital")
    parser.add_argument("--session", required=True, help="ID de esta grabacion/sesion independiente")
    parser.add_argument("--interval", type=float, default=1.0, help="Segundos entre cuadros (default: 1)")
    parser.add_argument("--duration", type=float, default=120.0, help="Duracion maxima en segundos (default: 120)")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximo de cuadros (default: 120)")
    args = parser.parse_args()
    if not SESSION_RE.fullmatch(args.session):
        parser.error("--session acepta letras, numeros, punto, guion y guion bajo (1-80 caracteres)")
    if args.camera and not CAMERA_RE.fullmatch(args.camera):
        parser.error("ID de camara invalido")
    if args.interval < 0.2 or args.interval > 60:
        parser.error("--interval debe estar entre 0.2 y 60 segundos")
    if args.duration <= 0 or args.duration > 3600 or not 1 <= args.max_frames <= 2000:
        parser.error("--duration debe ser 0-3600 y --max-frames de 1 a 2000")
    if args.video and not args.video.is_file():
        parser.error("el video local no existe")

    session_dir = OUT_ROOT / args.session
    if session_dir.exists() and any(session_dir.iterdir()):
        parser.error("esa sesion ya existe; usa otro --session para no mezclar grabaciones")
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        frames = (sample_camera(args.camera, session_dir, args.interval, args.duration, args.max_frames)
                  if args.camera else
                  sample_video(args.video, session_dir, args.interval, args.duration, args.max_frames))
    except Exception as exc:  # noqa: BLE001 - error corto sin imprimir URL ni datos del stream
        parser.error(str(exc))
    if not frames:
        parser.error("no se capturaron cuadros; confirma que la camara o el video tengan imagen")
    manifest = {"session_id": args.session, "source_kind": "camera" if args.camera else "video",
                "camera_id": args.camera, "interval_seconds": args.interval, "frames": frames}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cuadros guardados: {len(frames)}")
    print(f"Carpeta local: {session_dir.relative_to(ROOT)}")
    print("No se guardo la ruta del video ni se envio contenido a servicios externos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
