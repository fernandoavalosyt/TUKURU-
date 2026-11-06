"""Recortes de cada persona y seleccion del clip que se envia a Gemini.

Lo usan el pipeline en vivo y el replay, para que ambos manden exactamente lo mismo.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .heuristics import TrackState

CROP_INTERVAL = 0.5           # s entre recortes guardados
Clip = List[Tuple[float, bytes]]


def store_crop(ts: TrackState, frame: np.ndarray, scale: float, now: float, side: int,
               enhance: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
    """Guarda un recorte pequeno (persona completa con brazos, resolucion original)."""
    if now - ts.last_crop_time < CROP_INTERVAL:
        return
    ts.last_crop_time = now
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (ts.bbox / scale)
    pad_x, pad_y = (x2 - x1) * 0.30, (y2 - y1) * 0.12
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(fw, int(x2 + pad_x)), min(fh, int(y2 + pad_y))
    if right - left < 24 or bottom - top < 24:
        return
    crop = frame[top:bottom, left:right]
    if enhance is not None:
        crop = enhance(crop)
    longest = max(crop.shape[:2])
    if longest != side:    # se amplian los recortes chicos: una persona lejana no debe llegar como 200 px a Gemini
        k = side / longest
        crop = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)),
                          interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    ok, jpeg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if ok:
        ts.crops.append((now, jpeg.tobytes()))


def store_context_frame(ts: TrackState, frame: np.ndarray, scale: float, now: float) -> None:
    """Guarda la escena completa con el ID de la persona, para revisar estante y entorno."""
    if now - ts.last_context_crop_time < CROP_INTERVAL:
        return
    ts.last_context_crop_time = now
    image = frame.copy()
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in ts.bbox / scale)
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
    label = f"ID {ts.track_id}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    ly = max(th + 8, y1)
    cv2.rectangle(image, (x1, ly - th - 8), (min(width - 1, x1 + tw + 10), ly + baseline), (0, 220, 255), -1)
    cv2.putText(image, label, (x1 + 5, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    longest = max(height, width)
    if longest > 960:
        ratio = 960 / longest
        image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if ok:
        ts.context_crops.append((now, jpeg.tobytes()))


def select_clip(ts: TrackState, now: float, event: float, pre_seconds: float, frames: int) -> Optional[Clip]:
    """`frames` recortes desde `pre_seconds` s antes del ocultamiento hasta `now`, repartidos de forma pareja.

    El clip se ancla al ocultamiento (no a los ultimos N segundos) para que el momento en que el producto
    desaparece quede bien cubierto. Devuelve [(segundos_atras, jpeg)]."""
    recent = [(t, j) for t, j in ts.crops if event - pre_seconds <= t <= now]
    if len(recent) < 2:
        return None
    if len(recent) > frames:
        recent = [recent[round(i * (len(recent) - 1) / (frames - 1))] for i in range(frames)] if frames > 1 \
            else recent[-1:]
    return [(now - t, j) for t, j in recent]


def select_context_clip(ts: TrackState, now: float, event: float, pre_seconds: float,
                        frames: int) -> Optional[Clip]:
    """Selecciona cuadros de escena completa para Telegram, anclados al alcance del estante."""
    recent = [(t, jpeg) for t, jpeg in ts.context_crops if event - pre_seconds <= t <= now]
    if len(recent) < 2:
        return None
    if len(recent) > frames:
        recent = [recent[round(i * (len(recent) - 1) / (frames - 1))] for i in range(frames)] \
            if frames > 1 else recent[-1:]
    return [(now - t, jpeg) for t, jpeg in recent]
