"""Overlay del video: zonas de la tienda, cuadro por persona con su ID y nivel de sospecha,
estado de cada mano y la secuencia de sustraccion (alcance -> ocultamiento -> caja -> salida).

Solo ASCII: las fuentes de OpenCV no dibujan acentos.
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from .heuristics import TrackState
from .vision import KP_CONF, L_WRIST, R_WRIST, SKELETON
from .zones import ZoneSet

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {"ok": (80, 200, 80), "observacion": (0, 215, 255), "posible_ocultamiento": (0, 140, 255),
          "probable_sustraccion": (60, 60, 255)}
LEVEL_TEXT = {"ok": "normal", "observacion": "observacion", "posible_ocultamiento": "OCULTAMIENTO",
              "probable_sustraccion": "SUSTRACCION"}
ZONE_COLORS = {"estanteria": (255, 160, 60), "caja": (80, 200, 80), "salida": (60, 60, 255),
               "personal": (160, 160, 160)}
ZONE_TEXT = {"estanteria": "ESTANTE", "caja": "CAJA", "salida": "SALIDA", "personal": "PERSONAL"}
HAND_COLORS = {"alcance": (0, 165, 255), "vigilada": (0, 230, 255)}      # naranja: en el estante, amarillo: tomo algo
HAND_TEXT = {"alcance": "ALCANCE", "vigilada": "TOMO ALGO?"}
STAGES = (("alcance", "ALCANCE"), ("ocultamiento", "OCULTA"), ("caja", "CAJA"), ("salida", "SALIDA"))
DIM = (110, 110, 110)
PLACE_TEXT = {"bolsillo": "bolsillo", "bolso": "bolso", "pecho": "pecho", "manos_a_la_cintura": "cintura"}


def _label(img: np.ndarray, text: str, x: int, y: int, fg, bg, scale: float = 0.45) -> int:
    """Texto sobre fondo solido; devuelve el x donde termina."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 8, y + 3), bg, -1)
    cv2.putText(img, text, (x + 4, y), FONT, scale, fg, 1, cv2.LINE_AA)
    return x + tw + 10


def _draw_zones(out: np.ndarray, zones: ZoneSet) -> None:
    shade = out.copy()
    for kind, color in ZONE_COLORS.items():
        for poly in zones.polygons(kind):
            cv2.fillPoly(shade, [poly], color)
    cv2.addWeighted(shade, 0.12, out, 0.88, 0, out)
    for kind, color in ZONE_COLORS.items():
        for poly in zones.polygons(kind):
            cv2.polylines(out, [poly], True, color, 1, cv2.LINE_AA)
            cv2.putText(out, ZONE_TEXT[kind], (int(poly[0][0]) + 3, int(poly[0][1]) + 14), FONT, 0.4, color, 1,
                        cv2.LINE_AA)


def _draw_hands(out: np.ndarray, ts: TrackState, now: float) -> None:
    for wrist, state in zip((L_WRIST, R_WRIST), ts.hand_states(now)):
        x, y, conf = ts.kpts[wrist]
        if conf < KP_CONF or not state:
            continue
        color = HAND_COLORS[state]
        cv2.circle(out, (int(x), int(y)), 11, color, 2, cv2.LINE_AA)
        _label(out, HAND_TEXT[state], int(x) + 10, int(y) - 8, (0, 0, 0), color, 0.38)


def _draw_stage(out: np.ndarray, ts: TrackState, now: float, x: int, y: int) -> None:
    """Secuencia bajo el cuadro: las etapas ya cumplidas brillan, las pendientes se apagan."""
    done = ts.stage(now)
    color = COLORS.get(ts.level, COLORS["ok"])
    for key, text in STAGES:
        if key == "ocultamiento" and done[key] and ts.conceal_place:
            text = "OCULTA: " + PLACE_TEXT.get(ts.conceal_place, ts.conceal_place)
        x = _label(out, text, x, y, (0, 0, 0) if done[key] else DIM, color if done[key] else (45, 45, 45), 0.38)


def draw_tracks(frame: np.ndarray, tracks: Iterable[TrackState], camera_name: str, fps: float,
                zones: Optional[ZoneSet] = None, dark: bool = False, now: Optional[float] = None) -> np.ndarray:
    out = frame.copy()
    tracks = list(tracks)
    if zones is not None:
        _draw_zones(out, zones)
    for ts in tracks:
        moment = ts.last_seen if now is None else now
        color = COLORS.get(ts.level, COLORS["ok"])
        x1, y1, x2, y2 = (int(v) for v in ts.bbox)
        for a, b in SKELETON:                      # esqueleto discreto: lo importante son las manos
            if ts.kpts[a, 2] >= KP_CONF and ts.kpts[b, 2] >= KP_CONF:
                cv2.line(out, (int(ts.kpts[a, 0]), int(ts.kpts[a, 1])), (int(ts.kpts[b, 0]), int(ts.kpts[b, 1])),
                         (200, 200, 120), 1, cv2.LINE_AA)
        for wrist in (L_WRIST, R_WRIST):
            x, y, conf = ts.kpts[wrist]
            if conf >= KP_CONF:
                cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
        for hand_index, point in enumerate(ts.flow_wrist):
            if point is not None and now is not None and now - ts.flow_wrist_time <= 0.5:
                px, py = map(int, point)
                cv2.circle(out, (px, py), 7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(out, "flujo mano", (px + 7, max(12, py - 5)), FONT, 0.32,
                            (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if ts.is_staff:
            label = f"ID {ts.track_id} personal (excluido)"
        else:
            label = f"ID {ts.track_id} {LEVEL_TEXT.get(ts.level, ts.level)} {ts.score:.2f}"
            if ts.verdict:
                label += f" | Gemini {ts.verdict['state']} {round(ts.verdict['confidence'] * 100)}%"
        _label(out, label, x1, max(14, y1 - 4), (0, 0, 0), color, 0.5)
        if not ts.is_staff:
            _draw_hands(out, ts, moment)
            _draw_stage(out, ts, moment, x1, min(out.shape[0] - 6, y2 + 16))
    people = sum(1 for t in tracks if not t.is_staff)
    header = f"{camera_name} | {fps:.1f} fps | {people} cliente(s)" + (" | modo nocturno" if dark else "")
    cv2.putText(out, header, (8, 20), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, header, (8, 20), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def placeholder(text: str, size=(640, 360)) -> np.ndarray:
    img = np.full((size[1], size[0], 3), 30, np.uint8)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    cv2.putText(img, text, (max(8, (size[0] - tw) // 2), size[1] // 2), FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return img
