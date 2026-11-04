"""Dibuja el panel de Tukuru sobre un lienzo (video + personas + alertas) para
mostrar una repeticion como si se viera en el dashboard en tiempo real.

Replica la informacion de static/index.html: video anotado, pildoras de estado,
tabla de personas detectadas y lista de alertas con su evidencia.
Solo ASCII: las fuentes de OpenCV no dibujan acentos.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .heuristics import TrackState

BG, CARD, LINE = (25, 20, 15), (41, 33, 26), (64, 52, 42)
TXT, MUTED = (243, 237, 230), (165, 152, 139)
OK, WARN, BAD, ACC = (80, 185, 63), (32, 160, 240), (73, 81, 248), (248, 147, 68)
LEVEL_COLOR = {"ok": OK, "observacion": MUTED, "posible_ocultamiento": WARN, "probable_sustraccion": BAD}
LEVEL_TEXT = {"ok": "normal", "observacion": "observacion", "posible_ocultamiento": "POSIBLE OCULTAMIENTO",
              "probable_sustraccion": "PROBABLE SUSTRACCION", "sistema": "SISTEMA"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
THUMB = (96, 64)


def _text(img: np.ndarray, text: str, org: Tuple[int, int], color=TXT, scale: float = 0.45, thick: int = 1) -> int:
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)
    return org[0] + cv2.getTextSize(text, FONT, scale, thick)[0][0]


def _pill(img: np.ndarray, x: int, y: int, text: str, color) -> int:
    width = cv2.getTextSize(text, FONT, 0.45, 1)[0][0] + 20
    cv2.rectangle(img, (x, y), (x + width, y + 24), CARD, -1)
    cv2.rectangle(img, (x, y), (x + width, y + 24), LINE, 1)
    _text(img, text, (x + 10, y + 17), color)
    return x + width + 8


def _wrap(text: str, width: int, scale: float = 0.42) -> List[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if cv2.getTextSize(candidate, FONT, scale, 1)[0][0] > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else lines


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _card(img: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), CARD, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), LINE, 1)
    _text(img, title, (x + 12, y + 22), TXT, 0.55, 1)


def _people_table(img: np.ndarray, x: int, y: int, w: int, tracks: Iterable[TrackState], now: float) -> None:
    cols = [("ID", 0), ("Estado", 38), ("Punt.", 238), ("Alcan.", 286), ("Ocult.", 340), ("Lugar", 394),
            ("Mirada", 514), ("Caja", 562), ("Gemini", 606)]
    for name, dx in cols:
        _text(img, name, (x + dx, y), MUTED, 0.4)
    cv2.line(img, (x, y + 6), (x + w, y + 6), LINE, 1)
    rows = sorted((t for t in tracks if now - t.last_seen <= 10.0), key=lambda t: t.track_id)
    if not rows:
        _text(img, "Sin personas en escena.", (x, y + 28), MUTED)
        return
    for i, ts in enumerate(rows[:4]):
        ry = y + 26 + i * 22
        s = ts.summary(now)
        state = "personal" if ts.is_staff else LEVEL_TEXT.get(ts.level, ts.level)
        color = MUTED if ts.is_staff else LEVEL_COLOR.get(ts.level, TXT)
        verdict = ts.verdict
        gem = f"{verdict['state']} {round(verdict['confidence'] * 100)}%" if verdict else "-"
        cells = [str(ts.track_id), state, f"{ts.score:.2f}", str(s["alcances_2min"]), str(s["ocultamientos"]),
                 (s["lugar_ocultamiento"] or "-").replace("_", " "), str(s["giros_de_cabeza"]),
                 "si" if s["paso_por_caja"] else "no", gem]
        for (_, dx), cell in zip(cols, cells):
            _text(img, cell, (x + dx, ry), color if dx in (38, 238) else TXT, 0.42,
                  2 if dx == 38 and ts.level in ("posible_ocultamiento", "probable_sustraccion") else 1)


def _alert_cards(img: np.ndarray, x: int, y: int, w: int, alerts: List[Dict[str, Any]], thumbs: Dict[str, np.ndarray],
                 camera: str, now: float, max_cards: int = 4) -> None:
    if not alerts:
        _text(img, "Sin alertas.", (x, y + 14), MUTED)
        return
    for i, alert in enumerate(alerts[:max_cards]):
        top = y + i * 112
        color = LEVEL_COLOR.get(alert["level"], ACC)
        cv2.rectangle(img, (x, top), (x + w, top + 104), BG, -1)
        cv2.rectangle(img, (x, top), (x + w, top + 104), LINE, 1)
        cv2.rectangle(img, (x, top), (x + 4, top + 104), color, -1)
        thumb = thumbs.get(alert["id"])
        if thumb is not None:
            img[top + 8:top + 8 + THUMB[1], x + 12:x + 12 + THUMB[0]] = thumb
        tx = x + 12 + THUMB[0] + 10
        who = f" - persona ID {alert['track_id']}" if alert["track_id"] is not None else ""
        _text(img, LEVEL_TEXT.get(alert["level"], alert["level"].upper()), (tx, top + 20), color, 0.48, 2)
        _text(img, f"{camera}{who}", (tx, top + 38), TXT, 0.42)
        for j, line in enumerate(_wrap(alert["reason"], w - (tx - x) - 10)[:3]):
            _text(img, line, (tx, top + 56 + j * 15), TXT, 0.4)
        _text(img, f"{_clock(alert['time'])} - fuente: {alert['source'] or '-'}", (tx, top + 98), MUTED, 0.38)
        age = now - alert["time"]
        if age < 6:
            _text(img, "NUEVA", (x + w - 62, top + 20), BAD, 0.45, 2)


def render_dashboard(video: np.ndarray, tracks: Iterable[TrackState], alerts: List[Dict[str, Any]],
                     thumbs: Dict[str, np.ndarray], now: float, camera: str, fps: float, gemini_text: str,
                     size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
    tracks = list(tracks)
    canvas = np.full((size[1], size[0], 3), BG, np.uint8)
    unacked = len(alerts)

    # encabezado
    _text(canvas, "Tukuru", (16, 30), TXT, 0.8, 2)
    x = _pill(canvas, 110, 10, gemini_text, WARN)
    x = _pill(canvas, x, 10, f"Alertas sin reconocer: {unacked}", BAD if unacked else OK)
    _pill(canvas, x, 10, f"video {_clock(now)}", MUTED)
    cv2.line(canvas, (0, 44), (size[0], 44), LINE, 1)

    # video (borde rojo si hay una alerta reciente)
    vw, vh = 768, 432
    _card(canvas, 8, 54, vw + 16, vh + 60, f"{camera} - conectada - {fps:.1f} fps analisis")
    shown = cv2.resize(video, (vw, vh), interpolation=cv2.INTER_CUBIC)
    recent = [a for a in alerts if 0 <= now - a["time"] < 2.0]
    if recent:
        flash = BAD if any(a["level"] == "probable_sustraccion" for a in recent) else WARN
        cv2.rectangle(shown, (0, 0), (vw - 1, vh - 1), flash, 6)
        cv2.rectangle(shown, (0, vh - 34), (vw, vh), flash, -1)
        _text(shown, "ALERTA: " + LEVEL_TEXT.get(recent[0]["level"], ""), (12, vh - 11), (255, 255, 255), 0.65, 2)
    canvas[84:84 + vh, 16:16 + vw] = shown

    # personas
    _card(canvas, 8, 54 + vh + 70, vw + 16, 720 - (54 + vh + 70) - 8, "Personas detectadas")
    _people_table(canvas, 20, 54 + vh + 106, vw - 8, tracks, now)

    # alertas
    ax = 8 + vw + 16 + 8
    aw = size[0] - ax - 8
    _card(canvas, ax, 54, aw, 720 - 54 - 8, "Alertas")
    _alert_cards(canvas, ax + 8, 84, aw - 16, alerts[::-1], thumbs, camera, now)
    return canvas
