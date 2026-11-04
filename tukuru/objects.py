"""Experimento local mano-objeto-destino. Solo observacion; no modifica TrackState ni alertas.

Coordenadas publicas en el cuadro reducido; inferencia sobre recortes del original.
COCO no cubre todos los productos: ausencia de deteccion significa desconocido.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .heuristics import TrackState, analyze_body, concealment_place
from .vision import KP_CONF, L_WRIST, R_WRIST

CONTAINERS = {"handbag", "backpack", "suitcase", "basket", "shopping basket"}
IGNORED = {"person", "chair", "bench", "dining table", "refrigerator", "tv", "couch", "bed"}


@dataclass
class Detection:
    label: str
    confidence: float
    box: np.ndarray


def distance(point, box) -> float:
    """Distancia al rectangulo (cero dentro), no a su centro."""
    return float(np.linalg.norm(np.maximum(np.maximum(box[:2] - point, point - box[2:]), 0)))


class SmallDetector:
    def __init__(self, path: Path, device: str) -> None:
        if not path.is_file():
            raise ValueError(f"Falta el detector local {path}. No se descarga automaticamente.")
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        if self.model.task != "detect":
            raise ValueError("--object-model requiere pesos de deteccion, no de pose o segmentacion")
        self.device = device

    def detect(self, frame: np.ndarray, scale: float, ts: TrackState) -> list[Detection]:
        # Incluye brazos fuera del bbox; usa la resolucion original, sin el overlay.
        points = [ts.bbox[:2], ts.bbox[2:]]
        points.extend(p[:2] for p in ts.kpts[[L_WRIST, R_WRIST]] if p[2] >= KP_CONF)
        points = np.asarray(points) / scale
        pad = max(24.0, (ts.bbox[3] - ts.bbox[1]) / scale * 0.18)
        lo = np.maximum(0, np.floor(points.min(axis=0) - pad)).astype(int)
        hi = np.minimum(frame.shape[1::-1], np.ceil(points.max(axis=0) + pad)).astype(int)
        if np.any(hi - lo < 24):
            return []
        result = self.model.predict(frame[lo[1]:hi[1], lo[0]:hi[0]], imgsz=416, conf=0.25,
                                    device=self.device, max_det=30, verbose=False)[0]
        if result.boxes is None:
            return []
        return [Detection(str(result.names[int(cls)]), float(conf),
                          (box + np.tile(lo, 2)) * scale)
                for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())
                if str(result.names[int(cls)]) not in IGNORED]


@dataclass
class Relation:
    label: str = ""
    center: Optional[np.ndarray] = None
    state: str = "sin_evidencia"
    hits: int = 0
    last_seen: float = -100.0
    proposed: str = ""
    proposed_hits: int = 0


@dataclass
class Session:
    until: float = -1.0
    last_call: float = -100.0
    triggers: list = field(default_factory=lambda: [None, None])
    hands: list[Relation] = field(default_factory=lambda: [Relation(), Relation()])


class ObjectExperiment:
    """Ventana de 8 s por alcance confirmado, <=2 Hz/persona y <=4 Hz global.

    Una deteccion puede pertenecer a una sola mano. Asociaciones espaciales ambiguas
    se omiten. La proximidad al bolso/cuerpo NO confirma insercion ni sustraccion.
    """
    def __init__(self, detector) -> None:
        self.detector = detector
        self.sessions: dict[int, Session] = {}
        self.samples: list[dict] = []
        self.events: list[dict] = []
        self.calls = 0
        self.seconds = 0.0
        self.last_call = -100.0
        self.boxes: list[Detection] = []

    def step(self, frame, scale, tracks, zones, now: float) -> None:
        visible = [t for t in tracks if now - t.last_seen < 0.15 and not t.is_staff]
        for key in list(self.sessions):
            if not any(t.track_id == key for t in visible):
                del self.sessions[key]
        for ts in visible:
            session = self.sessions.setdefault(ts.track_id, Session())
            for i, hand in enumerate(ts.hands):
                if (hand.checked and hand.armed_until > now and hand.reach_start is not None
                        and hand.reach_start != session.triggers[i]):
                    session.triggers[i] = hand.reach_start
                    session.until = now + 8.0
                    session.hands[i] = Relation()
        self.boxes = []
        if now - self.last_call < 0.25 - 1e-6:
            return
        eligible = [t for t in visible if self.sessions[t.track_id].until >= now
                    and now - self.sessions[t.track_id].last_call >= 0.5 - 1e-6]
        if not eligible:
            return
        ts = min(eligible, key=lambda t: self.sessions[t.track_id].last_call)
        session = self.sessions[ts.track_id]
        started = time.perf_counter()
        detections = self.detector.detect(frame, scale, ts)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        session.last_call = self.last_call = now
        self.boxes = detections
        self.samples.append({"t": round(now, 3), "id": ts.track_id,
                             "detections": [{"label": d.label, "confidence": round(d.confidence, 3),
                                             "box": d.box.round(2).tolist()} for d in detections]})
        self.observe(ts, session, detections, visible, zones, now)

    def observe(self, ts, session, detections, visible, zones, now: float) -> None:
        body = analyze_body(ts.kpts, ts.bbox)
        wrists = [(t.track_id, i, p[:2]) for t in visible
                  for i, p in enumerate(t.kpts[[L_WRIST, R_WRIST]]) if p[2] >= KP_CONF]
        containers = [d for d in detections if d.label in CONTAINERS]
        assigned = {}
        for d in detections:
            if d.label in CONTAINERS:
                continue
            near = sorted((distance(p, d.box), key, i) for key, i, p in wrists)
            if not near or near[0][0] > body.scale * 0.45:
                continue
            # Dos manos/personas demasiado cerca: no inventar propiedad.
            if len(near) > 1 and near[1][0] - near[0][0] < body.scale * 0.12:
                continue
            _, key, hand = near[0]
            if key == ts.track_id:
                assigned.setdefault(hand, []).append(d)
        for i, relation in enumerate(session.hands):
            options = assigned.get(i, [])
            if relation.center is not None and now - relation.last_seen <= 1.1:
                options = [d for d in options if d.label == relation.label
                           and np.linalg.norm((d.box[:2] + d.box[2:]) / 2 - relation.center) <= body.scale]
            if not options:
                if relation.label and now - relation.last_seen > 0.8:
                    relation.hits = relation.proposed_hits = 0
                    self._state(ts, i, relation, "desconocido", now)
                continue
            wrist = ts.kpts[[L_WRIST, R_WRIST][i], :2]
            d = min(options, key=lambda x: distance(wrist, x.box))
            center = (d.box[:2] + d.box[2:]) / 2
            continuing = d.label == relation.label and now - relation.last_seen <= 1.1
            relation.hits = relation.hits + 1 if continuing else 1
            if not continuing:
                relation.state, relation.proposed, relation.proposed_hits = "sin_evidencia", "", 0
            relation.label, relation.center, relation.last_seen = d.label, center, now
            if relation.hits < 2:
                continue
            if zones.contains("estanteria", center):
                state = "objeto_en_estanteria"
            elif any(distance(center, c.box) < body.scale * 0.15 for c in containers):
                state = "objeto_junto_a_contenedor"
            elif concealment_place(wrist, body):
                state = "objeto_junto_al_cuerpo"
            else:
                state = "objeto_junto_a_mano"
            relation.proposed_hits = relation.proposed_hits + 1 if relation.proposed == state else 1
            relation.proposed = state
            if relation.proposed_hits >= 2:
                self._state(ts, i, relation, state, now)

    def _state(self, ts, hand, relation, state, now):
        if relation.state != state:
            self.events.append({"t": round(now, 3), "id": ts.track_id, "hand": hand,
                                "label": relation.label, "from": relation.state, "state": state})
            relation.state = state

    def draw(self, frame):
        for d in self.boxes:
            x1, y1, x2, y2 = d.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 0), 1)
            cv2.putText(frame, f"obj: {d.label} {d.confidence:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)
        return frame

    def report(self):
        return {"mode": "observacion_sin_cambios_en_alertas", "calls": self.calls,
                "inference_seconds": round(self.seconds, 3),
                "mean_inference_ms": round(1000 * self.seconds / max(1, self.calls), 2),
                "events": self.events, "samples": self.samples}
