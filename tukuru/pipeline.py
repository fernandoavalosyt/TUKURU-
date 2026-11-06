"""Pipeline de una camara: captura -> optimizador -> YOLO-pose+tracking ->
heuristica de robo (zonas) -> (Gemini revisa el clip si hay ocultamiento) -> decision -> alerta -> video."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .alerts import AlertManager
from .camera import CameraStream
from .config import CameraConfig, Settings
from .clips import select_clip, store_context_frame, store_crop
from .decision import apply_verdict, decide
from .gemini import GeminiJudge
from .heuristics import TrackState, update_track
from .motion import WristMotionTracker
from .optimizer import FpsGate, FrameOptimizer
from .render import draw_tracks, placeholder
from .vision import Person, PoseTracker
from .zones import ZoneSet
from .tracker_config import bytetrack_config

logger = logging.getLogger("tukuru.pipeline")

TRACK_TTL = 10.0              # s sin ver a una persona para olvidar su ID
DISPLAY_TRACK_MAX_AGE = 1.0   # s: oculta del panel tracks retenidos pero no detectados recientemente
MIN_PERSON_HEIGHT = 0.10      # fraccion del alto del frame: ignora personas muy lejanas
REVIEW_GIVE_UP = 45.0         # s esperando cupo de Gemini antes de quedarse con el respaldo local


class CameraPipeline(threading.Thread):
    def __init__(self, cfg: CameraConfig, settings: Settings, model_path: str, judge: GeminiJudge,
                 alerts: AlertManager) -> None:
        super().__init__(name=f"pipeline-{cfg.id}", daemon=True)
        self.cfg, self.s, self.judge, self.alerts = cfg, settings, judge, alerts
        self.stream = CameraStream(cfg)
        self.optimizer = FrameOptimizer(settings.max_side)
        self.model_path = model_path
        self.tracker: Optional[PoseTracker] = None     # se carga en run(): tarda y no debe frenar la web
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._tracks: Dict[int, TrackState] = {}
        self._wrist_motion = WristMotionTracker()
        self._jpeg: Optional[bytes] = None
        self._last_seq = -1
        self._fps_out = 0.0
        self._last_infer = 0.0
        self._zones: Optional[ZoneSet] = None
        self._zones_size = (0, 0)
        self._had_people = False
        self._offline_flagged = False

    # ---- API para la web ---------------------------------------------------
    def jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            people = [{
                "id": ts.track_id, "level": ts.level, "source": ts.level_source, "staff": ts.is_staff,
                "reason": ts.level_reason, "seconds_visible": round(now - ts.first_seen),
                "verdict": ts.verdict, "etapa": ts.stage(now), "manos": ts.hand_states(now), **ts.summary(now),
            } for ts in sorted(self._tracks.values(), key=lambda t: t.track_id)
                if now - ts.last_seen <= DISPLAY_TRACK_MAX_AGE]
        return {"camera": self.cfg.public(), "stream": self.stream.status(), "fps": round(self._fps_out, 1),
                "dark_mode": self.optimizer.is_dark, "people": people}

    def stop(self) -> None:
        self._stop_event.set()
        self.stream.stop()

    # ---- bucle -----------------------------------------------------------
    def run(self) -> None:
        self.stream.start()
        try:
            person_conf = self.cfg.yolo_conf if self.cfg.yolo_conf is not None else self.s.yolo_conf
            self.tracker = PoseTracker(self.model_path, self.s.device, person_conf,
                                       self.s.max_side,
                                       bytetrack_config(self.s.data_dir, self.s.track_buffer_frames))
        except Exception:  # noqa: BLE001
            logger.exception("[%s] no se pudo cargar el modelo", self.cfg.id)
            self.stream.stop()
            return
        gate = FpsGate(self.s.inference_fps)
        while not self._stop_event.is_set():
            try:
                self._tick(gate)
            except Exception:  # noqa: BLE001 - un frame malo no debe tumbar la camara
                logger.exception("[%s] error procesando frame", self.cfg.id)
                self._stop_event.wait(0.5)
        self.stream.join(timeout=3)

    def _tick(self, gate: FpsGate) -> None:
        frame, seq, frame_time = self.stream.latest()
        now = time.time()
        self._check_offline()
        if frame is None:
            with self._lock:
                self._tracks.clear()
                self._wrist_motion.reset()
            self._publish(placeholder(f"{self.cfg.name}: SIN SENAL"))
            self._stop_event.wait(0.5)
            return
        # Con la zona vacia se analiza mas despacio: ahorra CPU.
        gate.set_fps(self.s.inference_fps if self._had_people else self.s.idle_fps)
        if seq == self._last_seq or not gate.ready(now):
            self._stop_event.wait(0.02)
            return
        self._last_seq = seq

        small, scale = self.optimizer.process(frame)
        h, w = small.shape[:2]
        zones = self._zones_for(w, h)
        people = [p for p in self.tracker.infer(small, self.s.max_side) if self._accept(p, w, h)]
        self._had_people = bool(people)
        started = time.time()
        self._fps_out = 0.9 * self._fps_out + 0.1 / max(1e-3, started - self._last_infer) if self._last_infer \
            else self._fps_out
        self._last_infer = started

        with self._lock:
            seen = set()
            new_events: Dict[int, list] = {}
            for person in people:
                ts = self._tracks.get(person.track_id)
                if ts is None:
                    ts = self._tracks[person.track_id] = TrackState(person.track_id, self.cfg.id, now, now)
                seen.add(person.track_id)
            self._wrist_motion.update(small, people, self._tracks, zones, now)
            for person in people:
                ts = self._tracks[person.track_id]
                previous_events = {(event_time, kind) for event_time, kind, _ in ts.events}
                update_track(ts, person, zones, now, self.s)
                new_events[person.track_id] = [event for event in ts.events
                                               if (event[0], event[1]) not in previous_events]
                self._store_crop(ts, frame, scale, now)
                store_context_frame(ts, frame, scale, now)
            for track_id in [t for t, ts in self._tracks.items() if now - ts.last_seen > TRACK_TTL]:
                del self._tracks[track_id]
            tracks = list(self._tracks.values())
            visible_tracks = [ts for ts in tracks if now - ts.last_seen <= DISPLAY_TRACK_MAX_AGE]

            for ts in tracks:
                if ts.track_id not in seen:
                    continue
                self._maybe_review(ts, now)
                decision = decide(ts, now, self.s)
                # Solo capturamos interacciones que siguen por debajo de una alerta.
                # Esta muestra queda local para revision y no entra en AlertManager._alerts.
                if decision[0] in ("ok", "observacion"):
                    for _, event_kind, _ in new_events.get(ts.track_id, []):
                        self.alerts.observe_event(ts, event_kind, now, self.cfg.name,
                                                  recording_id=self.cfg.recording_group_id or None)
                # La evidencia se dibuja solo si realmente se emite una alerta.
                self.alerts.evaluate(ts, decision, now, self.cfg.name,
                                     lambda: draw_tracks(small, visible_tracks, self.cfg.name, self._fps_out, zones,
                                                         now=now),
                                     recording_id=self.cfg.recording_group_id or None)

        self._publish(draw_tracks(small, visible_tracks, self.cfg.name, self._fps_out, zones,
                                  self.optimizer.is_dark, now=now))

    # ---- piezas -------------------------------------------------------------
    def _zones_for(self, w: int, h: int) -> ZoneSet:
        if self._zones is None or self._zones_size != (w, h):
            self._zones, self._zones_size = ZoneSet(self.cfg.zones, w, h), (w, h)
        return self._zones

    def _accept(self, person: Person, w: int, h: int) -> bool:
        min_height = self.cfg.min_person_height if self.cfg.min_person_height is not None else MIN_PERSON_HEIGHT
        return (person.bbox[3] - person.bbox[1]) >= min_height * h

    def _store_crop(self, ts: TrackState, frame: np.ndarray, scale: float, now: float) -> None:
        store_crop(ts, frame, scale, now, self.s.gemini_crop_side,
                   self.optimizer.enhance if self.optimizer.is_dark else None)

    def _maybe_review(self, ts: TrackState, now: float) -> None:
        """Pide a Gemini el dictamen del ultimo ocultamiento, un instante despues de que ocurrio."""
        event = ts.review_event
        if event is None or ts.review_pending or now < ts.review_due:
            return
        if self.judge.disabled_reason is not None:       # sin Gemini: decide el respaldo local
            ts.review_event = None
            return
        frames = select_clip(ts, now, event, self.s.clip_seconds, self.s.gemini_frames)
        if frames:
            context = {**ts.summary(now), "visible_seg": round(now - ts.first_seen)}
            ts.review_pending = True
            if self.judge.review(f"{self.cfg.id}-{ts.track_id}", frames, context,
                                 lambda k, v, e, track=ts, ev=event: self._on_verdict(track, ev, v, e)):
                ts.review_event = None
                return
            ts.review_pending = False
        if now - event > REVIEW_GIVE_UP:
            ts.review_event = None

    def _on_verdict(self, ts: TrackState, event: float, verdict: Optional[Dict[str, Any]],
                    error: Optional[str]) -> None:
        with self._lock:
            ts.review_pending = False
            if verdict is not None:
                apply_verdict(ts, event, verdict, time.time(), self.s)

    def _check_offline(self) -> None:
        offline = self.stream.offline_seconds()
        if offline >= self.s.camera_offline_seconds:
            self.alerts.system_alert(self.cfg.id, self.cfg.name, "offline",
                                     f"Camara sin senal desde hace {offline:.0f} s: no hay vigilancia en esta zona")
            self._offline_flagged = True
        elif self._offline_flagged and offline == 0.0:
            self.alerts.system_alert(self.cfg.id, self.cfg.name, "offline", None)
            self._offline_flagged = False

    def _publish(self, image: np.ndarray) -> None:
        ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self._lock:
                self._jpeg = jpeg.tobytes()
