"""Alertas: una por escalada de nivel o por cada nuevo ocultamiento (con ALERT_REPEAT_SECONDS entre repeticiones), evidencia
(foto anotada), registro JSONL, webhook opcional y estado de reconocimiento (ack) para el panel.
En SHADOW_MODE solo se registran (sin webhook ni aviso en el panel) para calibrar sin molestar."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from .clips import select_clip, select_context_clip
from .config import Settings
from .decision import ALERT_MIN_RANK, LEVEL_RANK
from .heuristics import TrackState
from .ml.features import incident_features
from .ml.repository import IncidentRepository
from .ml.model import ShadowModel

logger = logging.getLogger("tukuru.alerts")

RESOLVE_AFTER_OK_SECONDS = 8.0


class AlertManager:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.dir = settings.data_dir / "alerts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.dir / "alerts.jsonl"
        self.session_id = uuid.uuid4().hex
        self.incidents = IncidentRepository(settings.data_dir / "incidents.sqlite3")
        self.shadow_model = None
        if settings.ml_shadow_model:
            try:
                self.shadow_model = ShadowModel.load(settings.ml_shadow_model)
                logger.info("Modelo ML en sombra: %s", self.shadow_model.version)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("No se pudo cargar modelo ML en sombra (%s); siguen las reglas", type(exc).__name__)
        self._lock = threading.Lock()
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._system_active: set[str] = set()
        self._notifier: Optional[Callable[[Dict[str, Any]], None]] = None
        self._load_recent()
        self._purge_old()

    # ---- consulta ----------------------------------------------------------
    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._serializable_record(a) for a in list(self._alerts)[-limit:][::-1]]

    @staticmethod
    def _serializable_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Los JPEG del clip son internos para Telegram; no pertenecen al JSON publico."""
        return {key: value for key, value in record.items() if key != "clip_frames"}

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for a in self._alerts if not a["ack"] and not a.get("resolved"))

    def set_notifier(self, notifier: Callable[[Dict[str, Any]], None]) -> None:
        self._notifier = notifier

    def review(self, alert_id: str, label: str, reviewer_id: int | str, source: str = "telegram") -> bool:
        """Registra la revision humana; ack/dismiss no son etiquetas de entrenamiento."""
        if label not in ("confirmado", "descartado", "no_concluyente"):
            raise ValueError("revision invalida")
        with self._lock:
            alert = next((a for a in self._alerts if a["id"] == alert_id and a.get("track_id") is not None), None)
            try:
                if self.incidents.latest_review(alert_id) == (label, source, str(reviewer_id)):
                    return True
                if not self.incidents.record_review(alert_id, label, source, str(reviewer_id)):
                    return False
            except Exception:  # noqa: BLE001 - una revision debe quedar guardada
                logger.exception("No se pudo guardar revision de %s", alert_id)
                return False
            if alert:
                alert["review"] = label
                alert["reviewer_id"] = reviewer_id
            self._append_log({"event": "review", "id": alert_id, "label": label,
                              "reviewer_id": reviewer_id, "source": source, "time": time.time()})
            return True

    def ack(self, alert_id: str) -> bool:
        with self._lock:
            for alert in self._alerts:
                if alert["id"] == alert_id:
                    alert["ack"] = True
                    self._append_log({"event": "ack", "id": alert_id, "time": time.time()})
                    return True
        return False

    def dismiss(self, alert_id: str) -> bool:
        """Quita una alerta de la vista. El registro (alerts.jsonl) y la foto se conservan como auditoria."""
        with self._lock:
            for alert in self._alerts:
                if alert["id"] == alert_id:
                    self._alerts.remove(alert)
                    self._append_log({"event": "dismiss", "id": alert_id, "time": time.time()})
                    return True
        return False

    def dismiss_all(self) -> int:
        """Limpia la vista de alertas. Devuelve cuantas se quitaron."""
        with self._lock:
            count = len(self._alerts)
            if count:
                self._alerts.clear()
                self._append_log({"event": "dismiss_all", "time": time.time()})
            return count

    def snapshot_path(self, name: str) -> Optional[Path]:
        path = (self.dir / name).resolve()
        return path if path.parent == self.dir.resolve() and path.is_file() else None

    # ---- alertas de personas ---------------------------------------------
    def observe_event(self, ts: TrackState, event_kind: str, now: float, camera_name: str,
                      recording_id: Optional[str] = None) -> bool:
        """Muestrea alcances/devoluciones locales para revision, sin crear una alerta."""
        if event_kind not in ("alcance", "devolucion") or ts.is_staff:
            return False
        incident_id = uuid.uuid4().hex[:12]
        record = {"id": incident_id, "time": now, "camera_id": ts.camera_id,
                  "camera_name": camera_name, "session_id": self.session_id, "recording_id": recording_id,
                  "track_id": ts.track_id, "level": "observacion", "source": "muestreo_local",
                  "reason": f"Muestra local: {event_kind}", "features": incident_features(ts, now),
                  "snapshot": None, "record_type": "observacion"}
        if self.shadow_model:
            try:
                record["ml_shadow_score"] = round(self.shadow_model.score(record["features"]), 4)
                record["ml_model_version"] = self.shadow_model.version
            except (ValueError, TypeError) as exc:
                logger.debug("No se pudo puntuar muestra local (%s)", type(exc).__name__)
        try:
            stored = self.incidents.record_observation(record)
        except Exception:  # noqa: BLE001 - la captura no debe detener la camara
            logger.exception("No se pudo guardar una muestra local de %s", camera_name)
            return False
        if not stored:
            return False
        # store_context_frame ya conserva localmente una imagen amplia con el ID de esta persona.
        if ts.context_crops:
            filename = f"{datetime.fromtimestamp(now):%Y%m%d_%H%M%S}_{ts.camera_id}_obs_{incident_id}.jpg"
            try:
                (self.dir / filename).write_bytes(ts.context_crops[-1][1])
                self.incidents.update_observation_snapshot(incident_id, filename)
            except OSError:
                logger.warning("No se pudo guardar la evidencia local de una muestra (%s)", incident_id)
        logger.debug("Muestra local guardada | %s | persona ID %s | %s", camera_name, ts.track_id, event_kind)
        return True

    def evaluate(self, ts: TrackState, decision: tuple, now: float, camera_name: str,
                 snapshot: Callable[[], Optional[np.ndarray]], recording_id: Optional[str] = None) -> None:
        level, source, reason = decision
        self._resolve_cleared(ts)
        ts.level, ts.level_source, ts.level_reason = level, source, reason
        rank = LEVEL_RANK[level]
        if rank < ALERT_MIN_RANK:
            # Por debajo del nivel de alerta: tras un rato tranquilo se rearma para el siguiente evento.
            if ts.ok_since is None:
                ts.ok_since = now
            if now - ts.ok_since >= RESOLVE_AFTER_OK_SECONDS and ts.alert_level_sent != "ok":
                ts.alert_level_sent, ts.alert_conceals_sent = "ok", 0
            return
        ts.ok_since = None
        conceals = len(ts.active_conceals())
        escalated = rank > LEVEL_RANK[ts.alert_level_sent]
        repeated = conceals > ts.alert_conceals_sent and now - ts.alert_sent_at >= self.s.alert_repeat_seconds
        if escalated or repeated:
            self._emit(ts, level, source, reason, now, camera_name, snapshot(), recording_id)
            ts.alert_level_sent, ts.alert_conceals_sent, ts.alert_sent_at = level, conceals, now

    def _emit(self, ts: TrackState, level: str, source: str, reason: str, now: float, camera_name: str,
              image: Optional[np.ndarray], recording_id: Optional[str] = None) -> None:
        alert_id = uuid.uuid4().hex[:8]
        photo = self._save_snapshot(alert_id, ts.camera_id, self._mark_alert_person(image, ts))
        recent_reaches = [t for t in ts.reaches if 0 <= now - t <= self.s.conceal_window + self.s.clip_seconds]
        context_anchor = max(recent_reaches, default=max(ts.active_conceals(), default=now))
        context_clip = select_context_clip(ts, now, context_anchor, 2.0, min(10, self.s.gemini_frames))
        if context_clip is None:
            context_clip = select_clip(ts, now, max(ts.active_conceals(), default=context_anchor),
                                       self.s.clip_seconds, min(10, self.s.gemini_frames))
        record = {
            "id": alert_id, "time": now, "iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "camera_id": ts.camera_id, "camera_name": camera_name, "session_id": self.session_id,
            "recording_id": recording_id,
            "track_id": ts.track_id, "level": level,
            "source": source, "reason": reason, "indicators": ts.summary(now), "verdict": ts.verdict,
            "features": incident_features(ts, now),
            "conceal_event": (conceal_ev := max(ts.active_conceals(), default=None)),
            "snapshot": photo, "ack": self.s.shadow_mode, "shadow": self.s.shadow_mode,
            "clip_frames": [jpeg for _, jpeg in (context_clip or [])],
        }
        if self.shadow_model:
            try:
                record["ml_shadow_score"] = round(self.shadow_model.score(record["features"]), 4)
                record["ml_model_version"] = self.shadow_model.version
            except (ValueError, TypeError) as exc:
                logger.warning("No se pudo puntuar incidente %s con ML (%s)", alert_id, type(exc).__name__)
        self._store(record)
        try:
            self.incidents.record_incident(record)
        except Exception:  # noqa: BLE001 - SQLite no debe detener la vigilancia
            logger.exception("No se pudo guardar incidente %s en SQLite", alert_id)
        if self._notifier and not self.s.shadow_mode:
            self._notifier(record)
        logger.warning("%sALERTA %s | %s | persona ID %s | %s (%s)", "[SOMBRA] " if self.s.shadow_mode else "",
                       level.upper(), camera_name, ts.track_id, reason, source or "-")

    @staticmethod
    def _mark_alert_person(image: Optional[np.ndarray], ts: TrackState) -> Optional[np.ndarray]:
        """Resalta en la evidencia a la persona cuyo track disparo la alerta."""
        if image is None:
            return None
        marked = image.copy()
        height, width = marked.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in ts.bbox)
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        color = (0, 220, 255)
        label = f"ID {ts.track_id}"
        scale, thickness = 0.65, 2
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        label_x = min(max(0, x1), max(0, width - text_w - 10))
        label_y = max(text_h + 8, min(y1, height - baseline - 4))
        cv2.rectangle(marked, (label_x, label_y - text_h - 8),
                      (label_x + text_w + 10, label_y + baseline), color, -1)
        cv2.putText(marked, label, (label_x + 5, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return marked

    def _resolve_cleared(self, ts: TrackState) -> None:
        """Anota secuencias anuladas sin convertirlas en etiquetas humanas."""
        if ts.cleared_until <= 0:
            return
        with self._lock:
            for alert in self._alerts:
                event = alert.get("conceal_event")
                if (alert.get("session_id") != self.session_id or alert.get("camera_id") != ts.camera_id
                        or alert.get("track_id") != ts.track_id or event is None or event > ts.cleared_until
                        or alert.get("resolved")):
                    continue
                returned = any(t >= event and kind == "devolucion" for t, kind, _ in ts.events)
                alert["resolved"] = "devolucion" if returned else "revision_visual"
                self._append_log({"event": "resolve", "id": alert["id"], "reason": alert["resolved"],
                                  "time": time.time()})

    # ---- alertas de sistema (camara sin senal) ---------------------------
    def system_alert(self, camera_id: str, camera_name: str, key: str, message: Optional[str]) -> None:
        """message=None resuelve la alerta de sistema `key`; si no, la emite una sola vez."""
        full_key = f"{camera_id}:{key}"
        if message is None:
            self._system_active.discard(full_key)
            return
        if full_key in self._system_active:
            return
        self._system_active.add(full_key)
        now = time.time()
        self._store({"id": uuid.uuid4().hex[:8], "time": now,
                     "iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"), "camera_id": camera_id,
                     "camera_name": camera_name, "track_id": None, "level": "sistema", "source": "sistema",
                     "reason": message, "indicators": {}, "verdict": None, "snapshot": None, "ack": False})
        logger.error("ALERTA SISTEMA | %s | %s", camera_name, message)

    # ---- almacenamiento ---------------------------------------------------
    def _store(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._alerts.append(record)
            self._append_log(record)
        if self.s.alert_webhook_url and not record.get("shadow"):
            threading.Thread(target=self._post_webhook, args=(record,), daemon=True).start()

    def _append_log(self, record: Dict[str, Any]) -> None:
        try:
            loggable = self._serializable_record(record)
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(loggable, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("No se pudo escribir %s", self.log_file)

    def _save_snapshot(self, alert_id: str, camera_id: str, image: Optional[np.ndarray]) -> Optional[str]:
        if image is None:
            return None
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{camera_id}_{alert_id}.jpg"
        try:
            cv2.imwrite(str(self.dir / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return name
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo guardar la evidencia")
            return None

    def _post_webhook(self, record: Dict[str, Any]) -> None:
        try:
            import requests

            requests.post(self.s.alert_webhook_url, json=self._serializable_record(record), timeout=5).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webhook fallo: %s", exc.__class__.__name__)

    def _load_recent(self) -> None:
        if not self.log_file.is_file():
            return
        try:
            for line in self.log_file.read_text(encoding="utf-8").splitlines()[-400:]:
                item = json.loads(line)
                if item.get("event") == "ack":
                    for alert in self._alerts:
                        if alert["id"] == item["id"]:
                            alert["ack"] = True
                elif item.get("event") == "review":
                    for alert in self._alerts:
                        if alert["id"] == item["id"]:
                            alert["review"] = item["label"]
                            alert["reviewer_id"] = item["reviewer_id"]
                elif item.get("event") == "resolve":
                    for alert in self._alerts:
                        if alert["id"] == item["id"]:
                            alert["resolved"] = item["reason"]
                elif item.get("event") == "dismiss":
                    self._alerts = deque((a for a in self._alerts if a["id"] != item["id"]), maxlen=200)
                elif item.get("event") == "dismiss_all":
                    self._alerts.clear()
                elif "level" in item:
                    self._alerts.append(item)
                    if item.get("track_id") is not None:
                        try:
                            self.incidents.record_incident(item)
                        except Exception:  # noqa: BLE001 - historial antiguo no debe impedir el arranque
                            logger.exception("No se pudo migrar alerta %s a SQLite", item.get("id"))
        except (OSError, ValueError):
            logger.warning("No se pudo leer el historial de alertas previo")

    def _purge_old(self) -> None:
        limit = time.time() - self.s.alert_retention_days * 86400
        for photo in self.dir.glob("*.jpg"):
            try:
                if photo.stat().st_mtime < limit:
                    photo.unlink()
            except OSError:
                pass
