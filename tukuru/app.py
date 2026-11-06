"""Supervisor: crea y administra los pipelines de camara en caliente."""

from __future__ import annotations

import logging
import json
import base64
import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from dataclasses import replace

from .alerts import AlertManager
from .config import CameraConfig, CameraStore, ConfigError, Settings
from .credentials import CredentialError, open_credentials
from .gemini import GeminiJudge
from .pipeline import CameraPipeline
from .vision import ensure_model
from .telegram import TelegramBridge
from .ml.features import FEATURE_VERSION
from .zone_suggestions import DEFAULT_ZONE_MODEL, encode_frame, suggest_zones

logger = logging.getLogger("tukuru.app")


def saved_gemini_enabled(path: Path, default: bool) -> bool:
    """La preferencia del panel prevalece sobre el valor inicial del entorno."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved.get("enabled"), bool):
            return saved["enabled"]
    except (OSError, ValueError, AttributeError):
        pass
    return default


class Tukuru:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            credentials = open_credentials(settings.credentials_store)
        except CredentialError as exc:
            raise ConfigError(str(exc)) from exc
        self.store = CameraStore(settings.data_dir / "cameras.json", settings.max_cameras, credentials)
        self.alerts = AlertManager(settings)
        self.telegram = (TelegramBridge(settings.telegram_bot_token, settings.telegram_chat_id,
                                        settings.telegram_allowed_user_ids, self.alerts.review, self.alerts.dir)
                         if settings.telegram_bot_token and settings.telegram_chat_id else None)
        if self.telegram:
            self.alerts.set_notifier(self.telegram.notify)
        self._gemini_state_path = settings.data_dir / "gemini_state.json"
        gemini_enabled = saved_gemini_enabled(self._gemini_state_path, settings.gemini_enabled)
        self.judge = GeminiJudge(replace(settings, gemini_enabled=gemini_enabled))
        self.model_path = ensure_model(settings.models_dir, settings.yolo_model)
        self._pipelines: Dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._zone_lock = threading.Lock()
        self.started = time.time()

    def start(self) -> None:
        if self.telegram:
            self.telegram.start()
        for cam in self.store.list():
            if cam.enabled:
                self._start_pipeline(cam)
        if not self._pipelines:
            logger.warning("Sin camaras configuradas: agreguelas desde el panel web.")

    def stop(self) -> None:
        with self._lock:
            pipelines = list(self._pipelines.values())
            self._pipelines.clear()
        for p in pipelines:
            p.stop()
        for p in pipelines:
            p.join(timeout=5)
        self.judge.shutdown()
        if self.telegram:
            self.telegram.stop()

    def _start_pipeline(self, cam: CameraConfig) -> None:
        pipeline = CameraPipeline(cam, self.s, self.model_path, self.judge, self.alerts)
        with self._lock:
            self._pipelines[cam.id] = pipeline
        pipeline.start()

    # ---- administracion de camaras (API) ----------------------------------
    def save_camera(self, cam: CameraConfig) -> CameraConfig:
        cam = self.store.upsert(cam)
        self._stop_pipeline(cam.id)
        if cam.enabled:
            self._start_pipeline(cam)
        return cam

    def set_enabled(self, cam_id: str, enabled: bool) -> Optional[CameraConfig]:
        cam = self.store.get(cam_id)
        return self.save_camera(replace(cam, enabled=enabled)) if cam else None

    def delete_camera(self, cam_id: str) -> bool:
        removed = self.store.remove(cam_id)
        self._stop_pipeline(cam_id)
        return removed

    def _stop_pipeline(self, cam_id: str) -> None:
        with self._lock:
            pipeline = self._pipelines.pop(cam_id, None)
        if pipeline:
            pipeline.stop()
            pipeline.join(timeout=5)

    def pipeline(self, cam_id: str) -> Optional[CameraPipeline]:
        with self._lock:
            return self._pipelines.get(cam_id)

    def set_gemini_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Aplica la preferencia ahora y la conserva sin escribir en .env."""
        self.judge.set_enabled(enabled)
        saved = True
        tmp = self._gemini_state_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps({"enabled": enabled}), encoding="utf-8")
            tmp.replace(self._gemini_state_path)
        except OSError:
            saved = False
            logger.exception("No se pudo guardar la preferencia de Gemini")
        return {"gemini": self.judge.health(), "saved": saved}

    def _camera_zone_jpeg(self, cam_id: str) -> bytes:
        pipeline = self.pipeline(cam_id)
        if pipeline is None:
            raise ValueError("La camara debe estar conectada para editar sus zonas")
        frame, _, captured_at = pipeline.stream.latest()
        if frame is None or time.time() - captured_at > 5:
            raise ValueError("La camara no tiene un cuadro reciente; revise la conexion")
        return encode_frame(frame)

    def camera_zone_frame(self, cam_id: str) -> Dict[str, str]:
        """Captura local para dibujar zonas, sin consultar servicios externos."""
        jpeg = self._camera_zone_jpeg(cam_id)
        return {"preview": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}

    def suggest_camera_zones(self, cam_id: str, model: str = DEFAULT_ZONE_MODEL) -> Dict[str, Any]:
        """Solo una pulsacion del panel envia una captura; no guarda zonas automaticamente."""
        jpeg = self._camera_zone_jpeg(cam_id)
        if not self._zone_lock.acquire(blocking=False):
            raise ValueError("Ya hay un analisis de zonas en curso")
        try:
            zones, notes = suggest_zones(jpeg, self.s, model)
            return {"zones": zones, "notes": notes, "model": model,
                    "preview": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}
        finally:
            self._zone_lock.release()

    # ---- estado -------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            pipelines = list(self._pipelines.values())
        cameras: List[Dict[str, Any]] = [p.status() for p in pipelines]
        running = {c["camera"]["id"] for c in cameras}
        stopped = [{"camera": c.public(), "stream": {"state": "disabled"}, "people": [], "fps": 0}
                   for c in self.store.list() if c.id not in running]
        try:
            learning = self.alerts.incidents.review_summary(FEATURE_VERSION)
            learning["calidad_datos"] = self.alerts.incidents.quality_summary(FEATURE_VERSION)
        except Exception:  # noqa: BLE001 - una consulta de estado no debe tumbar el panel
            logger.exception("No se pudo consultar resumen de revisiones")
            learning = None
        return {"uptime_seconds": round(time.time() - self.started), "cameras": cameras + stopped,
                "gemini": self.judge.health(), "active_alerts": self.alerts.active_count(),
                "max_cameras": self.s.max_cameras, "credentials_store": self.store.mode,
                "learning": learning, "ml_shadow_active": self.alerts.shadow_model is not None,
                "telegram_configured": self.telegram is not None}
