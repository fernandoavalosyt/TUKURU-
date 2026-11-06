"""Panel web (Flask): video en vivo, personas con ID, alertas y alta de camaras con zonas y credenciales."""

from __future__ import annotations

import hmac
import logging
import re
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import __version__
from .app import Tukuru
from .camera import test_connection
from .config import validate_camera
from .ml.dataset import labeled_rows
from .ml.novelty import novelty_percentiles
from .zone_suggestions import DEFAULT_ZONE_MODEL

logger = logging.getLogger("tukuru.web")
STATIC = Path(__file__).parent / "static"


def create_web(app_core: Tukuru) -> Flask:
    web = Flask(__name__, static_folder=None)
    web.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    s = app_core.s

    @web.before_request
    def guard():
        if s.web_user and s.web_password:
            auth = request.authorization
            if not (auth and hmac.compare_digest(auth.username or "", s.web_user)
                    and hmac.compare_digest(auth.password or "", s.web_password)):
                return Response("Autenticacion requerida", 401, {"WWW-Authenticate": 'Basic realm="Tukuru"'})
        # Los cambios solo se aceptan como JSON (evita formularios cruzados desde otros sitios).
        if request.method in ("POST", "PUT", "DELETE") and request.path.startswith("/api/"):
            if request.method != "DELETE" and not request.is_json:
                return jsonify(error="Content-Type debe ser application/json"), 415
            if request.headers.get("X-Requested-With") != "tukuru":
                return jsonify(error="Falta la cabecera X-Requested-With"), 400

    @web.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @web.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @web.get("/healthz")
    def healthz():
        return jsonify(status="ok", version=__version__)

    @web.get("/api/status")
    def status():
        return jsonify(app_core.status())

    @web.post("/api/gemini")
    def set_gemini():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or type(payload.get("enabled")) is not bool:
            return jsonify(error="enabled debe ser true o false"), 400
        try:
            return jsonify(app_core.set_gemini_enabled(payload["enabled"]))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    # ---- video ------------------------------------------------------------
    @web.get("/stream/<cam_id>")
    def stream(cam_id: str):
        pipeline = app_core.pipeline(cam_id)
        if pipeline is None:
            abort(404)

        def generate():
            last = None
            while True:
                jpeg = pipeline.jpeg()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() \
                        + b"\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.1)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---- alertas ------------------------------------------------------------
    @web.get("/api/alerts")
    def alerts():
        limit = request.args.get("limit", 50, type=int)
        return jsonify(app_core.alerts.recent(max(1, min(limit, 200))))

    @web.post("/api/alerts/<alert_id>/ack")
    def ack(alert_id: str):
        return (jsonify(ok=True) if app_core.alerts.ack(alert_id) else (jsonify(error="no existe"), 404))

    @web.post("/api/alerts/<alert_id>/review")
    def review_alert(alert_id: str):
        payload = request.get_json(silent=True) or {}
        label = payload.get("label") if isinstance(payload, dict) else None
        if label not in ("confirmado", "descartado", "no_concluyente"):
            return jsonify(error="revision invalida"), 400
        reviewer = s.web_user or "operador_local"
        return (jsonify(ok=True) if app_core.alerts.review(alert_id, label, reviewer, "panel")
                else (jsonify(error="alerta no disponible"), 404))

    @web.post("/api/reviews/batch")
    def review_batch():
        payload = request.get_json(silent=True) or {}
        ids, label = payload.get("ids"), payload.get("label")
        if (not isinstance(ids, list) or not ids or len(ids) > 100 or
                any(not isinstance(item, str) for item in ids) or
                label not in ("confirmado", "descartado", "no_concluyente")):
            return jsonify(error="ids (1-100) y label de revision son requeridos"), 400
        reviewer = s.web_user or "operador_local"
        saved = sum(app_core.alerts.review(item, label, reviewer, "panel_lote") for item in set(ids))
        return jsonify(ok=saved == len(set(ids)), saved=saved, requested=len(set(ids)))

    @web.get("/api/reviews/queue")
    def review_queue():
        limit = max(1, min(request.args.get("limit", 100, type=int), 200))
        since = time.time() - s.alert_retention_days * 86400
        cases = app_core.alerts.incidents.review_candidates(limit, since)
        try:
            references = labeled_rows(app_core.alerts.incidents.path)
            novelty = novelty_percentiles(cases, references)
        except Exception:  # noqa: BLE001 - la prioridad experimental no debe afectar la cola
            logger.exception("No se pudo calcular la prioridad exploratoria de novedad")
            novelty = {}
        for case in cases:
            case["evidence_url"] = f"/evidence/{case['snapshot']}" if case.get("snapshot") else None
            case["novelty_percentile"] = novelty.get(case["id"])
            case.pop("features", None)
        cases.sort(key=lambda case: (case["shadow_score"] is None,
            abs((case["shadow_score"] or 0.0)-0.5) if case["shadow_score"] is not None else 1.0,
            -(case["novelty_percentile"] or 0.0), -case["time"]))
        return jsonify(cases)

    @web.post("/api/recordings/assign")
    def assign_recording():
        payload = request.get_json(silent=True) or {}
        ids, recording_id = payload.get("ids"), payload.get("recording_id")
        if (not isinstance(ids, list) or not ids or len(ids) > 100 or
                any(not isinstance(item, str) for item in ids) or
                not isinstance(recording_id, str) or
                not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", recording_id)):
            return jsonify(error="seleccione de 1 a 100 incidentes e indique un ID de origen valido"), 400
        updated = app_core.alerts.incidents.assign_recording(ids, recording_id)
        return jsonify(ok=True, updated=updated)

    @web.delete("/api/alerts/<alert_id>")
    def alert_dismiss(alert_id: str):
        return (jsonify(ok=True) if app_core.alerts.dismiss(alert_id) else (jsonify(error="no existe"), 404))

    @web.delete("/api/alerts")
    def alerts_dismiss_all():
        return jsonify(ok=True, removed=app_core.alerts.dismiss_all())

    @web.get("/evidence/<name>")
    def evidence(name: str):
        path = app_core.alerts.snapshot_path(name)
        if path is None:
            abort(404)
        return send_from_directory(path.parent, path.name)

    # ---- camaras (credenciales) ---------------------------------------------
    @web.get("/api/cameras")
    def cameras():
        return jsonify([{**c.public(), "password_in": app_core.store.where_password(c.id)}
                        for c in app_core.store.list()])

    @web.get("/api/cameras/<cam_id>/zone-frame")
    def camera_zone_frame(cam_id: str):
        try:
            return jsonify(app_core.camera_zone_frame(cam_id))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @web.post("/api/cameras/<cam_id>/suggest-zones")
    def suggest_camera_zones(cam_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Solicitud invalida"), 400
        model = payload.get("model") or DEFAULT_ZONE_MODEL
        if not isinstance(model, str) or model not in (DEFAULT_ZONE_MODEL, s.gemini_model):
            return jsonify(error="Modelo de zonas no permitido"), 400
        try:
            return jsonify(app_core.suggest_camera_zones(cam_id, model))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:  # noqa: BLE001 - errores de SDK pueden incluir datos de solicitud
            logger.warning("No se pudieron sugerir zonas para %s (%s)", cam_id, type(exc).__name__)
            return jsonify(error=f"Gemini {model}: fallo inesperado ({type(exc).__name__}); revise el registro"), 502

    def _parse(payload):
        if not isinstance(payload, dict):
            raise ValueError("payload invalido")
        previous = app_core.store.get(str(payload.get("id") or ""))
        if previous and "zones" not in payload:       # un cliente que no menciona zonas no las borra
            payload = {**payload, "zones": previous.zones}
        return validate_camera(payload, existing_password=previous.password if previous else "")

    @web.post("/api/cameras/test")
    def camera_test():
        try:
            cfg = _parse(request.get_json())
        except ValueError as exc:
            return jsonify(ok=False, message=str(exc)), 400
        return jsonify(test_connection(cfg))

    @web.post("/api/cameras")
    def camera_save():
        try:
            cfg = _parse(request.get_json())
            saved = app_core.save_camera(cfg)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(saved.public()), 201

    @web.post("/api/cameras/<cam_id>/enabled")
    def camera_enabled(cam_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify(error="enabled debe ser true o false"), 400
        cam = app_core.set_enabled(cam_id, payload["enabled"])
        return (jsonify(cam.public()) if cam else (jsonify(error="no existe"), 404))

    @web.delete("/api/cameras/<cam_id>")
    def camera_delete(cam_id: str):
        return (jsonify(ok=True) if app_core.delete_camera(cam_id) else (jsonify(error="no existe"), 404))

    @web.errorhandler(400)
    @web.errorhandler(413)
    def bad_request(exc):
        return jsonify(error="solicitud invalida"), exc.code

    return web
