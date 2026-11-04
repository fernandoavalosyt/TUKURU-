"""Panel web (Flask): video en vivo, personas con ID, alertas y alta de camaras con zonas y credenciales."""

from __future__ import annotations

import hmac
import logging
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import __version__
from .app import Tukuru
from .camera import test_connection
from .config import validate_camera

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
