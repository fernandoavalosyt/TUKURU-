"""Sugerencias de zonas: validacion local sin llamar a Gemini."""

import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from tukuru.app import Tukuru
from tukuru.config import Settings
from tukuru.web import create_web
from tukuru.zone_suggestions import DEFAULT_ZONE_MODEL, suggest_zones, validate_proposal


class ZoneSuggestionTests(unittest.TestCase):
    def test_validates_proposed_polygons_without_inventing_missing_zones(self):
        zones, notes = validate_proposal({
            "estanteria": [[[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]]],
            "notas": "No se ve caja ni salida."})
        self.assertEqual(set(zones), {"estanteria"})
        self.assertIn("No se ve caja", notes)
        with self.assertRaisesRegex(ValueError, "coordenadas"):
            validate_proposal({"estanteria": [[[0, 0], [1.1, 0], [0, 1]]]})
        with self.assertRaisesRegex(ValueError, "pequena"):
            validate_proposal({"estanteria": [[[0.1, 0.1], [0.101, 0.1], [0.1, 0.101]]]})
        with self.assertRaisesRegex(ValueError, "no identifico"):
            validate_proposal({})

    def test_only_explicit_post_sends_recent_frame_for_analysis(self):
        settings = Settings.from_env()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        core = Tukuru.__new__(Tukuru)
        core.s = settings
        core._zone_lock = threading.Lock()
        core.pipeline = Mock(return_value=SimpleNamespace(
            stream=SimpleNamespace(latest=lambda: (frame, 1, time.time()))))
        suggestion = ({"estanteria": [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]]}, "Revisar encuadre")
        with patch("tukuru.app.suggest_zones", return_value=suggestion) as ask:
            client = create_web(core).test_client()
            local_frame = client.get("/api/cameras/CAM-01/zone-frame")
            self.assertEqual(local_frame.status_code, 200)
            self.assertTrue(local_frame.json["preview"].startswith("data:image/jpeg;base64,"))
            ask.assert_not_called()
            self.assertEqual(client.get("/api/cameras/CAM-01/suggest-zones").status_code, 405)
            ask.assert_not_called()
            invalid = client.post("/api/cameras/CAM-01/suggest-zones", json={"model": "otro-modelo"},
                                  headers={"X-Requested-With": "tukuru"})
            self.assertEqual(invalid.status_code, 400)
            ask.assert_not_called()
            response = client.post("/api/cameras/CAM-01/suggest-zones", json={},
                                   headers={"X-Requested-With": "tukuru"})
            self.assertEqual(response.status_code, 200)
            sent = ask.call_args.args[0]
            decoded = cv2.imdecode(np.frombuffer(sent, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (360, 640))
            self.assertEqual(response.json["zones"], suggestion[0])
            self.assertEqual(response.json["model"], DEFAULT_ZONE_MODEL)
            self.assertTrue(response.json["preview"].startswith("data:image/jpeg;base64,"))
            ask.side_effect = RuntimeError("detalle privado")
            failed = client.post("/api/cameras/CAM-01/suggest-zones", json={},
                                 headers={"X-Requested-With": "tukuru"})
            self.assertEqual(failed.status_code, 502)
            self.assertIn("RuntimeError", failed.json["error"])
            self.assertNotIn("detalle privado", failed.json["error"])

    def test_stale_frame_does_not_call_gemini(self):
        core = Tukuru.__new__(Tukuru)
        core.s = Settings.from_env()
        core._zone_lock = threading.Lock()
        core.pipeline = Mock(return_value=SimpleNamespace(
            stream=SimpleNamespace(latest=lambda: (np.zeros((10, 10, 3), dtype=np.uint8), 1, time.time() - 10))))
        with patch("tukuru.app.suggest_zones") as ask:
            with self.assertRaisesRegex(ValueError, "cuadro reciente"):
                core.suggest_camera_zones("CAM-01")
            ask.assert_not_called()

    def test_service_errors_are_specific_without_exposing_sdk_message(self):
        from google.genai import errors

        settings = replace(Settings.from_env(), gemini_api_key="x" * 24)
        for code, expected in ((404, "modelo no disponible"), (429, "cuota agotada"),
                               (503, "servicio temporalmente ocupado")):
            with self.subTest(code=code), patch("google.genai.Client") as client:
                client.return_value.models.generate_content.side_effect = errors.APIError(
                    code, {"error": {"message": "detalle privado", "status": "ERROR"}})
                with self.assertRaisesRegex(ValueError, expected) as caught:
                    suggest_zones(b"imagen_falsa", settings)
                self.assertNotIn("detalle privado", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
