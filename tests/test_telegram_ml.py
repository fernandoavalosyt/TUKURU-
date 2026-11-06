"""Contrato local de incidentes y revisiones; no usa red ni modelos."""

import sqlite3
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

from tukuru.alerts import AlertManager
from tukuru.config import Settings, validate_camera
from tukuru.heuristics import TrackState
from tukuru.ml.dataset import labeled_rows
from tukuru.ml.repository import IncidentRepository
from tukuru.telegram import TelegramBridge, parse_review
from tukuru.web import create_web


class IncidentReviewTests(unittest.TestCase):
    def test_recording_group_is_persisted_and_validated(self):
        camera = validate_camera({"host": "192.168.1.10", "recording_group_id": "tienda-turno-01"})
        self.assertEqual(camera.public()["recording_group_id"], "tienda-turno-01")
        self.assertEqual(validate_camera(camera.to_json()).recording_group_id, "tienda-turno-01")
        with self.assertRaises(ValueError):
            validate_camera({"host": "192.168.1.10", "recording_group_id": "tienda/cliente"})

    def test_panel_alerts_omit_binary_clip_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(7, "CAM-01", 95.0, 100.0)
            ts.conceals.append(99.5)
            ts.crops.extend([(99.0, b"jpeg-uno"), (100.0, b"jpeg-dos")])
            alerts._emit(ts, "posible_ocultamiento", "local", "Revisar", 100.0, "Tienda", None)

            self.assertEqual(len(alerts._alerts[-1]["clip_frames"]), 2)
            client = create_web(SimpleNamespace(s=settings, alerts=alerts)).test_client()
            response = client.get("/api/alerts")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json[0]["camera_name"], "Tienda")
            self.assertNotIn("clip_frames", response.json[0])

    def test_review_summary_uses_latest_label_and_current_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(7, "CAM-01", 95.0, 100.0)
            alerts._emit(ts, "posible_ocultamiento", "local", "x", 100.0, "Tienda", None)
            first = alerts.recent()[0]["id"]
            alerts._emit(ts, "probable_sustraccion", "local", "x", 101.0, "Tienda", None)
            second = alerts.recent()[0]["id"]
            self.assertTrue(alerts.review(first, "confirmado", "operador", "panel"))
            self.assertTrue(alerts.review(first, "descartado", "operador", "panel"))
            self.assertTrue(alerts.review(second, "no_concluyente", "operador", "panel"))
            self.assertEqual(alerts.incidents.review_summary(1),
                             {"confirmado": 0, "descartado": 1, "no_concluyente": 1,
                              "sesiones_revisadas": 1})
            with sqlite3.connect(Path(temp) / "incidents.sqlite3") as db:
                db.execute("UPDATE incidents SET feature_version=0 WHERE id=?", (second,))
            self.assertEqual(alerts.incidents.review_summary(1)["no_concluyente"], 0)

    def test_database_migrates_previous_schema_without_erasing_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incidents.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute("""CREATE TABLE incidents (id TEXT PRIMARY KEY, occurred_at REAL NOT NULL,
                    camera_id TEXT NOT NULL, track_id INTEGER NOT NULL, level TEXT NOT NULL,
                    source TEXT NOT NULL, reason TEXT NOT NULL, feature_version INTEGER NOT NULL,
                    features_json TEXT NOT NULL, snapshot TEXT)""")
                db.execute("""INSERT INTO incidents VALUES
                    ('old',1,'CAM-01',1,'posible_ocultamiento','local','x',0,'{}',NULL)""")
            IncidentRepository(path)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("SELECT session_id, ml_shadow_score, recording_id FROM incidents WHERE id='old'").fetchone(),
                                 ("legacy", None, None))

    def test_review_batch_and_queue_keep_cases_reviewable(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            for timestamp in (time.time() - 50.0, time.time()):
                ts = TrackState(7, "CAM-01", 95.0, timestamp)
                alerts._emit(ts, "posible_ocultamiento", "local", "Revisar", timestamp, "Tienda", None,
                             recording_id="misma-fuente")
            client = create_web(SimpleNamespace(s=settings, alerts=alerts)).test_client()
            headers = {"X-Requested-With": "tukuru"}
            queue = client.get("/api/reviews/queue")
            self.assertEqual(queue.status_code, 200)
            self.assertEqual(len(queue.json), 1)  # alertas repetidas del mismo track se presentan como un caso
            ids = [row["id"] for row in alerts.recent()]
            response = client.post("/api/reviews/batch", json={"ids": ids, "label": "no_concluyente"}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["saved"], 2)
            self.assertEqual(labeled_rows(Path(temp) / "incidents.sqlite3"), [])

    def test_unidentified_recordings_are_not_exported_for_training(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(7, "CAM-01", 95.0, 100.0)
            alerts._emit(ts, "posible_ocultamiento", "local", "Revisar", 100.0, "Tienda", None)
            alerts.review(alerts.recent()[0]["id"], "confirmado", "op", "panel")
            self.assertEqual(labeled_rows(Path(temp) / "incidents.sqlite3"), [])
            quality = alerts.incidents.quality_summary(1)
            self.assertEqual(quality["origen_desconocido"], 1)
            self.assertEqual(quality["filas_entrenables"], 0)
            incident_id = alerts.recent()[0]["id"]
            client = create_web(SimpleNamespace(s=settings, alerts=alerts)).test_client()
            result = client.post("/api/recordings/assign", json={"ids": [incident_id],
                "recording_id": "fuente-verificada"}, headers={"X-Requested-With": "tukuru"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json["updated"], 1)
            self.assertEqual(labeled_rows(Path(temp) / "incidents.sqlite3")[0]["session_id"], "fuente-verificada")

    def test_alert_and_review_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(7, "CAM-01", 95.0, 100.0)
            ts.score = 0.8
            alerts._emit(ts, "posible_ocultamiento", "local", "Revision humana requerida", 100.0,
                         "Tienda", None, recording_id="grabacion-prueba")
            record = alerts.recent()[0]
            self.assertEqual(record["features"]["version"], 1)
            self.assertTrue(alerts.review(record["id"], "descartado", 42))
            self.assertTrue(alerts.review(record["id"], "descartado", 42))
            self.assertFalse(alerts.review("ffffffff", "confirmado", 42))
            with sqlite3.connect(Path(temp) / "incidents.sqlite3") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM incidents").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT count(*) FROM reviews").fetchone()[0], 1)
            self.assertEqual(labeled_rows(Path(temp) / "incidents.sqlite3")[0]["label"], 0)
            self.assertTrue(alerts.review(record["id"], "no_concluyente", 42))
            self.assertEqual(labeled_rows(Path(temp) / "incidents.sqlite3"), [])
            restored = AlertManager(settings).recent()[0]
            self.assertEqual(restored["review"], "no_concluyente")

    def test_panel_review_endpoint_uses_same_local_record(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(7, "CAM-01", 95.0, 100.0)
            alerts._emit(ts, "posible_ocultamiento", "local", "Revisar", 100.0, "Tienda", None)
            alert_id = alerts.recent()[0]["id"]
            client = create_web(SimpleNamespace(s=settings, alerts=alerts)).test_client()
            headers = {"X-Requested-With": "tukuru"}
            result = client.post(f"/api/alerts/{alert_id}/review", json={"label": "confirmado"}, headers=headers)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(alerts.recent()[0]["review"], "confirmado")
            with sqlite3.connect(Path(temp) / "incidents.sqlite3") as db:
                self.assertEqual(db.execute("SELECT label, reviewer_source FROM reviews").fetchone(),
                                 ("confirmado", "panel"))

    def test_telegram_review_still_works_after_panel_dismisses_alert(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp))
            alerts = AlertManager(settings)
            ts = TrackState(7, "CAM-01", 95.0, 100.0)
            alerts._emit(ts, "posible_ocultamiento", "local", "Revisar", 100.0, "Tienda", None,
                         recording_id="telegram-test-source")
            alert_id = alerts.recent()[0]["id"]
            self.assertTrue(alerts.dismiss(alert_id))
            self.assertTrue(alerts.review(alert_id, "confirmado", 42))
            self.assertEqual(labeled_rows(Path(temp) / "incidents.sqlite3")[0]["label"], 1)

    def test_telegram_callback_requires_allowed_user_and_chat(self):
        update = {"callback_query": {"id": "q", "from": {"id": 42},
                                     "message": {"chat": {"id": -100123}},
                                     "data": "review:abcdef12:n"}}
        self.assertEqual(parse_review(update, "-100123", (42,)),
                         ("abcdef12", "no_concluyente", 42))
        self.assertIsNone(parse_review(update, "-100123", (43,)))
        with tempfile.TemporaryDirectory() as temp:
            review = Mock(return_value=True)
            bridge = TelegramBridge("token", "-100123", (42,), review, Path(temp))
            bridge._call = Mock(return_value={"ok": True})
            bridge._handle(update)
            review.assert_called_once_with("abcdef12", "no_concluyente", 42)
            bridge._call.assert_called_once_with("answerCallbackQuery",
                                                 {"callback_query_id": "q", "text": "Revision guardada"})

    def test_telegram_message_has_review_buttons_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            bridge = TelegramBridge("token", "-100123", (42,), Mock(), Path(temp))
            bridge._call = Mock(return_value={"ok": True})
            bridge._send({"id": "abcdef12", "camera_name": "Tienda", "track_id": 7,
                          "level": "posible_ocultamiento", "reason": "Revisar", "snapshot": None})
            method, data = bridge._call.call_args.args
            self.assertEqual(method, "sendMessage")
            self.assertEqual(data["chat_id"], "-100123")
            self.assertIn("review:abcdef12:d", data["reply_markup"])


if __name__ == "__main__":
    unittest.main()
