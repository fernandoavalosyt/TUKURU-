"""Pruebas locales del muestreo de aprendizaje y prioridad exploratoria."""

from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from tukuru.alerts import AlertManager
from tukuru.config import Settings
from tukuru.heuristics import TrackState
from tukuru.ml.novelty import novelty_percentiles, reference_readiness
from tukuru.optimizer import FpsGate
from tukuru.pipeline import CameraPipeline
from tukuru.vision import Person
from tukuru.config import validate_camera


def normal_rows(count: int = 30) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append({"incident_id": str(index), "session_id": f"source-{index // 6}",
                     "camera_id": "CAM-01", "track_id": index % 6,
                     "occurred_at": 1000.0 + index * 180.0, "label": 0, "version": 1,
                     "score": (index % 6) * 0.04, "seg_visible": 60, "alcances_2min": 1,
                     "lugar_ocultamiento": "desconocido"})
    return rows


class LearningSamplingTests(unittest.TestCase):
    def test_observation_is_persisted_locally_without_becoming_an_alert(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(11, "CAM-01", time.time() - 5, time.time())
            ok, encoded = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))
            self.assertTrue(ok)
            ts.context_crops.append((time.time(), encoded.tobytes()))

            self.assertTrue(alerts.observe_event(ts, "alcance", time.time(), "Pasillo", "turno-1"))
            self.assertFalse(alerts.observe_event(ts, "alcance", time.time() + 1, "Pasillo", "turno-1"))
            self.assertEqual(alerts.recent(), [])
            cases = alerts.incidents.review_candidates(since=time.time() - 60)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["record_type"], "observacion")
            self.assertEqual(cases[0]["recording_known"], True)
            self.assertIsNotNone(alerts.snapshot_path(cases[0]["snapshot"]))

    def test_pipeline_samples_new_reach_without_emitting_alert(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), data_dir=Path(temp), shadow_mode=False)
            alerts = AlertManager(settings)
            camera = validate_camera({"id": "CAM-01", "host": "192.168.1.20",
                                      "recording_group_id": "tienda-turno-a"})
            pipeline = CameraPipeline(camera, settings, "unused.pt", SimpleNamespace(disabled_reason="off"), alerts)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            now = time.time()
            person = Person(4, np.asarray([10, 5, 40, 90], dtype=np.float32), 0.9,
                            np.zeros((17, 3), dtype=np.float32))
            pipeline.stream = SimpleNamespace(latest=lambda: (frame, 1, now), offline_seconds=lambda: 0.0)
            pipeline.optimizer = SimpleNamespace(process=lambda image: (image, 1.0), is_dark=False)
            pipeline.tracker = SimpleNamespace(infer=lambda *_: [person])
            pipeline._wrist_motion = SimpleNamespace(update=lambda *_: None)

            def add_reach(track, _person, _zones, timestamp, _settings):
                track.events.append((timestamp, "alcance", 0.5))

            with patch("tukuru.pipeline.update_track", side_effect=add_reach), \
                    patch("tukuru.pipeline.decide", return_value=("observacion", "local", "interaccion")), \
                    patch("tukuru.pipeline.draw_tracks", side_effect=lambda image, *_args, **_kwargs: image):
                pipeline._tick(FpsGate(8))

            self.assertEqual(alerts.recent(), [])
            cases = alerts.incidents.review_candidates(since=now - 60)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["recording_known"], True)
            self.assertEqual(cases[0]["reason"], "Muestra local: alcance")

    def test_novelty_priority_stays_off_until_enough_sources_and_negatives(self):
        rows = normal_rows(29)
        _, reason = reference_readiness(rows)
        self.assertIn("30 episodios", reason)
        self.assertEqual(novelty_percentiles([], rows), {})

        rows = normal_rows()
        for row in rows:
            row["session_id"] = "un-solo-grupo"
        _, reason = reference_readiness(rows)
        self.assertIn("5 grupos de origen", reason)
        self.assertEqual(novelty_percentiles([], rows), {})

    def test_novelty_priority_ranks_outlier_without_creating_a_label(self):
        rows = normal_rows()
        cases = [
            {"id": "common", "features": rows[0]},
            {"id": "novel", "features": {"version": 1, "score": 8.0, "seg_visible": 600,
                                             "alcances_2min": 20, "lugar_ocultamiento": "bolsillo"}},
        ]
        scores = novelty_percentiles(cases, rows)
        self.assertGreater(scores["novel"], scores["common"])
        self.assertEqual({row["label"] for row in rows}, {0})


if __name__ == "__main__":
    unittest.main()
