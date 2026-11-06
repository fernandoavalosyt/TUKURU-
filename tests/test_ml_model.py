"""Validacion del modelo en sombra y de la separacion de datos."""

import json
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tukuru.alerts import AlertManager
from tukuru.config import Settings
from tukuru.heuristics import TrackState
from tukuru.ml.model import FEATURE_NAMES, ShadowModel, feature_vector
from tukuru.ml.train import fit, split_for_validation


def artifact():
    n = len(FEATURE_NAMES)
    return {"version": "test", "algorithm": "logistic_regression", "feature_version": 1,
            "feature_names": FEATURE_NAMES, "mean": [0] * n, "scale": [1] * n,
            "coef": [0] * n, "intercept": 0}


class ShadowModelTests(unittest.TestCase):
    def test_model_rejects_wrong_schema_and_scores_without_sklearn(self):
        model = ShadowModel(artifact())
        features = {"version": 1, "score": 0.8, "lugar_ocultamiento": "bolsillo"}
        self.assertEqual(model.score(features), 0.5)
        self.assertEqual(len(feature_vector(features)), len(FEATURE_NAMES))
        bad = artifact()
        bad["feature_version"] = 99
        with self.assertRaises(ValueError):
            ShadowModel(bad)

    def test_shadow_score_is_recorded_without_changing_alert_level(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            path.write_text(json.dumps(artifact()))
            settings = replace(Settings.from_env(), data_dir=Path(temp), ml_shadow_model=path, shadow_mode=False)
            alerts = AlertManager(settings)
            ts = TrackState(1, "CAM-01", 0.0, 5.0)
            alerts._emit(ts, "posible_ocultamiento", "local", "Revision", 5.0, "Tienda", None)
            record = alerts.recent()[0]
            self.assertEqual(record["level"], "posible_ocultamiento")
            self.assertEqual(record["ml_shadow_score"], 0.5)

    def test_training_refuses_one_session_and_keeps_sessions_separate(self):
        with self.assertRaisesRegex(ValueError, "4 sesiones"):
            split_for_validation([{"session_id": "una", "camera_id": "C", "track_id": 1,
                                   "occurred_at": 1, "label": 1}])
        rows = [{"session_id": f"s{session}", "camera_id": "C", "track_id": track,
                 "occurred_at": session * 100 + track, "label": track % 2}
                for session in range(4) for track in range(20)]
        train, test = split_for_validation(rows)
        self.assertFalse({r["session_id"] for r in train} & {r["session_id"] for r in test})
        self.assertEqual(len(train) + len(test), len(rows))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn es opcional para entrenar")
    def test_trainer_fits_synthetic_groups_and_exports_runtime_model(self):
        rows = [{"session_id": f"s{session}", "camera_id": "C", "track_id": track,
                 "occurred_at": session * 100 + track, "label": track % 2,
                 "version": 1, "score": 0.8 if track % 2 else 0.2,
                 "lugar_ocultamiento": "bolsillo" if track % 2 else "desconocido"}
                for session in range(4) for track in range(20)]
        trained = fit(rows)
        self.assertGreater(trained["metrics"]["average_precision"], 0.9)
        model = ShadowModel(trained)
        self.assertGreater(model.score(rows[1]), model.score(rows[0]))


if __name__ == "__main__":
    unittest.main()
