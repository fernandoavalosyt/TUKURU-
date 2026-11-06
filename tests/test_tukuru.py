import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.preparar_entorno import cuda_index_for
from tukuru.alerts import AlertManager
from tukuru.config import ConfigError, Settings, validate_camera
from tukuru.decision import decide
from tukuru.heuristics import TrackState, clear_through, update_track
from tukuru.vision import Person, PoseTracker, fp16_arguments, resolve_device
from tukuru.tracker_config import bytetrack_config
from tukuru.telegram import parse_review
from tukuru.zones import ZoneSet


def kp(over=None):
    """Persona de pie con los brazos a los lados (coordenadas de 640x360)."""
    pts = {0: (320, 100), 1: (310, 92), 2: (330, 92), 3: (298, 96), 4: (342, 96), 5: (280, 150), 6: (360, 150),
           7: (270, 200), 8: (370, 200), 9: (230, 250), 10: (410, 250), 11: (290, 260), 12: (350, 260)}
    pts.update(over or {})
    k = np.zeros((17, 3))
    for i, (x, y) in pts.items():
        k[i] = (x, y, 0.9)
    return k


BBOX = np.array([250, 70, 390, 300.0])                 # pies en (320, 300)
EXIT_BBOX = np.array([480, 70, 620, 300.0])            # pies en la zona de salida
CAJA_BBOX = np.array([250, 10, 390, 100.0])            # pies en la zona de caja
STAFF_BBOX = np.array([10, 70, 100, 300.0])            # pies en la zona de personal
REACH = {9: (150, 150)}                                 # muneca izquierda dentro de la estanteria
POCKET = {9: (300, 262)}                                # muneca izquierda en el bolsillo
ZONES = {"estanteria": [[[0.0, 0.1], [0.3, 0.1], [0.3, 0.7], [0.0, 0.7]]],
         "caja": [[[0.35, 0.0], [0.65, 0.0], [0.65, 0.3], [0.35, 0.3]]],
         "salida": [[[0.8, 0.5], [1.0, 0.5], [1.0, 1.0], [0.8, 1.0]]],
         "personal": [[[0.0, 0.8], [0.2, 0.8], [0.2, 1.0], [0.0, 1.0]]]}


class CameraValidation(unittest.TestCase):
    def test_url_is_split_and_password_masked(self):
        cam = validate_camera({"url": "rtsp://admin:p%40ss@192.168.1.10:554/Streaming/Channels/102"})
        self.assertEqual((cam.user, cam.password, cam.host, cam.port), ("admin", "p@ss", "192.168.1.10", 554))
        self.assertIn("p%40ss", cam.source())
        self.assertNotIn("p%40ss", cam.safe_url())
        self.assertNotIn("password", cam.public())

    def test_url_without_credentials_keeps_separate_user_password(self):
        cam = validate_camera({"url": "rtsp://192.168.1.3:8080/h264_ulaw.sdp", "user": "avalos", "password": "x"})
        self.assertEqual((cam.user, cam.password), ("avalos", "x"))
        self.assertEqual(cam.path, "/h264_ulaw.sdp")

    def test_brand_default_path_and_port(self):
        cam = validate_camera({"host": "10.0.0.5", "brand": "dahua", "user": "u", "password": "x"})
        self.assertEqual(cam.port, 554)
        self.assertIn("subtype=1", cam.path)

    def test_rejects_bad_input(self):
        for bad in ({"host": "a/b"}, {"host": "10.0.0.5", "port": 99999}, {"host": "10.0.0.5", "protocol": "file"},
                    {"host": "10.0.0.5", "id": "../x"}, {"host": "10.0.0.5", "password": "x"},
                    {"host": "10.0.0.5", "user": "a\nb"}, {"host": "10.0.0.5", "path": "rtsp://evil"},
                    {"host": ""}):
            with self.assertRaises(ValueError, msg=bad):
                validate_camera(bad)

    def test_existing_password_kept_on_edit(self):
        cam = validate_camera({"host": "10.0.0.5", "user": "u"}, existing_password="secreto")
        self.assertEqual(cam.password, "secreto")


class SettingsValidation(unittest.TestCase):
    def test_invalid_values_reported_together(self):
        os.environ.update(INFERENCE_FPS="abc", YOLO_CONF="5", WEB_HOST="0.0.0.0")
        try:
            with self.assertRaises(ConfigError) as ctx:
                Settings.from_env()
            msg = str(ctx.exception)
            self.assertIn("INFERENCE_FPS", msg)
            self.assertIn("YOLO_CONF", msg)
            self.assertIn("DASHBOARD_PASSWORD", msg)
        finally:
            for k in ("INFERENCE_FPS", "YOLO_CONF", "WEB_HOST"):
                os.environ.pop(k, None)

    def test_defaults_ok(self):
        s = Settings.from_env()
        self.assertEqual(s.inference_fps, 8.0)
        self.assertEqual(s.max_cameras, 3)
        self.assertEqual(s.track_buffer_frames, 60)

    def test_track_buffer_is_configurable_with_safe_range(self):
        for value in (15, 30, 60, 240):
            with patch.dict(os.environ, {"TRACK_BUFFER_FRAMES": str(value)}):
                self.assertEqual(Settings.from_env().track_buffer_frames, value)
        with patch.dict(os.environ, {"TRACK_BUFFER_FRAMES": "241"}):
            with self.assertRaisesRegex(ConfigError, "TRACK_BUFFER_FRAMES"):
                Settings.from_env()

    def test_tracker_yaml_uses_configured_buffer_without_mutating_ultralytics_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(bytetrack_config(Path(tmp), 60))
            content = path.read_text(encoding="utf-8")
            self.assertIn("track_buffer: 60", content)
            self.assertIn("track_low_thresh: 0.1", content)
            self.assertTrue(path.is_relative_to(Path(tmp)))

    def test_device_choices_and_default(self):
        with patch.dict(os.environ, {"DEVICE": ""}):
            self.assertEqual(Settings.from_env().device, "auto")
        for choice in ("auto", "cpu", "cuda:0", "mps"):
            with patch.dict(os.environ, {"DEVICE": choice}):
                self.assertEqual(Settings.from_env().device, choice)
        with patch.dict(os.environ, {"DEVICE": "cuda:99"}):
            with self.assertRaisesRegex(ConfigError, "DEVICE"):
                Settings.from_env()

    def test_telegram_requires_complete_configuration(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token", "TELEGRAM_CHAT_ID": "",
                                     "TELEGRAM_ALLOWED_USER_IDS": ""}):
            with self.assertRaisesRegex(ConfigError, "Telegram requiere"):
                Settings.from_env()
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token", "TELEGRAM_CHAT_ID": "-100123",
                                     "TELEGRAM_ALLOWED_USER_IDS": "123,456"}):
            s = Settings.from_env()
            self.assertEqual((s.telegram_chat_id, s.telegram_allowed_user_ids), ("-100123", (123, 456)))
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "",
                                     "TELEGRAM_ALLOWED_USER_IDS": ""}):
            self.assertIsNone(Settings.from_env().telegram_bot_token)

    def test_shadow_model_path_is_relative_to_project(self):
        with patch.dict(os.environ, {"ML_SHADOW_MODEL": "data/models_ml/candidato.json"}):
            path = Settings.from_env().ml_shadow_model
        self.assertEqual(path, Path(__file__).resolve().parent.parent / "data/models_ml/candidato.json")

    def test_probable_score_must_exceed_possible(self):
        os.environ.update(LOCAL_POSSIBLE_SCORE="0.9", LOCAL_PROBABLE_SCORE="0.8")
        try:
            with self.assertRaises(ConfigError):
                Settings.from_env()
        finally:
            for k in ("LOCAL_POSSIBLE_SCORE", "LOCAL_PROBABLE_SCORE"):
                os.environ.pop(k, None)


class TelegramReview(unittest.TestCase):
    def test_only_allowed_user_in_configured_chat_can_review(self):
        update = {"callback_query": {"from": {"id": 42}, "message": {"chat": {"id": -100123}},
                                     "data": "review:abcdef12:c"}}
        self.assertEqual(parse_review(update, "-100123", (42,)), ("abcdef12", "confirmado", 42))
        self.assertIsNone(parse_review(update, "-100999", (42,)))
        self.assertIsNone(parse_review(update, "-100123", (43,)))
        update["callback_query"]["data"] = "review:abcdef12:bad"
        self.assertIsNone(parse_review(update, "-100123", (42,)))

class DeviceSelection(unittest.TestCase):
    def test_fp16_arguments_follow_ultralytics_version(self):
        with patch.dict(sys.modules, {"ultralytics.cfg": types.SimpleNamespace(DEFAULT_CFG_DICT={"quantize": None})}):
            self.assertEqual(fp16_arguments(), {"quantize": 16})
        with patch.dict(sys.modules, {"ultralytics.cfg": types.SimpleNamespace(DEFAULT_CFG_DICT={})}):
            self.assertEqual(fp16_arguments(), {"half": True})

    def test_cuda_index_matches_driver_support(self):
        self.assertTrue(cuda_index_for("13.1").endswith("/cu128"))
        self.assertTrue(cuda_index_for("13.2").endswith("/cu132"))
        self.assertTrue(cuda_index_for("12.6").endswith("/cu126"))
        with self.assertRaises(ValueError):
            cuda_index_for("12.5")

    def test_auto_and_explicit_fallback(self):
        self.assertEqual(resolve_device("auto", True, False), "cuda:0")
        self.assertEqual(resolve_device("auto", False, True), "cpu")
        self.assertEqual(resolve_device("cuda:0", False, False), "cpu")
        self.assertEqual(resolve_device("mps", False, False), "cpu")
        self.assertEqual(resolve_device("mps", False, True), "mps")
        self.assertEqual(resolve_device("cpu", True, True), "cpu")

    def test_cuda_warmup_error_falls_back_to_cpu_without_real_torch(self):
        devices = []

        class FakeYOLO:
            def __init__(self, path):
                self.path = path

            def predict(self, image, *, device, half=False, **kwargs):
                devices.append((device, half))
                if device == "cuda:0":
                    raise RuntimeError("CUDA out of memory")

        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )
        with patch.dict(sys.modules, {"torch": torch, "ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)}):
            tracker = PoseTracker("falso.pt", "auto", 0.4, 32)
        self.assertEqual((tracker.device, tracker.half), ("cpu", False))
        self.assertEqual(devices, [("cuda:0", True), ("cpu", False)])

    def test_cuda_inference_error_retries_on_cpu_without_real_torch(self):
        calls = []

        class FakeYOLO:
            def __init__(self, path):
                self.path = path

            def predict(self, image, *, device, **kwargs):
                calls.append(("warmup", device))

            def track(self, image, *, device, **kwargs):
                calls.append(("track", device))
                if device == "cuda:0":
                    raise RuntimeError("CUDA out of memory")
                return []

        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )
        with patch.dict(sys.modules, {"torch": torch, "ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)}):
            tracker = PoseTracker("falso.pt", "auto", 0.4, 32)
            self.assertEqual(tracker.infer(np.zeros((32, 32, 3), dtype=np.uint8), 32), [])
        self.assertEqual((tracker.device, tracker.half), ("cpu", False))
        self.assertEqual(calls, [("warmup", "cuda:0"), ("track", "cuda:0"),
                                 ("warmup", "cpu"), ("track", "cpu")])


class Zones(unittest.TestCase):
    def test_valid_zones_roundtrip(self):
        cam = validate_camera({"host": "10.0.0.5", "zones": ZONES})
        self.assertEqual(set(cam.zones), set(ZONES))
        self.assertEqual(cam.public()["zones"], cam.zones)

    def test_rejects_bad_zones(self):
        poly = [[[0, 0], [1, 0], [1, 1]]]
        for bad in ({"pasillo": poly}, {"caja": [[[0, 0], [1, 0]]]}, {"caja": [[[0, 0], [2, 0], [1, 1]]]}):
            with self.assertRaises(ValueError, msg=bad):
                validate_camera({"host": "10.0.0.5", "zones": bad})

    def test_zoneset_contains(self):
        zs = ZoneSet(ZONES, 640, 360)
        self.assertTrue(zs.contains("estanteria", (100, 100)))
        self.assertFalse(zs.contains("estanteria", (300, 100)))
        self.assertFalse(zs.contains("inexistente", (1, 1)))


class Scene:
    """Escenas sinteticas a 5 cuadros por segundo."""

    def setUp(self):
        self.s = Settings.from_env()
        self.zones = ZoneSet(ZONES, 640, 360)
        self.ts = TrackState(1, "C", 0.0, 0.0)
        self.t = 0.0

    def play(self, seconds, over=None, bbox=BBOX, nose=None):
        end = self.t + seconds
        while self.t < end:
            over_now = dict(over or {})
            if nose is not None:
                over_now[0] = nose
            update_track(self.ts, Person(1, bbox, 0.9, kp(over_now)), self.zones, self.t, self.s)
            self.t += 0.2

    def steal(self, reach=1.0, pocket=1.5):
        self.play(reach, REACH)
        self.play(pocket, POCKET)


class PanelStage(Scene, unittest.TestCase):
    """Lo que el panel y el overlay muestran de la secuencia de sustraccion."""

    def test_nothing_before_any_reach(self):
        self.play(3)
        self.assertEqual(self.ts.hand_states(self.t), ["", ""])
        self.assertFalse(any(self.ts.stage(self.t).values()))

    def test_hand_in_shelf_is_reach_then_watched(self):
        self.play(1.0, REACH)
        self.assertEqual(self.ts.hand_states(self.t - 0.2), ["alcance", ""])
        self.play(0.6)                                   # saco la mano: sigue vigilada hasta que expire la ventana
        self.assertEqual(self.ts.hand_states(self.t), ["vigilada", ""])
        self.assertTrue(self.ts.stage(self.t)["alcance"])
        self.assertFalse(self.ts.stage(self.t)["ocultamiento"])

    def test_stage_follows_the_sequence(self):
        self.steal()
        self.assertTrue(self.ts.stage(self.t)["ocultamiento"])
        self.assertFalse(self.ts.stage(self.t)["salida"])
        self.play(1.5, bbox=EXIT_BBOX)
        self.assertTrue(self.ts.stage(self.t)["salida"])
        self.assertFalse(self.ts.stage(self.t)["caja"])


class Overlay(Scene, unittest.TestCase):
    def test_draw_tracks_runs_with_zones_and_sequence(self):
        from tukuru.render import draw_tracks
        self.steal()
        self.ts.verdict = {"state": "probable_sustraccion", "confidence": 0.9}
        frame = np.zeros((360, 640, 3), np.uint8)
        out = draw_tracks(frame, [self.ts], "Cam", 8.0, self.zones)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)              # dibujo algo
        self.assertEqual(int(frame.sum()), 0)              # y no modifico el original


class Sequence(Scene, unittest.TestCase):
    def test_hands_at_sides_never_suspect(self):
        self.play(60)
        self.assertEqual(self.ts.local_state, "normal")
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "ok")
        self.assertFalse(self.ts.conceals)

    def test_reach_and_return_is_not_concealment(self):
        self.play(2)
        self.play(1.0, REACH)
        self.play(3)
        self.assertEqual(len(self.ts.reaches), 1)
        self.assertFalse(self.ts.conceals)
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "ok")

    def test_brush_past_shelf_is_ignored(self):
        self.play(0.2, REACH)
        self.play(3)
        self.assertFalse(self.ts.reaches)

    def test_hand_in_pocket_without_reach_is_ignored(self):
        self.play(5, POCKET)
        self.assertFalse(self.ts.conceals)

    def test_long_browsing_at_shelf_does_not_arm_concealment(self):
        self.play(8, REACH)
        self.play(3, POCKET)
        self.assertFalse(self.ts.conceals)

    def test_reach_then_pocket_triggers_concealment_and_review(self):
        self.play(2)
        self.steal()
        self.assertEqual(len(self.ts.conceals), 1)
        self.assertEqual(self.ts.conceal_place, "bolsillo")
        self.assertIsNotNone(self.ts.review_event)
        self.assertEqual(self.ts.local_state, "ocultamiento")
        self.assertGreaterEqual(self.ts.score, 0.5)

    def test_pocket_after_window_expires_is_ignored(self):
        self.play(1.0, REACH)
        self.play(self.s.conceal_window + 2)
        self.play(3, POCKET)
        self.assertFalse(self.ts.conceals)

    def test_lost_wrist_next_to_body_counts_as_hidden_hand(self):
        self.play(1.0, REACH)
        end = self.t + 1.5
        while self.t < end:
            k = kp({7: (300, 250)})
            k[9, 2] = 0.0                               # muneca no visible, codo junto a la cadera
            update_track(self.ts, Person(1, BBOX, 0.9, k), self.zones, self.t, self.s)
            self.t += 0.2
        self.assertEqual(len(self.ts.conceals), 1)

    def test_head_scanning_is_counted(self):
        for i in range(6):
            self.play(0.4, nose=(300, 100) if i % 2 == 0 else (340, 100))
        self.assertGreaterEqual(self.ts.head_turns(self.t), 2)

    def test_staff_zone_excludes_person(self):
        self.play(3, bbox=STAFF_BBOX)
        self.assertTrue(self.ts.is_staff)
        self.steal()
        self.assertFalse(self.ts.conceals)
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "ok")


class Decision(Scene, unittest.TestCase):
    def verdict(self, state, conf=0.9):
        self.ts.review_event, self.ts.review_pending = None, False
        self.ts.verdict = {"state": state, "confidence": conf, "reasoning": "x", "action": "oculta_en_ropa"}
        self.ts.verdict_event = self.ts.conceals[-1]

    def test_waits_for_gemini_without_alerting(self):
        self.play(2)
        self.steal()
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "observacion")   # revision pendiente: no se alerta a ciegas

    def test_single_concealment_without_return_prompts_early_review(self):
        self.play(2)
        self.steal()
        self.play(3.2)
        self.assertEqual(decide(self.ts, self.t, self.s)[:2], ("posible_ocultamiento", "local"))

    def test_high_score_not_silenced_by_pending_gemini(self):
        self.steal()
        self.ts.score = self.s.local_possible_score + .1
        self.ts.review_pending = True
        self.assertEqual(decide(self.ts, self.t, self.s)[:2], ("posible_ocultamiento", "local"))

    def test_early_review_requires_observed_concealment(self):
        self.ts.score = 2.0
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "ok")

    def test_gemini_confirms_then_exit_escalates(self):
        self.play(2)
        self.steal()
        self.verdict("ocultamiento")
        self.assertEqual(decide(self.ts, self.t, self.s)[:2], ("posible_ocultamiento", "gemini"))
        self.play(2, bbox=EXIT_BBOX)
        self.assertEqual(self.ts.local_state, "salida_con_ocultamiento")
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "probable_sustraccion")

    def test_gemini_low_confidence_confirmation_is_not_enough(self):
        self.steal()
        self.verdict("ocultamiento", conf=0.4)
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "observacion")

    def test_gemini_clears_the_event(self):
        self.steal()
        self.verdict("normal")
        clear_through(self.ts, self.ts.conceals[-1])
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "ok")
        self.assertEqual(self.ts.local_state, "normal")

    def test_new_concealment_after_clear_is_detected_again(self):
        self.steal()
        self.verdict("normal")
        clear_through(self.ts, self.ts.conceals[-1])
        self.play(self.s.conceal_window + 1)
        self.steal()
        self.assertEqual(len(self.ts.active_conceals()), 1)
        self.assertEqual(self.ts.local_state, "ocultamiento")

    def test_local_fallback_needs_exit_to_reach_probable(self):
        self.steal()
        self.ts.review_event = None                              # Gemini apagado
        self.assertEqual(decide(self.ts, self.t, self.s)[0], "observacion")
        self.play(2, bbox=EXIT_BBOX)
        self.assertEqual(decide(self.ts, self.t, self.s)[:2], ("probable_sustraccion", "local"))

    def test_local_fallback_paid_customer_is_not_probable(self):
        self.play(4, bbox=CAJA_BBOX)
        self.assertIsNotNone(self.ts.paid_at)
        self.steal()
        self.ts.review_event = None
        self.play(2, bbox=EXIT_BBOX)
        self.assertNotEqual(decide(self.ts, self.t, self.s)[0], "probable_sustraccion")

    def test_repeated_concealments_raise_score(self):
        self.steal()
        first = self.ts.score
        self.play(self.s.conceal_window + 1)
        self.steal()
        self.assertGreater(self.ts.score, first + 0.5)


class PerHand(Scene, unittest.TestCase):
    """Una persona parada junto al estante suele tener una mano en la zona y la otra en la cadera."""

    def test_other_hand_staying_in_shelf_does_not_block_concealment(self):
        both = {9: (150, 150), 10: (150, 150)}
        self.play(1.0, both)
        self.play(1.5, {9: (150, 150), 10: (345, 262)})              # derecha al bolsillo, izquierda sigue en el estante
        self.assertEqual(len(self.ts.conceals), 1)

    def test_hands_together_at_waist_after_reach_is_concealment(self):
        self.play(1.0, REACH)
        self.play(1.5, {9: (235, 215), 10: (285, 215)})              # ninguna mano en un lugar de ocultamiento por si sola
        self.assertEqual(self.ts.conceal_place, "manos_a_la_cintura")

    def test_hands_together_without_reach_is_ignored(self):
        self.play(4, {9: (235, 215), 10: (285, 215)})
        self.assertFalse(self.ts.conceals)

    def test_returning_to_the_same_spot_cancels_concealment(self):
        self.play(1.0, REACH)
        self.play(1.5, POCKET)
        self.assertEqual(len(self.ts.active_conceals()), 1)
        self.play(2.0)
        self.play(1.0, REACH)                                        # vuelve al mismo sitio: lo devolvio
        self.assertFalse(self.ts.active_conceals())
        self.assertIsNone(self.ts.review_event)
        self.assertEqual(self.ts.local_state, "normal")

    def test_returning_to_a_different_spot_does_not_cancel(self):
        self.play(1.0, REACH)
        self.play(1.5, POCKET)
        self.play(2.0)
        self.play(1.0, {9: (20, 80)})                                # otro punto del estante
        self.assertEqual(len(self.ts.active_conceals()), 1)

    def test_returning_after_the_window_does_not_cancel(self):
        self.play(1.0, REACH)
        self.play(1.5, POCKET)
        self.play(self.s.return_window + 1)
        self.play(1.0, REACH)
        self.assertEqual(len(self.ts.active_conceals()), 1)

    def test_walking_past_the_shelf_is_not_a_reach(self):
        self.ts.speed = 2.0                                          # ya venia caminando
        for i in range(30):
            bbox = BBOX + np.array([25.0 * i, 0, 25.0 * i, 0])     # ~1,5 anchos de hombro por segundo
            update_track(self.ts, Person(1, bbox, 0.9, kp(REACH)), self.zones, self.t, self.s)
            self.t += 0.2
        self.assertFalse(self.ts.reaches)


class AlertRepeat(unittest.TestCase):
    def test_cleared_sequence_is_resolved_but_still_needs_human_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(Settings.from_env(), data_dir=Path(tmp), shadow_mode=False)
            manager = AlertManager(settings)
            ts = TrackState(1, "C", 0.0, 0.0)
            ts.conceals.append(1.0)
            manager.evaluate(ts, ("posible_ocultamiento", "local", "x"), 2.0, "cam", lambda: None)
            alert_id = manager.recent()[0]["id"]
            self.assertEqual(manager.active_count(), 1)

            ts.events.append((3.0, "devolucion", 0.0))
            ts.cleared_until = 1.0
            manager.evaluate(ts, ("observacion", "local", "devuelto"), 3.0, "cam", lambda: None)
            self.assertEqual(manager.active_count(), 0)
            self.assertEqual(manager.recent()[0]["resolved"], "devolucion")
            self.assertNotIn("review", manager.recent()[0])
            self.assertEqual(AlertManager(settings).recent()[0]["resolved"], "devolucion")
            self.assertTrue(manager.review(alert_id, "descartado", "operador", "panel"))

    def test_gemini_clear_is_not_mislabeled_as_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = AlertManager(replace(Settings.from_env(), data_dir=Path(tmp), shadow_mode=False))
            ts = TrackState(1, "C", 0.0, 0.0)
            ts.conceals.append(1.0)
            manager.evaluate(ts, ("posible_ocultamiento", "local", "x"), 2.0, "cam", lambda: None)
            ts.cleared_until = 1.0
            manager.evaluate(ts, ("ok", "", ""), 3.0, "cam", lambda: None)
            self.assertEqual(manager.recent()[0]["resolved"], "revision_visual")

    def test_repeats_respect_cooldown_but_escalation_is_immediate(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = AlertManager(replace(Settings.from_env(), data_dir=Path(tmp), shadow_mode=False))
            ts = TrackState(1, "C", 0.0, 0.0)
            possible = ("posible_ocultamiento", "local", "x")
            ts.conceals.append(1.0)
            manager.evaluate(ts, possible, 1.0, "cam", lambda: None)
            ts.conceals.append(5.0)
            manager.evaluate(ts, possible, 5.0, "cam", lambda: None)       # dentro del periodo: no repite
            self.assertEqual(len(manager.recent()), 1)
            manager.evaluate(ts, possible, 40.0, "cam", lambda: None)      # pasado el periodo: repite
            self.assertEqual(len(manager.recent()), 2)
            manager.evaluate(ts, ("probable_sustraccion", "local", "x"), 41.0, "cam", lambda: None)
            self.assertEqual(len(manager.recent()), 3)                     # escalar siempre avisa


class AlertDismiss(unittest.TestCase):
    def make(self, tmp):
        return AlertManager(replace(Settings.from_env(), data_dir=Path(tmp), shadow_mode=False))

    def emit_two(self, manager):
        for i, t in enumerate((1.0, 50.0)):
            ts = TrackState(i, "C", 0.0, 0.0)
            ts.conceals.append(t)
            manager.evaluate(ts, ("posible_ocultamiento", "local", "x"), t, "cam", lambda: None)

    def test_dismiss_one_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make(tmp)
            self.emit_two(manager)
            first, second = (a["id"] for a in manager.recent()[::-1])
            self.assertTrue(manager.dismiss(first))
            self.assertFalse(manager.dismiss(first))
            self.assertEqual([a["id"] for a in manager.recent()], [second])
            self.assertEqual(manager.active_count(), 1)
            self.assertEqual([a["id"] for a in self.make(tmp).recent()], [second])      # tras reiniciar

    def test_dismiss_all_survives_restart_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make(tmp)
            self.emit_two(manager)
            self.assertEqual(manager.dismiss_all(), 2)
            self.assertEqual(manager.dismiss_all(), 0)
            self.assertEqual(self.make(tmp).recent(), [])
            self.emit_two(manager)                                                       # alertas nuevas si aparecen
            self.assertEqual(len(self.make(tmp).recent()), 2)
            self.assertGreaterEqual(len(manager.log_file.read_text(encoding="utf-8").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
