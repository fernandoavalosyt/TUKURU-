import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from tukuru.alerts import AlertManager
from tukuru.config import ConfigError, Settings, validate_camera
from tukuru.decision import decide
from tukuru.heuristics import TrackState, clear_through, update_track
from tukuru.vision import Person
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

    def test_probable_score_must_exceed_possible(self):
        os.environ.update(LOCAL_POSSIBLE_SCORE="0.9", LOCAL_PROBABLE_SCORE="0.8")
        try:
            with self.assertRaises(ConfigError):
                Settings.from_env()
        finally:
            for k in ("LOCAL_POSSIBLE_SCORE", "LOCAL_PROBABLE_SCORE"):
                os.environ.pop(k, None)


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


if __name__ == "__main__":
    unittest.main()
