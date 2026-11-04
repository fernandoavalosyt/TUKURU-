"""Relaciones sinteticas: sin red, pesos ni Gemini."""
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np

from tukuru.heuristics import TrackState
from tukuru.objects import Detection, ObjectExperiment, Session, SmallDetector
from tukuru.vision import L_WRIST, R_WRIST
from tukuru.zones import ZoneSet


def person(key=1, x=140):
    ts = TrackState(key, "test", 0, 0)
    ts.bbox = np.array([100., 50., 240., 300.])
    ts.kpts[[5, 6, 11, 12]] = [[130, 110, .9], [210, 110, .9], [140, 240, .9], [200, 240, .9]]
    ts.kpts[L_WRIST] = [x, 190, .9]
    return ts


def item(x=140, label="bottle"):
    return Detection(label, .9, np.array([x - 5., 180., x + 5., 200.]))


class ObjectRelations(unittest.TestCase):
    def setUp(self):
        self.exp = ObjectExperiment(Mock())
        self.ts = person()
        self.session = Session()
        self.zones = ZoneSet({"estanteria": [[[0, 0], [.3, 0], [.3, 1], [0, 1]]]}, 400, 400)

    def observe(self, detections, now, visible=None):
        self.exp.observe(self.ts, self.session, detections, visible or [self.ts], self.zones, now)

    def confirm(self, detections, start=0):
        for now in (start, start + .5, start + 1):
            self.observe(detections, now)

    def test_persistence_required(self):
        self.observe([item()], 0)
        self.assertFalse(self.exp.events)
        self.observe([], 1)
        self.assertNotEqual(self.session.hands[0].state, "objeto_junto_a_mano")

    def test_visible_object_then_missing_is_unknown(self):
        self.confirm([item()])
        self.assertEqual(self.session.hands[0].state, "objeto_junto_a_mano")
        self.observe([], 2)
        self.assertEqual(self.session.hands[0].state, "desconocido")

    def test_shelf_destination_requires_visible_object(self):
        self.confirm([item()])
        self.ts.kpts[L_WRIST, 0] = 100
        self.observe([item(100)], 1.5)
        self.observe([item(100)], 2)
        self.assertEqual(self.session.hands[0].state, "objeto_en_estanteria")
        self.assertEqual(self.exp.events[-1]["from"], "objeto_junto_a_mano")

    def test_bag_proximity_does_not_claim_insertion(self):
        bag = Detection("handbag", .9, np.array([130., 180., 180., 230.]))
        self.confirm([item(), bag])
        self.assertEqual(self.session.hands[0].state, "objeto_junto_a_contenedor")
        self.observe([bag], 2)
        self.assertEqual(self.session.hands[0].state, "desconocido")

    def test_bag_alone_is_not_a_held_product(self):
        self.confirm([item(label="handbag")])
        self.assertFalse(self.exp.events)

    def test_other_person_nearer_owns_detection(self):
        other = person(2, 140)
        self.ts.kpts[L_WRIST, 0] = 170
        for now in (0, .5, 1):
            self.observe([item()], now, [self.ts, other])
        self.assertFalse(self.exp.events)

    def test_two_close_hands_are_ambiguous(self):
        self.ts.kpts[R_WRIST] = self.ts.kpts[L_WRIST]
        self.confirm([item()])
        self.assertFalse(self.exp.events)

    def test_large_jump_is_not_same_object(self):
        self.confirm([item()])
        self.ts.kpts[L_WRIST, 0] = 300
        self.observe([item(300)], 1.5)
        self.assertEqual(self.session.hands[0].last_seen, 1)

    def test_class_change_does_not_inherit_relation(self):
        self.confirm([item()])
        self.observe([item(label="cell phone")], 1.5)
        self.assertEqual(self.session.hands[0].label, "bottle")

    def test_inference_is_gated_limited_and_expires(self):
        detector = Mock()
        detector.detect.return_value = []
        exp = ObjectExperiment(detector)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        exp.step(frame, 1, [self.ts], self.zones, 0)
        detector.detect.assert_not_called()
        hand = self.ts.hands[0]
        hand.checked, hand.reach_start, hand.armed_until = True, 0, 8
        for now in np.arange(.3, 10, .1):
            self.ts.last_seen = float(now)
            exp.step(frame, 1, [self.ts], self.zones, float(now))
        self.assertLessEqual(detector.detect.call_count, 17)
        self.assertGreater(detector.detect.call_count, 0)
        self.assertTrue(all(s["t"] <= 8.3 for s in exp.samples))
        self.assertEqual(self.ts.score, 0)
        self.assertEqual(self.ts.level, "ok")

    def test_global_budget_is_shared_between_people(self):
        detector = Mock()
        detector.detect.return_value = []
        exp = ObjectExperiment(detector)
        people = [person(key) for key in (1, 2, 3)]
        for ts in people:
            h = ts.hands[0]
            h.checked, h.reach_start, h.armed_until = True, 0, 8
        for now in np.arange(.3, 3, .1):
            for ts in people:
                ts.last_seen = float(now)
            exp.step(np.zeros((400, 400, 3)), 1, people, self.zones, float(now))
        self.assertEqual({s["id"] for s in exp.samples}, {1, 2, 3})
        times = [s["t"] for s in exp.samples]
        self.assertTrue(all(b - a >= .249 for a, b in zip(times, times[1:])))

    def test_crop_coordinates_map_back_to_small_frame(self):
        detector = SmallDetector.__new__(SmallDetector)
        detector.device = "cpu"
        # Simula una caja en el recorte original. El padding es .18 * 250 / .5 = 90.
        def tensor(data):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(data)))
        boxes = SimpleNamespace(xyxy=tensor([[10, 20, 30, 40]]), cls=tensor([0]), conf=tensor([.9]))
        detector.model = Mock()
        detector.model.predict.return_value = [SimpleNamespace(boxes=boxes, names={0: "bottle"})]
        found = detector.detect(np.zeros((800, 800, 3), dtype=np.uint8), .5, self.ts)
        # bbox original arranca en (200,100), recorte con padding arranca en (110,10).
        np.testing.assert_allclose(found[0].box, [60, 15, 70, 25])
        self.assertEqual(detector.model.predict.call_args.kwargs["imgsz"], 416)


if __name__ == "__main__":
    unittest.main()
