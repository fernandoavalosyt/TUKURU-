"""Continuidad optica limitada a munecas ya detectadas por pose."""
import unittest

import cv2
import numpy as np

from tukuru.heuristics import TrackState
from tukuru.motion import WristMotionTracker
from tukuru.vision import L_WRIST, Person
from tukuru.zones import ZoneSet


def textured_frame():
    rng = np.random.default_rng(4)
    return rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)


def pose(track_id=1, wrist=(60, 60), confidence=.9):
    points = np.zeros((17, 3), dtype=np.float32)
    points[L_WRIST] = [*wrist, confidence]
    return Person(track_id, np.array([30., 20., 90., 110.]), .9, points)


class WristMotionTests(unittest.TestCase):
    def setUp(self):
        self.zones = ZoneSet({"estanteria": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}, 120, 120)
        self.ts = TrackState(1, "test", 0, 0)
        self.ts.hands[0].reach_start = 0
        self.ts.hands[0].reach_last = .3
        self.ts.hands[0].checked = True
        self.ts.hands[0].armed_until = 5
        self.tracker = WristMotionTracker()
        self.first = textured_frame()
        self.tracker.update(self.first, [pose()], {1: self.ts}, self.zones, .3)

    def test_tracks_known_wrist_and_only_extends_observed_reach(self):
        matrix = np.float32([[1, 0, 4], [0, 1, 2]])
        moved = cv2.warpAffine(self.first, matrix, (120, 120))
        points = self.tracker.update(moved, [], {1: self.ts}, self.zones, .5)
        np.testing.assert_allclose(points[(1, 0)], [64, 62], atol=1.5)
        self.assertEqual(self.ts.hands[0].reach_last, .5)
        self.assertEqual(list(self.ts.conceals), [])
        self.assertEqual(list(self.ts.events), [])

    def test_flow_never_seeds_a_hand_without_a_prior_pose_point(self):
        empty = TrackState(2, "test", 0, 0)
        translated = cv2.warpAffine(self.first, np.float32([[1, 0, 3], [0, 1, 2]]), (120, 120))
        points = self.tracker.update(translated, [], {2: empty}, self.zones, .5)
        self.assertNotIn((2, 0), points)

    def test_low_texture_drops_motion_instead_of_guessing(self):
        flat = np.zeros_like(self.first)
        self.tracker.update(flat, [], {1: self.ts}, self.zones, .5)
        self.assertIsNone(self.ts.flow_wrist[0])
        self.assertEqual(self.ts.hands[0].reach_last, .3)

    def test_real_wrist_reanchors_and_clears_flow_marker(self):
        moved = cv2.warpAffine(self.first, np.float32([[1, 0, 4], [0, 1, 2]]), (120, 120))
        self.tracker.update(moved, [], {1: self.ts}, self.zones, .5)
        self.assertIsNotNone(self.ts.flow_wrist[0])
        self.tracker.update(moved, [pose(wrist=(64, 62))], {1: self.ts}, self.zones, .7)
        self.assertIsNone(self.ts.flow_wrist[0])


if __name__ == "__main__":
    unittest.main()
