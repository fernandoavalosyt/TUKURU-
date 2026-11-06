"""Continuidad optica de munecas cuando la pose pierde temporalmente a la persona.

El flujo conserva una trayectoria corta ya asociada a una mano detectada por YOLO-pose.
No detecta una mano nueva ni aporta evidencia de producto u ocultamiento por si solo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .heuristics import TrackState
from .vision import KP_CONF, L_WRIST, R_WRIST, Person
from .zones import ZoneSet

MAX_FRAME_GAP = 0.75
MIN_TRACKED_FEATURES = 5
MAX_FORWARD_BACK_ERROR = 2.0
MAX_FLOW_SPREAD = 3.0
MAX_POINT_STEP = 70.0
FLOW_CONFIDENCE = KP_CONF + 0.01
HAND_KEYPOINTS = (L_WRIST, R_WRIST)


@dataclass
class _FlowState:
    points: np.ndarray
    center: np.ndarray
    updated: float


class WristMotionTracker:
    """Sigue munecas conocidas con Lucas-Kanade y validacion forward-backward."""

    def __init__(self) -> None:
        self._previous: Optional[np.ndarray] = None
        self._states: Dict[Tuple[int, int], _FlowState] = {}

    def reset(self) -> None:
        self._previous = None
        self._states.clear()

    @staticmethod
    def _seed(point: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
        x, y = map(float, point)
        if x < 5 or y < 5 or x >= width - 5 or y >= height - 5:
            return None
        offsets = np.array([[-3, -3], [0, -3], [3, -3], [-3, 0], [0, 0], [3, 0], [-3, 3], [0, 3], [3, 3]],
                           dtype=np.float32)
        return np.ascontiguousarray((offsets + [x, y]).reshape(-1, 1, 2), dtype=np.float32)

    def _advance(self, old_gray: np.ndarray, gray: np.ndarray, state: _FlowState,
                 now: float) -> Optional[_FlowState]:
        if now - state.updated > MAX_FRAME_GAP:
            return None
        try:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, state.points, None, winSize=(21, 21), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03), minEigThreshold=1e-4)
            if next_points is None or status is None:
                return None
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, old_gray, next_points, None, winSize=(21, 21), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03), minEigThreshold=1e-4)
        except cv2.error:
            return None
        if back_points is None or back_status is None:
            return None
        p0, p1, pback = state.points[:, 0], next_points[:, 0], back_points[:, 0]
        valid = (status[:, 0] > 0) & (back_status[:, 0] > 0)
        valid &= np.isfinite(p1).all(axis=1) & np.isfinite(pback).all(axis=1)
        fb = np.linalg.norm(p0 - pback, axis=1)
        valid &= fb <= MAX_FORWARD_BACK_ERROR
        if int(valid.sum()) < MIN_TRACKED_FEATURES:
            return None
        deltas = p1[valid] - p0[valid]
        delta = np.median(deltas, axis=0)
        spread = np.linalg.norm(deltas - delta, axis=1)
        if float(np.median(spread)) > MAX_FLOW_SPREAD or float(np.linalg.norm(delta)) > MAX_POINT_STEP:
            return None
        center = state.center + delta
        tracked = p1[valid].reshape(-1, 1, 2).astype(np.float32)
        return _FlowState(tracked, center.astype(np.float32), now)

    def update(self, frame: np.ndarray, people: list[Person], tracks: Dict[int, TrackState], zones: ZoneSet,
               now: float) -> Dict[Tuple[int, int], np.ndarray]:
        """Rebasa con munecas reales y propaga solo manos con alcance en curso.

        La posicion optica se dibuja como diagnostico. Si queda dentro del estante, solo
        extiende el ultimo instante observado del alcance; no crea eventos ni puntuacion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape[:2]
        if self._previous is not None and self._previous.shape != gray.shape:
            self.reset()
        people_by_id = {person.track_id: person for person in people}
        active: Dict[Tuple[int, int], np.ndarray] = {}
        for ts in tracks.values():
            ts.flow_wrist = [None, None]
            ts.flow_wrist_time = 0.0
        next_states: Dict[Tuple[int, int], _FlowState] = {}

        for track_id, ts in tracks.items():
            person = people_by_id.get(track_id)
            for hand_index, keypoint in enumerate(HAND_KEYPOINTS):
                key = (track_id, hand_index)
                hand = ts.hands[hand_index]
                observed = person is not None and person.kpts[keypoint, 2] >= KP_CONF
                if observed:
                    pos = person.kpts[keypoint, :2].astype(np.float32).copy()
                    points = self._seed(pos, width, height)
                    if points is not None:
                        next_states[key] = _FlowState(points, pos, now)
                    continue

                # Mantiene la pista exclusivamente si ya habia un alcance observado.
                still_relevant = hand.reach_start is not None and (
                    now <= hand.armed_until or now - hand.reach_last <= MAX_FRAME_GAP)
                previous_state = self._states.get(key)
                if not still_relevant or previous_state is None or self._previous is None:
                    continue
                advanced = self._advance(self._previous, gray, previous_state, now)
                if advanced is None:
                    continue
                next_states[key] = advanced
                active[key] = advanced.center.copy()
                ts.flow_wrist[hand_index] = advanced.center.copy()
                ts.flow_wrist_time = now
                if zones.contains("estanteria", advanced.center):
                    hand.reach_last = now

        self._states = next_states
        self._previous = gray.copy()
        return active
