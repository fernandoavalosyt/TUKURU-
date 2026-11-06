"""Prioriza casos poco parecidos a ejemplos descartados por operadores.

No entrena ni persiste un modelo. Es una señal exploratoria de revision, nunca una
clasificacion de robo; se apaga si faltan ejemplos o variedad de origenes.
"""

from __future__ import annotations

import numpy as np

from .dataset import deduplicate_cases
from .model import feature_vector

MIN_NORMAL_EPISODES = 30
MIN_ORIGIN_GROUPS = 5
NEIGHBORS = 5


def reference_readiness(rows: list[dict]) -> tuple[list[dict], str | None]:
    normal = deduplicate_cases([row for row in rows if row.get("label") == 0])
    origins = {row.get("session_id") for row in normal if row.get("session_id")}
    if len(normal) < MIN_NORMAL_EPISODES:
        return normal, f"se requieren {MIN_NORMAL_EPISODES} episodios descartados; hay {len(normal)}"
    if len(origins) < MIN_ORIGIN_GROUPS:
        return normal, f"se requieren {MIN_ORIGIN_GROUPS} grupos de origen; hay {len(origins)}"
    return normal, None


def novelty_percentiles(cases: list[dict], reviewed_rows: list[dict]) -> dict[str, float]:
    """Percentil de distancia frente a negativos revisados, con calibracion leave-one-out."""
    reference, reason = reference_readiness(reviewed_rows)
    if reason or not cases:
        return {}
    try:
        matrix = np.stack([feature_vector(row) for row in reference])
        candidate_matrix = np.stack([feature_vector(case["features"]) for case in cases])
    except (ValueError, TypeError, KeyError):
        return {}
    if not np.isfinite(matrix).all() or not np.isfinite(candidate_matrix).all():
        return {}

    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / scale
    candidates = (candidate_matrix - matrix.mean(axis=0)) / scale
    k = min(NEIGHBORS, len(normalized) - 1)
    reference_distances = []
    for index, vector in enumerate(normalized):
        distances = np.linalg.norm(normalized - vector, axis=1)
        distances[index] = np.inf
        reference_distances.append(float(np.partition(distances, k - 1)[:k].mean()))
    reference_distances = np.asarray(reference_distances)

    scores: dict[str, float] = {}
    for case, vector in zip(cases, candidates):
        distances = np.linalg.norm(normalized - vector, axis=1)
        nearest = float(np.partition(distances, k - 1)[:k].mean())
        percentile = 100.0 * float(np.count_nonzero(reference_distances <= nearest)) / len(reference_distances)
        scores[str(case["id"])] = round(percentile, 1)
    return scores
