"""Exporta revisiones humanas y caracteristicas versionadas para analisis local."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from .features import FEATURE_VERSION
from .repository import IncidentRepository


def labeled_rows(database: Path) -> list[dict]:
    """Una fila por alerta, con la revision mas reciente y esquema actual."""
    IncidentRepository(database)  # aplica migraciones locales antes de consultar
    with sqlite3.connect(database) as db:
        rows = db.execute("""
            SELECT i.id, i.camera_id, i.recording_id, i.track_id, i.occurred_at, i.features_json, r.label
            FROM incidents AS i
            JOIN reviews AS r ON r.id = (
                SELECT r2.id FROM reviews AS r2 WHERE r2.incident_id = i.id
                ORDER BY r2.id DESC LIMIT 1
            )
            WHERE i.feature_version = ? AND r.label IN ('confirmado', 'descartado')
                AND i.recording_id IS NOT NULL AND i.recording_id != ''
            ORDER BY i.occurred_at, i.id
        """, (FEATURE_VERSION,)).fetchall()
    return [{"incident_id": incident_id, "camera_id": camera_id, "session_id": recording_id,
             "track_id": track_id, "occurred_at": occurred_at,
             "label": 1 if label == "confirmado" else 0, **json.loads(features)}
             for incident_id, camera_id, recording_id, track_id, occurred_at, features, label in rows]


def group_cases(rows: list[dict], window_seconds: float = 120.0) -> list[list[dict]]:
    """Agrupa alertas/muestras del mismo track separadas hasta 120 s."""
    ordered = sorted(rows, key=lambda row: (row["session_id"], row["camera_id"], row["track_id"],
                                            row["occurred_at"], str(row.get("incident_id", row.get("id", "")))))
    grouped: dict[tuple, list[list[dict]]] = {}
    for row in ordered:
        key = (row["session_id"], row["camera_id"], row["track_id"])
        episodes = grouped.setdefault(key, [])
        if episodes and row["occurred_at"] - episodes[-1][0]["occurred_at"] <= window_seconds:
            episodes[-1].append(row)
        else:
            episodes.append([row])
    return [episode for episodes in grouped.values() for episode in episodes]


def deduplicate_cases(rows: list[dict], window_seconds: float = 120.0) -> list[dict]:
    """Conserva el caso mas reciente por episodio; el origen sigue agrupado para el split."""
    return [episode[-1] for episode in group_cases(rows, window_seconds)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[2] / "data/incidents.sqlite3")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data/dataset.csv")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"No existe {args.db}")
    rows = labeled_rows(args.db)
    if not rows:
        print("Todavia no hay incidentes revisados con caracteristicas actuales; no se exporto un dataset.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exportados {len(rows)} incidentes revisados a {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
