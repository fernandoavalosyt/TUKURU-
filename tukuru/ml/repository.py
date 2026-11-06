"""Persistencia SQLite de incidentes y revisiones humanas."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any


class IncidentRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    occurred_at REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    feature_version INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    snapshot TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT NOT NULL REFERENCES incidents(id),
                    reviewer_source TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    label TEXT NOT NULL CHECK(label IN ('confirmado','descartado','no_concluyente')),
                    reviewed_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reviews_incident_idx ON reviews(incident_id, reviewed_at);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(incidents)")}
            if "session_id" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'")
            if "recording_id" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN recording_id TEXT")
            if "record_type" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN record_type TEXT NOT NULL DEFAULT 'alerta'")
            if "ml_shadow_score" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN ml_shadow_score REAL")
                db.execute("ALTER TABLE incidents ADD COLUMN ml_model_version TEXT")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def record_incident(self, alert: dict[str, Any]) -> None:
        features = alert.get("features") or {"version": 0}
        with closing(self._connect()) as db, db:
            db.execute("""INSERT OR IGNORE INTO incidents
                (id, occurred_at, camera_id, session_id, track_id, level, source, reason,
                 feature_version, features_json, snapshot, ml_shadow_score, ml_model_version, recording_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (alert["id"], alert["time"], alert["camera_id"], alert.get("session_id", "legacy"), alert["track_id"],
                 alert["level"], alert["source"], alert["reason"], features["version"],
                 json.dumps(features, ensure_ascii=False), alert.get("snapshot"),
                 alert.get("ml_shadow_score"), alert.get("ml_model_version"), alert.get("recording_id")))

    def record_observation(self, sample: dict[str, Any], *, camera_hour_limit: int = 20,
                           track_interval_seconds: float = 30.0) -> bool:
        """Guarda una muestra local de interaccion sin alertar; limita almacenamiento por camara/track."""
        features = sample.get("features") or {"version": 0}
        timestamp = float(sample["time"])
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("""SELECT COUNT(*) FROM incidents WHERE record_type='observacion'
                AND camera_id=? AND occurred_at>=?""", (sample["camera_id"], timestamp - 3600)).fetchone()[0]
            recent = db.execute("""SELECT MAX(occurred_at) FROM incidents WHERE record_type='observacion'
                AND camera_id=? AND track_id=? AND reason=?""",
                (sample["camera_id"], sample["track_id"], sample["reason"])).fetchone()[0]
            if count >= camera_hour_limit or (recent is not None and timestamp - recent < track_interval_seconds):
                db.rollback()
                return False
            db.execute("""INSERT INTO incidents
                (id,occurred_at,camera_id,session_id,track_id,level,source,reason,feature_version,
                 features_json,snapshot,ml_shadow_score,ml_model_version,recording_id,record_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'observacion')""",
                (sample["id"], timestamp, sample["camera_id"], sample.get("session_id", "legacy"),
                 sample["track_id"], "observacion", "muestreo_local", sample["reason"], features["version"],
                 json.dumps(features, ensure_ascii=False), None, sample.get("ml_shadow_score"),
                 sample.get("ml_model_version"), sample.get("recording_id")))
            db.commit()
            return True

    def update_observation_snapshot(self, incident_id: str, snapshot: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE incidents SET snapshot=? WHERE id=? AND record_type='observacion'",
                       (snapshot, incident_id))

    def latest_review(self, incident_id: str) -> tuple[str, str, str] | None:
        with closing(self._connect()) as db:
            row = db.execute("""SELECT label, reviewer_source, reviewer_id FROM reviews
                                WHERE incident_id = ? ORDER BY id DESC LIMIT 1""", (incident_id,)).fetchone()
        return tuple(row) if row else None

    def review_summary(self, feature_version: int) -> dict[str, int]:
        """Cuenta la ultima etiqueta por incidente del esquema vigente."""
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT i.session_id, r.label FROM incidents i
                LEFT JOIN reviews r ON r.id = (
                    SELECT MAX(r2.id) FROM reviews r2 WHERE r2.incident_id = i.id)
                WHERE i.feature_version = ?""", (feature_version,)).fetchall()
        counts = Counter(label for _, label in rows if label)
        return {"confirmado": counts["confirmado"], "descartado": counts["descartado"],
                "no_concluyente": counts["no_concluyente"],
                "sesiones_revisadas": len({session for session, label in rows if label})}

    def record_review(self, incident_id: str, label: str, reviewer_source: str, reviewer_id: str) -> bool:
        with closing(self._connect()) as db, db:
            cursor = db.execute("""INSERT INTO reviews (incident_id, reviewer_source, reviewer_id, label, reviewed_at)
                SELECT id, ?, ?, ?, ? FROM incidents WHERE id = ?""",
                (reviewer_source, reviewer_id, label, time.time(), incident_id))
            return cursor.rowcount == 1

    def quality_summary(self, feature_version: int) -> dict[str, Any]:
        """Reporte agregado sin revelar IDs, rutas, capturas ni credenciales."""
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT i.recording_id,i.session_id,r.label FROM incidents i
                LEFT JOIN reviews r ON r.id=(SELECT MAX(r2.id) FROM reviews r2 WHERE r2.incident_id=i.id)
                WHERE i.feature_version=?""", (feature_version,)).fetchall()
            observations = db.execute("""SELECT COUNT(*),SUM(CASE WHEN r.label IS NOT NULL THEN 1 ELSE 0 END)
                FROM incidents i LEFT JOIN reviews r ON r.id=(SELECT MAX(r2.id) FROM reviews r2
                    WHERE r2.incident_id=i.id) WHERE i.feature_version=? AND i.record_type='observacion'""",
                (feature_version,)).fetchone()
            events = db.execute("""SELECT recording_id,session_id,camera_id,track_id,occurred_at
                FROM incidents WHERE feature_version=? ORDER BY occurred_at DESC""", (feature_version,)).fetchall()
        labels = Counter(label for _, _, label in rows if label)
        groups: dict[tuple, list[float]] = {}
        for origin, session, camera, track, occurred in events:
            source = ("origin", origin) if origin else ("session", session)
            groups.setdefault((source, camera, track), []).append(occurred)
        clusters = 0
        for times in groups.values():
            cluster_start = None
            for occurred in times:
                if cluster_start is None or cluster_start - occurred > 120:
                    clusters += 1
                    cluster_start = occurred
        return {"incidentes": len(rows), "etiquetados": sum(labels.values()),
                "confirmado": labels["confirmado"], "descartado": labels["descartado"],
                "no_concluyente": labels["no_concluyente"],
                "origen_desconocido": sum(1 for origin, _, _ in rows if not origin),
                "sesiones_de_proceso": len({session for _, session, _ in rows}),
                "grupos_origen_declarados": len({origin for origin, _, _ in rows if origin}),
                "alertas_agrupables_en_revision": len(events)-clusters,
                "muestras_locales": observations[0] or 0,
                "muestras_locales_revisadas": observations[1] or 0,
                "filas_entrenables": sum(1 for origin, _, label in rows
                                         if origin and label in ("confirmado", "descartado"))}

    def review_candidates(self, limit: int = 200, since: float = 0.0) -> list[dict[str, Any]]:
        """Pendientes; fusiona visualmente repeticiones del mismo track separadas <=120 s."""
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT i.id,i.occurred_at,i.camera_id,i.track_id,i.level,i.reason,i.snapshot,
                    i.recording_id,i.session_id,r.label,i.ml_shadow_score,i.record_type,i.features_json FROM incidents i
                LEFT JOIN reviews r ON r.id=(SELECT MAX(r2.id) FROM reviews r2 WHERE r2.incident_id=i.id)
                WHERE i.occurred_at>=? ORDER BY i.occurred_at DESC LIMIT 5000""", (since,)).fetchall()
        pending = [r for r in rows if r[9] is None]
        groups: dict[tuple, list[list[tuple]]] = {}
        for row in pending:
            source = ("origin", row[7]) if row[7] else ("session", row[8])
            key = (source, row[2], row[3])
            clusters = groups.setdefault(key, [])
            if clusters and clusters[-1][-1][1] - row[1] <= 120:
                clusters[-1].append(row)
            else:
                clusters.append([row])
        representatives = [cluster for clusters in groups.values() for cluster in clusters]
        representatives.sort(key=lambda cluster: (cluster[0][10] is None,
            abs((cluster[0][10] or 0.0)-0.5) if cluster[0][10] is not None else 1.0,
            -cluster[0][1]))
        return [{"id": cluster[0][0], "incident_ids": [row[0] for row in cluster], "count": len(cluster),
                 "time": cluster[0][1], "camera_id": cluster[0][2], "track_id": cluster[0][3],
                 "level": cluster[0][4], "reason": cluster[0][5], "snapshot": cluster[0][6],
                 "recording_known": bool(cluster[0][7]), "shadow_score": cluster[0][10],
                 "record_type": (cluster[0][11] if len({row[11] for row in cluster}) == 1 else "mixto"),
                 "features": json.loads(cluster[0][12])}
                for cluster in representatives[:limit]]

    def assign_recording(self, incident_ids: list[str], recording_id: str) -> int:
        """Asigna origen solo cuando un operador identifica la grabacion de cada caso."""
        if not incident_ids:
            return 0
        with closing(self._connect()) as db, db:
            cursor = db.executemany("UPDATE incidents SET recording_id=? WHERE id=? AND (recording_id IS NULL OR recording_id='')",
                                    [(recording_id, item) for item in set(incident_ids)])
            return cursor.rowcount
