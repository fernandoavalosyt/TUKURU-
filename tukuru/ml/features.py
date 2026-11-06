"""Caracteristicas congeladas de alertas y muestras locales de interaccion."""

from __future__ import annotations

from typing import Any

from tukuru.heuristics import TrackState
from tukuru.vision import L_WRIST, R_WRIST

FEATURE_VERSION = 1


def incident_features(track: TrackState, now: float) -> dict[str, Any]:
    active = track.active_conceals()
    last_reach = track.reaches[-1] if track.reaches else None
    wrists = track.kpts[[L_WRIST, R_WRIST], 2]
    return {
        "version": FEATURE_VERSION,
        "score": round(float(track.score), 4),
        "seg_visible": round(max(0.0, now - track.first_seen), 2),
        "alcances_2min": sum(now - t <= 120 for t in track.reaches),
        "ocultamientos_activos": len(active),
        "seg_desde_ultimo_alcance": round(now - last_reach, 2) if last_reach is not None else None,
        "giros_cabeza": track.head_turns(now),
        "velocidad_cuerpo": round(float(track.speed), 3),
        "conf_muneca_min": round(float(min(wrists)), 3),
        "conf_muneca_max": round(float(max(wrists)), 3),
        "paso_por_caja": track.paid_at is not None,
        "salio": track.exit_time is not None,
        "lugar_ocultamiento": track.conceal_place or "desconocido",
    }
