"""Decision final: combina la heuristica local y el dictamen de Gemini.

Niveles: ok < observacion < posible_ocultamiento < probable_sustraccion.
Solo los dos ultimos generan alerta; `observacion` es informativo en el panel.

Reglas, en orden (sobre los ocultamientos que Gemini no ha descartado):
 1. Sin ocultamientos activos: observacion si hay comportamiento inusual, si no ok.
 2. Dictamen de Gemini que confirma ocultamiento (conf >= MIN_CONF_CONFIRM):
      posible_ocultamiento, o probable_sustraccion si la persona ya salio.
    (Si Gemini lo descarta, el pipeline anula el evento y no llega aqui.)
 3. Revision de Gemini en curso: se mantiene el nivel actual, sin escalar a ciegas.
 4. Respaldo local (Gemini apagado, sin cuota, indeterminado o con error): la puntuacion
    debe superar un umbral mas alto. Haber pasado por caja evita el nivel "probable"
    sin confirmacion, porque la pose sola no distingue lo que se pago.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .config import Settings
from .heuristics import TrackState, clear_through

LEVEL_RANK = {"ok": 0, "observacion": 1, "posible_ocultamiento": 2, "probable_sustraccion": 3}
ALERT_MIN_RANK = LEVEL_RANK["posible_ocultamiento"]
PLACE_TEXT = {"bolsillo": "el bolsillo", "pecho_o_abdomen": "el pecho o abdomen", "mano_oculta": "el cuerpo (mano oculta)"}


def decide(ts: TrackState, now: float, s: Settings) -> Tuple[str, str, str]:
    """Devuelve (nivel, fuente, motivo)."""
    if ts.is_staff:
        return "ok", "", ""

    active = ts.active_conceals()
    if not active:
        if ts.local_state == "observacion":
            return "observacion", "local", "Comportamiento inusual cerca de la estanteria"
        return "ok", "", ""

    exited = ts.local_state == "salida_con_ocultamiento"
    verdict = ts.verdict
    if (verdict is not None and verdict["state"] == "ocultamiento" and verdict["confidence"] >= s.min_conf_confirm
            and ts.verdict_event in active):
        if exited:
            return "probable_sustraccion", "gemini", f"{verdict['reasoning']} Salio sin devolver el producto."
        return "posible_ocultamiento", "gemini", verdict["reasoning"]

    if ts.review_pending or ts.review_event is not None:
        held = ts.level if ts.level in LEVEL_RANK else "ok"
        return (held, ts.level_source, ts.level_reason) if LEVEL_RANK[held] > 1 \
            else ("observacion", "local", "Verificando posible ocultamiento")

    place = PLACE_TEXT.get(ts.conceal_place, "el cuerpo")
    if exited and ts.paid_at is None and ts.score >= s.local_probable_score:
        return "probable_sustraccion", "local", f"Tomo del estante, llevo la mano a {place} y salio sin pasar por caja"
    if ts.score >= s.local_possible_score:
        times = "1 vez" if len(active) == 1 else f"{len(active)} veces"
        return "posible_ocultamiento", "local", f"Tomo del estante y llevo la mano a {place} ({times})"
    return "observacion", "local", "Posible ocultamiento sin confirmar"


def apply_verdict(ts: TrackState, event: float, verdict: Dict[str, Any], now: float, s: Settings) -> None:
    """Registra el dictamen de Gemini sobre un ocultamiento; si lo descarta con confianza, se anula el evento."""
    ts.verdict, ts.verdict_time, ts.verdict_event = verdict, now, event
    if verdict["state"] == "normal" and verdict["confidence"] >= s.min_conf_clear:
        clear_through(ts, event)
