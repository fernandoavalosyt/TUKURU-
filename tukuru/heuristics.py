"""Heuristica local de robo hormiga a partir de los puntos del cuerpo y las zonas.

A diferencia de la somnolencia (un estado lento y sostenido), la sustraccion es una
SECUENCIA corta por persona, que se sigue MANO POR MANO (una persona parada junto al
estante suele tener una mano en la zona y la otra en la cadera):

    alcance al estante  ->  ocultamiento (mano a bolsillo/bolso/pecho, o manos juntas a la cintura)
                        ->  salida sin pasar por caja

Si la misma mano vuelve al mismo sitio del estante poco despues, se interpreta como
"tomo y lo devolvio" y el ocultamiento se anula.

Cada etapa se detecta con senales baratas (munecas, caderas, zonas) y suma peso a una
puntuacion con decaimiento. Con solo la pose no se ve el producto: el objetivo local es
decidir CUANDO vale la pena gastar una consulta a Gemini (que mira el clip) y dar un
respaldo cuando Gemini no esta disponible. Los umbrales son punto de partida: calibrar
con video real de la tienda (SHADOW_MODE).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import Settings
from .vision import KP_CONF, L_ELBOW, L_EYE, L_HIP, L_SH, L_WRIST, NOSE, R_ELBOW, R_EYE, R_HIP, R_SH, \
    R_WRIST, Person
from .zones import ZoneSet

REACH_MIN_SECONDS = 0.25      # mano en el estante menos que esto = roce, no alcance
REACH_MAX_SECONDS = 6.0       # mas que esto = revisando/reponiendo, no toma rapida
HAND_GRACE = 0.5              # huecos de deteccion tolerados dentro de un mismo alcance
CONCEAL_GRACE = 0.4
POST_CONCEAL_REVIEW_DELAY = 3.0   # s tras el ocultamiento antes de pedir el dictamen (el clip incluye el "despues")
EXIT_MIN_SECONDS = 0.5
REACH_MAX_SPEED = 1.0         # anchos de hombro por segundo; mas rapido = va caminando, no alcanza
TOGETHER_MAX = 0.8            # distancia entre munecas (en anchos de hombro) para "manos juntas"
RETURN_SAME_SPOT = 1.5        # radio (anchos de hombro) para considerar que vuelve al mismo sitio
SCAN_THRESHOLD = 0.35         # |giro| minimo de la cabeza para contarlo como mirada lateral
SCAN_WINDOW = 10.0
SCAN_MIN_TURNS = 2
STAFF_SECONDS = 2.0           # s en zona de personal para considerarlo empleado
SCORE_TAU = 180.0             # s de decaimiento de cada evento
EVENT_KEEP = 900.0
OBSERVE_SCORE = 0.20
STAGE_REACH_SECONDS = 15.0    # un alcance cuenta como etapa vigente (panel/overlay) durante este tiempo

# Pesos de cada evento en la puntuacion.
W_REACH = 0.08
W_SCAN = 0.12
W_CONCEAL = 0.50
W_CONCEAL_REPEAT = 0.20       # extra por cada ocultamiento previo sin descartar
W_EXIT = 0.45


def _pt(kpts: np.ndarray, idx: int) -> Optional[np.ndarray]:
    return kpts[idx, :2] if kpts[idx, 2] >= KP_CONF else None


def _mean(points: List[np.ndarray]) -> Optional[np.ndarray]:
    return np.mean(points, axis=0) if points else None


@dataclass
class Body:
    scale: float                      # ancho de hombros (o 22% del alto del cuadro): unidad de distancia
    sh_mid: Optional[np.ndarray]
    hips: List[np.ndarray]
    hip_mid: Optional[np.ndarray]
    yaw: Optional[float]              # giro de la cabeza: ~0 de frente, +-1 de perfil


def analyze_body(kpts: np.ndarray, bbox: np.ndarray) -> Body:
    bh = max(1.0, float(bbox[3] - bbox[1]))
    ls, rs = _pt(kpts, L_SH), _pt(kpts, R_SH)
    both = ls is not None and rs is not None
    scale = max(float(np.linalg.norm(ls - rs)) if both else 0.0, 0.22 * bh)
    hips = [p for p in (_pt(kpts, L_HIP), _pt(kpts, R_HIP)) if p is not None]

    yaw = None
    nose, le, re = _pt(kpts, NOSE), _pt(kpts, L_EYE), _pt(kpts, R_EYE)
    if nose is not None:
        if le is not None and re is not None:
            yaw = float((nose[0] - (le[0] + re[0]) / 2) / max(1.0, abs(le[0] - re[0])))
        elif le is not None or re is not None:
            eye = le if le is not None else re
            yaw = math.copysign(1.0, float(nose[0] - eye[0]))
    return Body(scale, (ls + rs) / 2 if both else None, hips, _mean(hips), yaw)


def concealment_place(wrist: np.ndarray, body: Body) -> Optional[str]:
    """Donde esta una mano visible respecto al cuerpo: 'bolsillo' (cadera/cintura) o
    'pecho_o_abdomen' (bajo la chaqueta/camisa). None si esta en una posicion libre."""
    s = body.scale
    if body.hips:
        near = min(float(np.linalg.norm(wrist - h)) for h in body.hips)
        lo, hi = min(h[0] for h in body.hips) - 0.3 * s, max(h[0] for h in body.hips) + 0.3 * s
        if near <= 0.4 * s and lo <= wrist[0] <= hi:
            return "bolsillo"
    if body.sh_mid is not None:
        bottom = body.hip_mid[1] if body.hip_mid is not None else body.sh_mid[1] + 1.4 * s
        if abs(wrist[0] - body.sh_mid[0]) <= 0.35 * s and body.sh_mid[1] + 0.25 * s <= wrist[1] <= bottom:
            return "pecho_o_abdomen"
    return None


@dataclass
class HandInfo:
    pos: Optional[np.ndarray]         # muneca visible (None si no se ve)
    in_shelf: bool
    place: Optional[str]              # lugar de ocultamiento de ESTA mano


def hand_info(kpts: np.ndarray, body: Body, zones: ZoneSet, walking: bool) -> List[HandInfo]:
    """Estado de cada mano (izquierda, derecha). Caminando no hay alcance: solo se pasa junto al estante."""
    out = []
    for wrist_idx, elbow_idx in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
        wrist = _pt(kpts, wrist_idx)
        if wrist is not None:
            out.append(HandInfo(wrist, zones.contains("estanteria", wrist) and not walking,
                                concealment_place(wrist, body)))
            continue
        place = None
        if body.hip_mid is not None:
            # Muneca perdida con el codo junto al cuerpo: la mano esta tapada por el propio cuerpo o la ropa.
            elbow = _pt(kpts, elbow_idx)
            if elbow is not None and float(np.linalg.norm(elbow - body.hip_mid)) <= 0.8 * body.scale:
                place = "mano_oculta"
        out.append(HandInfo(None, False, place))
    return out


def hands_together_at_waist(hands: List[HandInfo], body: Body) -> bool:
    """Las dos manos juntas delante del cuerpo, entre el pecho y la cadera (p. ej. sobre la abertura de un bolso)."""
    if hands[0].pos is None or hands[1].pos is None or body.hip_mid is None:
        return False
    if float(np.linalg.norm(hands[0].pos - hands[1].pos)) > TOGETHER_MAX * body.scale:
        return False
    mid = (hands[0].pos + hands[1].pos) / 2
    dx, dy = (mid - body.hip_mid) / body.scale
    return -0.9 <= dy <= 0.2 and abs(dx) <= 0.9


class Sustain:
    """Mide cuanto tiempo se cumple una condicion, tolerando huecos cortos (`grace`)."""

    def __init__(self, grace: float = 0.5) -> None:
        self.grace = grace
        self.since: Optional[float] = None
        self._last_true = 0.0

    def update(self, condition: bool, now: float) -> float:
        if condition:
            if self.since is None:
                self.since = now
            self._last_true = now
        elif self.since is not None and now - self._last_true > self.grace:
            self.since = None
        return 0.0 if self.since is None else now - self.since

    def reset(self) -> None:
        self.since = None


@dataclass
class HandState:
    reach_start: Optional[float] = None
    reach_last: float = 0.0
    reach_pos: Optional[np.ndarray] = None
    checked: bool = False             # ya se evaluo si este alcance es una devolucion
    armed_until: float = 0.0          # hasta cuando un ocultamiento cuenta como posterior a un alcance
    pending_reach_at: Optional[float] = None
    pending_reach_pos: Optional[np.ndarray] = None
    returning_pending: bool = False


@dataclass
class TrackState:
    track_id: int
    camera_id: str
    first_seen: float
    last_seen: float
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    kpts: np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))
    flow_wrist: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None])
    flow_wrist_time: float = 0.0
    updated: bool = False
    is_staff: bool = False
    # secuencia de robo (por mano: 0 = izquierda, 1 = derecha)
    hands: List["HandState"] = field(default_factory=lambda: [HandState(), HandState()])
    conceal_timer: Sustain = field(default_factory=lambda: Sustain(CONCEAL_GRACE))
    conceal_from: Optional[np.ndarray] = None   # donde alcanzo el estante antes del ultimo ocultamiento
    center: Optional[np.ndarray] = None
    last_move: float = 0.0
    speed: float = 0.0                # velocidad del cuerpo (anchos de hombro por segundo, suavizada)
    exit_timer: Sustain = field(default_factory=lambda: Sustain(0.5))
    checkout_timer: Sustain = field(default_factory=lambda: Sustain(1.0))
    staff_timer: Sustain = field(default_factory=lambda: Sustain(1.0))
    paid_at: Optional[float] = None
    exit_time: Optional[float] = None
    unreturned_reach_exit: bool = False
    reaches: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    conceals: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    conceal_place: str = ""
    hand_place: str = ""              # ultimo lugar de mano oculta visto (la deteccion tolera huecos)
    events: Deque[Tuple[float, str, float]] = field(default_factory=lambda: deque(maxlen=100))
    cleared_until: float = 0.0        # eventos con t <= esto fueron descartados por Gemini
    yaw_sign: int = 0
    turns: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    scan_counted_until: float = 0.0
    # puntuacion y estado local
    score: float = 0.0
    local_state: str = "normal"       # normal | observacion | ocultamiento | salida_con_ocultamiento | personal
    # Gemini: un dictamen por ocultamiento
    review_event: Optional[float] = None   # ocultamiento que espera dictamen
    review_due: float = 0.0
    review_pending: bool = False
    verdict: Optional[Dict[str, Any]] = None
    verdict_time: float = 0.0
    verdict_event: Optional[float] = None
    # recortes JPEG recientes para Gemini: (timestamp, bytes)
    crops: Deque[Tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=40))
    # Cuadros amplios de la escena para evidencia/Telegram, separados de Gemini.
    context_crops: Deque[Tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=20))
    last_crop_time: float = 0.0
    last_context_crop_time: float = 0.0
    # decision final / alertas
    level: str = "ok"
    level_source: str = ""
    level_reason: str = ""
    ok_since: Optional[float] = None
    alert_level_sent: str = "ok"
    alert_conceals_sent: int = 0
    alert_sent_at: float = 0.0

    # ---- consultas ---------------------------------------------------------
    def active_conceals(self) -> List[float]:
        return [t for t in self.conceals if t > self.cleared_until]

    def head_turns(self, now: float) -> int:
        return sum(1 for t in self.turns if now - t <= SCAN_WINDOW)

    def score_at(self, now: float) -> float:
        total = sum(w * math.exp(-(now - t) / SCORE_TAU) for t, _, w in self.events if t > self.cleared_until)
        return float(min(3.0, total))

    def summary(self, now: float) -> Dict[str, Any]:
        """Indicadores compactos (se envian a Gemini como contexto y a la API)."""
        active = self.active_conceals()
        return {
            "estado_local": self.local_state,
            "puntuacion_local": round(self.score, 2),
            "alcances_2min": sum(1 for t in self.reaches if now - t <= 120),
            "ocultamientos": len(active),
            "lugar_ocultamiento": self.conceal_place or None,
            "giros_de_cabeza": self.head_turns(now),
            "seg_desde_ocultamiento": round(now - active[-1]) if active else None,
            "paso_por_caja": self.paid_at is not None,
            "salio": self.exit_time is not None,
        }

    def hand_states(self, now: float) -> List[str]:
        """Por mano (izquierda, derecha): "alcance" (en el estante), "vigilada" (tomo algo y aun puede ocultarlo) o ""."""
        states = []
        for hand in self.hands:
            if hand.reach_start is not None and now - hand.reach_last <= HAND_GRACE:
                states.append("alcance")
            elif now <= hand.armed_until:
                states.append("vigilada")
            else:
                states.append("")
        return states

    def stage(self, now: float) -> Dict[str, bool]:
        """Etapas de la secuencia de sustraccion que lleva esta persona (solo para mostrar)."""
        return {
            "alcance": any(self.hand_states(now)) or any(now - t <= STAGE_REACH_SECONDS for t in self.reaches),
            "ocultamiento": bool(self.active_conceals()),
            "caja": self.paid_at is not None,
            "salida": self.exit_time is not None,
        }


def _add(ts: TrackState, now: float, kind: str, weight: float) -> None:
    ts.events.append((now, kind, weight))


def _count_scan(ts: TrackState, now: float) -> None:
    """Mirar a los lados repetidamente (camaras, personal) antes de tomar/ocultar suma una vez por episodio."""
    if ts.head_turns(now) >= SCAN_MIN_TURNS and now >= ts.scan_counted_until:
        _add(ts, now, "mirada", W_SCAN)
        ts.scan_counted_until = now + SCAN_WINDOW


def _track_head(ts: TrackState, yaw: Optional[float], now: float) -> None:
    if yaw is None or abs(yaw) < SCAN_THRESHOLD:
        return
    sign = 1 if yaw > 0 else -1
    if ts.yaw_sign and sign != ts.yaw_sign:
        ts.turns.append(now)
    ts.yaw_sign = sign


def _check_return(ts: TrackState, hand: HandState, body: Body, now: float, s: Settings) -> None:
    """La misma zona del estante se vuelve a tocar poco despues de un ocultamiento: lo devolvio (o sigue mirando)."""
    active = ts.active_conceals()
    if not active or now - active[-1] > s.return_window or hand.reach_pos is None or ts.conceal_from is None:
        return
    if float(np.linalg.norm(hand.reach_pos - ts.conceal_from)) > RETURN_SAME_SPOT * body.scale:
        return
    if ts.review_event == active[-1]:
        ts.review_event = None
    _add(ts, now, "devolucion", 0.0)
    ts.cleared_until = max(ts.cleared_until, active[-1])


def update_track(ts: TrackState, person: Person, zones: ZoneSet, now: float, s: Settings) -> None:
    """Actualiza el estado de un track con un nuevo analisis de pose."""
    ts.last_seen, ts.updated = now, True
    ts.bbox, ts.kpts = person.bbox, person.kpts
    feet = ((person.bbox[0] + person.bbox[2]) / 2, person.bbox[3])

    # Empleados: quien permanece en la zona de personal queda excluido.
    if ts.staff_timer.update(zones.contains("personal", feet), now) >= STAFF_SECONDS:
        ts.is_staff = True
    if ts.is_staff:
        ts.local_state, ts.score = "personal", 0.0
        return

    body = analyze_body(person.kpts, person.bbox)
    _track_head(ts, body.yaw, now)
    center = np.array([(person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2])
    if ts.center is not None:
        dt = max(0.05, now - ts.last_move)
        ts.speed = 0.6 * ts.speed + 0.4 * float(np.linalg.norm(center - ts.center) / body.scale / dt)
    ts.center, ts.last_move = center, now
    hands = hand_info(person.kpts, body, zones, walking=ts.speed > REACH_MAX_SPEED)

    # 1) Alcance, mano por mano: la mano entra al estante y sale despues de un momento corto.
    for hand, info in zip(ts.hands, hands):
        if info.in_shelf:
            if hand.reach_start is None:
                hand.reach_start, hand.reach_pos, hand.checked = now, info.pos, False
                hand.returning_pending = False
            hand.reach_last = now
            held = now - hand.reach_start
            if held > REACH_MAX_SECONDS:
                hand.armed_until = 0.0
            elif held >= REACH_MIN_SECONDS:
                hand.armed_until = now + s.conceal_window
                if (hand.pending_reach_at is not None and hand.pending_reach_pos is not None and info.pos is not None
                        and float(np.linalg.norm(info.pos - hand.pending_reach_pos)) <= RETURN_SAME_SPOT * body.scale):
                    hand.pending_reach_at = hand.pending_reach_pos = None
                    hand.returning_pending = True
                if not hand.checked:
                    hand.checked = True
                    _check_return(ts, hand, body, now, s)
        elif hand.reach_start is not None and now - hand.reach_last > HAND_GRACE:
            held = hand.reach_last - hand.reach_start
            hand.reach_start = None
            if REACH_MIN_SECONDS <= held <= REACH_MAX_SECONDS:
                ts.reaches.append(hand.reach_last)
                _add(ts, hand.reach_last, "alcance", W_REACH)
                _count_scan(ts, now)
                if not hand.returning_pending:
                    hand.pending_reach_at = hand.reach_last
                    hand.pending_reach_pos = hand.reach_pos.copy() if hand.reach_pos is not None else None
            hand.returning_pending = False

    # 2) Ocultamiento: tras un alcance, esa mano queda junto al cuerpo (bolsillo/bolso/pecho),
    #    o las dos manos se juntan a la cintura.
    armed = [now <= h.armed_until and not i.in_shelf for h, i in zip(ts.hands, hands)]
    place = next((i.place for a_, i in zip(armed, hands) if a_ and i.place), None)
    if place is None and any(armed) and hands_together_at_waist(hands, body):
        place = "manos_a_la_cintura"
    if place:
        ts.hand_place = place
    if ts.conceal_timer.update(place is not None, now) >= s.conceal_min_seconds:
        prior = len(ts.active_conceals())
        ts.conceals.append(now)
        ts.conceal_place = place or ts.hand_place
        reached = [h for h in ts.hands if h.reach_pos is not None]
        ts.conceal_from = max(reached, key=lambda h: h.reach_last).reach_pos if reached else None
        _add(ts, now, "ocultamiento", W_CONCEAL + W_CONCEAL_REPEAT * prior)
        _count_scan(ts, now)
        for hand in ts.hands:
            hand.armed_until = 0.0
        ts.conceal_timer.reset()
        ts.review_event, ts.review_due = now, now + POST_CONCEAL_REVIEW_DELAY

    # 3) Caja y salida (se miden con los pies).
    if ts.checkout_timer.update(zones.contains("caja", feet), now) >= s.checkout_min_seconds and ts.paid_at is None:
        ts.paid_at = now
    if ts.exit_time is None and ts.exit_timer.update(zones.contains("salida", feet), now) >= EXIT_MIN_SECONDS:
        ts.exit_time = now
        if ts.active_conceals():
            _add(ts, now, "salida", W_EXIT)
        elif ts.paid_at is None and any(
                hand.pending_reach_at is not None and now - hand.pending_reach_at <= s.conceal_window
                for hand in ts.hands):
            ts.unreturned_reach_exit = True

    while ts.events and now - ts.events[0][0] > EVENT_KEEP:
        ts.events.popleft()
    ts.score = ts.score_at(now)
    active = ts.active_conceals()
    if active and ts.exit_time is not None and ts.exit_time >= active[0]:
        ts.local_state = "salida_con_ocultamiento"
    elif active:
        ts.local_state = "ocultamiento"
    elif ts.score >= OBSERVE_SCORE:
        ts.local_state = "observacion"
    else:
        ts.local_state = "normal"


def clear_through(ts: TrackState, event_time: float) -> None:
    """Gemini descarto el ocultamiento: se anulan ese evento y todo lo anterior."""
    ts.cleared_until = max(ts.cleared_until, event_time)
    ts.score = ts.score_at(ts.last_seen)
    if not ts.active_conceals():
        ts.local_state = "normal" if ts.score < OBSERVE_SCORE else "observacion"
