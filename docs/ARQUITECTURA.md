# Arquitectura

## Flujo por camara (`pipeline.py`, un hilo por camara)

```
CameraStream.latest()          frame mas reciente (camera.py)
  -> FrameOptimizer.process    reduce a <=640 px, devuelve (frame_reducido, escala), modo nocturno
  -> PoseTracker.infer         YOLO-pose + ByteTrack -> [Person(track_id, bbox, conf, kpts 17x3)]
  -> por persona:
       update_track(ts, person, zones, now, settings)     # heuristics.py
       store_crop(ts, frame_original, escala, now, ...)   # clips.py  (solo si Gemini esta activo)
       _maybe_review(ts, now)                             # pide dictamen a Gemini si toca
       decide(ts, now, settings) -> (nivel, fuente, motivo)   # decision.py
       alerts.evaluate(ts, decision, ...)                 # alerts.py
  -> draw_tracks -> JPEG para /stream/<cam>
```

`FpsGate`: `INFERENCE_FPS` (8) con gente en escena, `IDLE_FPS` (2) vacia. El robo dura 1-3 s: no bajar de ~8 fps.

## `TrackState` (heuristics.py) - estado por persona

- Identidad: `track_id`, `camera_id`, `first_seen`, `last_seen`, `bbox`, `kpts`, `is_staff`.
- Por mano (`hands[0]`=izq, `hands[1]`=der, `HandState`): `reach_start/last/pos`, `armed_until` (hasta cuando un ocultamiento cuenta como posterior a un alcance), `checked`.
- Secuencia: `reaches`, `conceals` (tiempos), `conceal_place`, `hand_place`, `conceal_from` (donde alcanzo antes del ocultamiento), `exit_time`, `paid_at`, `cleared_until`, `events` (t, tipo, peso), `speed`, `turns` (giros de cabeza).
- Revision de Gemini: `review_event` (ocultamiento que espera dictamen), `review_due`, `review_pending`, `verdict`, `verdict_time`, `verdict_event`, `crops` (JPEG recientes).
- Decision/alertas: `score`, `local_state`, `level`, `level_source`, `level_reason`, `alert_level_sent`, `alert_conceals_sent`, `alert_sent_at`.

## Heuristica local (`update_track`)

Orden en cada cuadro:
1. **Personal**: pies en zona `personal` >= 2 s => `is_staff` (excluido, sticky).
2. **Cuerpo**: `analyze_body` (escala = ancho de hombros o 22% del alto del cuadro; caderas; giro de cabeza). Velocidad del cuerpo (EMA). Caminando (`speed > 1.0` anchos de hombro/s) => no hay alcance.
3. **Alcance por mano**: muneca en zona `estanteria` entre 0.25 y 6 s (`REACH_MIN/MAX`). Al cumplir 0.25 s arma la mano (`armed_until = now + CONCEAL_WINDOW`) y comprueba **devolucion** (`_check_return`). Al terminar suma evento `alcance`.
4. **Ocultamiento**: mano armada y fuera del estante en `concealment_place` (`bolsillo` cerca de cadera, `pecho_o_abdomen`, o muneca perdida con codo junto al cuerpo = `mano_oculta`), o **ambas manos juntas** entre pecho y cadera (`manos_a_la_cintura`). Debe sostenerse `CONCEAL_MIN_SECONDS`. Crea evento `ocultamiento`, desarma las manos, fija `review_event` y `review_due = now + 3 s`.
5. **Caja y salida** con los pies: `caja` >= `CHECKOUT_MIN_SECONDS` => `paid_at`; `salida` >= 0.5 s => `exit_time` (+ evento `salida` si hay ocultamiento activo).
6. **Puntuacion**: suma de pesos con decaimiento exponencial (tau 180 s), ignorando eventos <= `cleared_until`. Pesos: alcance 0.08, mirada 0.12, ocultamiento 0.50 (+0.20 por cada previo), salida 0.45.
7. **`local_state`**: `personal` | `salida_con_ocultamiento` | `ocultamiento` | `observacion` (score >= 0.20) | `normal`.

Constantes (arriba de `heuristics.py`): `REACH_*`, `HAND_GRACE`, `CONCEAL_GRACE`, `POST_CONCEAL_REVIEW_DELAY` (3.0), `EXIT_MIN_SECONDS`, `REACH_MAX_SPEED`, `TOGETHER_MAX`, `RETURN_SAME_SPOT`, `SCAN_*`, pesos `W_*`.

**Devolucion**: si una mano entra al estante (confirmada >= 0.25 s) dentro de `RETURN_WINDOW_SECONDS` del ultimo ocultamiento activo y a menos de 1.5 anchos de hombro de donde alcanzo antes, el ocultamiento se anula (`cleared_until`), se registra evento `devolucion` y se cancela su revision pendiente.

## Decision (`decision.py`)

Niveles: `ok < observacion < posible_ocultamiento < probable_sustraccion`. Solo los dos ultimos alertan (`ALERT_MIN_RANK`).

Sobre los ocultamientos **activos** (no anulados):
1. Sin ocultamientos: `observacion` si `local_state == observacion`, si no `ok`.
2. Dictamen de Gemini `ocultamiento` con confianza >= `MIN_CONF_CONFIRM` y `verdict_event` entre los activos: `posible_ocultamiento`, o `probable_sustraccion` si ya salio.
3. Revision pendiente: se mantiene el nivel (minimo `observacion` "Verificando"), sin escalar a ciegas.
4. Respaldo local (Gemini apagado/fallo/indeterminado): `probable` si salio, no paso por caja y score >= `LOCAL_PROBABLE_SCORE` (0.90); `posible` si score >= `LOCAL_POSSIBLE_SCORE` (0.70); si no `observacion`.

`apply_verdict(ts, event, verdict, now, s)`: guarda el dictamen; si es `normal` con confianza >= `MIN_CONF_CLEAR`, `clear_through(ts, event)` anula ese evento y los anteriores.

## Gemini (`gemini.py`, `clips.py`)

- Una consulta por ocultamiento, `review_due` = evento + 3 s. Clip = hasta `GEMINI_FRAMES` (12) recortes entre `CLIP_SECONDS` (4) s antes del evento y el instante de la consulta, repartidos parejo. Recortes de persona completa con margen (30% ancho, 12% alto), ampliados/reducidos a `GEMINI_CROP_SIDE` (448).
- Salida estructurada `Verdict`: `state` (`ocultamiento|normal|indeterminado`), `confidence`, `action`, `product`, `place`, `looked_around`, `reasoning`.
- Prompt (`SYSTEM_PROMPT`): analista de prevencion de perdidas; pide mirar manos y bolso; `normal` solo si se ve que el producto queda visible/se devuelve/no hubo toma; ante duda, `indeterminado` (nunca `normal`); nunca afirmar "robo".
- Control de gasto: cooldown por persona, `GEMINI_MAX_CALLS_PER_HOUR`, pausa 60 s ante 429, desactivacion ante clave invalida/modelo 404, **2 reintentos** ante 503/504.
- Cada consulta ronda 14k tokens (12 recortes).

## Alertas (`alerts.py`)

Emite al **escalar de nivel**, o ante un nuevo ocultamiento si pasaron `ALERT_REPEAT_SECONDS` (30). Tras `RESOLVE_AFTER_OK_SECONDS` por debajo del nivel de alerta se rearma. Cada alerta guarda foto anotada (`data/alerts/*.jpg`), registro JSONL (`alerts.jsonl`), indicadores (`ts.summary`) y dictamen; webhook opcional. `SHADOW_MODE`: registra con `shadow=true`, `ack=true`, sin webhook ni aviso. Alerta de sistema si la camara pierde señal > `CAMERA_OFFLINE_SECONDS`.

## Configuracion y datos

- `Settings.from_env()` valida rangos y reporta todos los errores juntos. Variables: ver `.env.example` (fuente de verdad) y README.
- Camaras: `data/cameras.json`. Las contrasenas van al llavero del sistema (`credentials.py`, paquete `keyring`, servicio `tukuru`, usuario `camara:<id>`) y el archivo queda con `password: ""`; sin llavero quedan en el archivo (chmod 600). `CREDENTIALS_STORE=auto|keyring|file`. Un archivo antiguo en texto plano se migra al cargar. Se siembran desde `CAMERA_n_*` del `.env` solo si el archivo no existe. Cada camara lleva `zones`.
- Zonas: `{"estanteria": [poligono,...], "caja": [...], "salida": [...], "personal": [...]}`; cada poligono 3-16 puntos `[x, y]` normalizados; hasta 8 por tipo. `estanteria` se mide con **munecas**; `caja`, `salida`, `personal` con los **pies** (centro inferior del bbox).

## API web (`web.py`)

`GET /` panel, `/healthz`, `/api/status`, `/stream/<cam>` (MJPEG), `/api/alerts`, `POST /api/alerts/<id>/ack`, `/evidence/<archivo>`, `GET/POST /api/cameras`, `POST /api/cameras/test`, `DELETE /api/cameras/<id>`. Escrituras: JSON + cabecera `X-Requested-With: tukuru`. Basic Auth si `DASHBOARD_USER/PASSWORD`; con `WEB_HOST` expuesto son obligatorios.
