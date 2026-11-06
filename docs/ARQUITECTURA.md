# Arquitectura

## Observador experimental de objetos (`objects.py`)

Disponible solamente en `replay` mediante `--object-model RUTA`, apagado por defecto.
Una sola instancia de YOLO de detección, independiente del modelo de pose. Cada alcance
confirmado abre una ventana de 8 s. Máximo 2 inferencias/s por persona y 4/s globales,
con selección de la persona que lleva más tiempo sin analizar. Recorte de persona y
muñecas sobre el cuadro original, `imgsz=416`, máximo 30 detecciones por llamada.
No hay cola, hilos nuevos ni modelos por persona. Las sesiones desaparecen al perder el track.

Asocia detecciones a la muñeca visible más cercana por distancia al rectángulo;
rechaza distancias grandes y ambigüedad entre manos/personas. Persiste clase y centro
con una puerta espacial/temporal; exige varias observaciones para cambiar de estado.
Esto es asociación aproximada, no identificación visual inequívoca del mismo producto.
Registra `objeto_en_estanteria`, `objeto_junto_a_mano`, `objeto_junto_al_cuerpo`,
`objeto_junto_a_contenedor` y `desconocido` al perder evidencia. La posición del objeto,
no solo la mano, determina proximidad al estante/contenedor. No determina inserción,
pago, propiedad ni robo. Un celular conserva su etiqueta, no se convierte en producto.

Se exportan detecciones con clase, confianza y caja en coordenadas del cuadro reducido,
transiciones por mano y tiempos. La línea de tiempo original permanece separada.
No escribe en `TrackState`, no interviene en `decide`, ni se envía como contexto a Gemini.
`CONTAINERS` admite etiquetas de canasta para futuros pesos ajustados; COCO no incluye
una clase general de canasta ni de producto de abarrotes.

## Incidentes, revision y ML en sombra

`AlertManager` conserva el registro JSONL y la foto. Cada alerta de persona nueva
congela un vector de caracteristicas de `TrackState` en `data/incidents.sqlite3`, con
identificador de sesion para evaluar por grupos. El panel y el bot opcional de Telegram
registran revisiones humanas con historial (`confirmado`, `descartado`, `no_concluyente`).
Reconocer o retirar una alerta de la vista no es una etiqueta. Telegram usa sondeo de
`callback_query`, exige chat e IDs de usuario autorizados y nunca bloquea los hilos de
camara al enviar avisos.
Cuando se anula el ocultamiento asociado a una alerta, esta queda marcada como
`resolved` en el historial JSONL y en la vista. Deja de contar como pendiente, pero
no recibe una etiqueta de entrenamiento sin revision humana. El panel reproduce un
aviso sonoro solo ante alertas nuevas y activas, tras la primera interaccion que el
navegador exige para permitir audio.
`GET /api/status` incluye el resumen de las ultimas etiquetas por incidente del
esquema vigente, sesiones registradas y estados de configuracion de Telegram/modelo.
Las sesiones registradas no garantizan videos independientes.

## Control de Gemini y propuesta de zonas

El panel usa `POST /api/gemini` para activar o desactivar el juez en caliente. La
preferencia booleana se guarda en `data/gemini_state.json`, sin clave. Al desactivar,
no se admiten consultas nuevas; resultados de la generacion anterior no se aplican.
Una llamada que ya salio a la red puede terminar. El archivo guardado prevalece sobre
`GEMINI_ENABLED` al reiniciar; sin archivo se usa ese valor inicial.

`POST /api/cameras/<id>/suggest-zones` solo se ejecuta por pulsacion del operador.
Toma un cuadro reciente sin anotaciones de la camara conectada, lo reduce a un maximo
de 640 px y envia un JPEG a Gemini. Usa por defecto `gemini-3-flash-preview` para
esta tarea; el operador puede elegir el modelo configurado del juez. Los errores HTTP
de modelo, cuota y servicio se traducen a mensajes sin exponer la clave. La respuesta
puede tardar hasta al menos 60 s antes de agotar el tiempo de espera. La respuesta
estructurada se valida como poligonos
normalizados antes de enviarla al panel. El panel superpone los poligonos a la misma
captura y exige `Usar propuesta` y `Guardar y conectar` para modificar la camara.
Ni la propuesta ni el modelo cambian las alertas hasta guardar zonas nuevas; las
zonas no visibles deben quedar vacias.

`GET /api/cameras/<id>/zone-frame` devuelve una captura local de la camara conectada,
sin consultar Gemini. El editor SVG del panel usa esa imagen para crear rectangulos,
mover poligonos o ajustar sus vertices por mouse. Las posiciones se mantienen en
coordenadas normalizadas y se copian al formulario de la camara; la API existente de
guardado sigue validando 3-16 puntos por poligono, 0-1 por coordenada y hasta 8
poligonos por tipo. Las propuestas de Gemini usan el mismo editor y exigen pulsar
`Usar zonas dibujadas` antes de guardarlas.

`tukuru.ml.dataset` exporta solo incidentes revisados con el esquema vigente;
`tukuru.ml.train` exige varias sesiones y ambas clases en entrenamiento/evaluacion.
Entrena una regresion logistica fuera de la app y exporta coeficientes a JSON. Si
`ML_SHADOW_MODEL` apunta al candidato, Tukuru anota una puntuacion experimental en las
alertas sin cambiar `decide()` ni su nivel. La exportacion puede incluir muestras
locales etiquetadas, ademas de alertas; aun no representa eventos que pose/tracking no
llegaron a observar.

## Procedencia y revision agrupada

La base SQLite conserva `session_id` (ejecucion) y `recording_id` (origen de la
grabacion) como conceptos distintos. El formulario de camaras permite definir un grupo
estable por fuente/sesion; esa relacion depende de la declaracion correcta del
operador y no se verifica automaticamente. En `replay`, el origen por defecto es
SHA-256 del archivo. No se guarda
la ruta de video. Registros anteriores o streams sin grupo configurado tienen origen
desconocido y no salen en el dataset supervisado. `GET /api/reviews/queue` agrupa
alertas pendientes y muestras locales de la misma camara/track/origen en ventanas de
120 s; el panel permite seleccionar y etiquetar grupos por lote. No propaga etiquetas a futuros
incidentes ni cambia el nivel de alerta.
Si hay puntuacion `ML_SHADOW_MODEL`, los casos cercanos a 0.5 se ordenan primero para
revision incierta; la puntuacion se muestra como experimental y no se interpreta como
probabilidad calibrada. Si existen al menos 30 episodios `descartado` y 5 grupos de
origen declarados, la cola calcula un percentil k-NN de distancia a esos casos como
prioridad exploratoria de novedad. No etiqueta, entrena ni activa nada; el reporte
`tukuru.ml.quality` indica cuando la referencia no alcanza. El pipeline muestrea eventos `alcance` y `devolucion` por
debajo del nivel de alerta. El limite es 20 muestras por camara/hora y 30 s por
seguimiento/evento; las imagenes quedan locales y se purgan con la retencion de
evidencia. La señal de novedad solo ordena revisiones; no es una deteccion de robo ni
una probabilidad calibrada.

## Flujo por camara (`pipeline.py`, un hilo por camara)


```
CameraStream.latest()          frame mas reciente (camera.py)
  -> FrameOptimizer.process    reduce a <=640 px, devuelve (frame_reducido, escala), modo nocturno
  -> PoseTracker.infer         YOLO-pose + ByteTrack -> [Person(track_id, bbox, conf, kpts 17x3)]
  -> WristMotionTracker        flujo optico forward-backward para una muneca ya detectada durante un alcance
  -> por persona:
       update_track(ts, person, zones, now, settings)     # heuristics.py
       store_crop(ts, frame_original, escala, now, ...)   # clips.py  (solo si Gemini esta activo)
       observe_event(...)                                 # muestreo acotado local; no alerta/notifica
       _maybe_review(ts, now)                             # pide dictamen a Gemini si toca
       decide(ts, now, settings) -> (nivel, fuente, motivo)   # decision.py
       alerts.evaluate(ts, decision, ...)                 # alerts.py
  -> draw_tracks -> JPEG para /stream/<cam>
```

`FpsGate`: `INFERENCE_FPS` (8) con gente en escena, `IDLE_FPS` (2) vacia. El robo dura 1-3 s: no bajar de ~8 fps. ByteTrack conserva los tracks perdidos `TRACK_BUFFER_FRAMES` (60 por defecto, ~7,5 s a 8 fps) usando un YAML generado en `data/trackers/`; no se modifica el YAML de Ultralytics.

`WristMotionTracker` usa Lucas-Kanade con un parche local de puntos y validacion
forward-backward para prolongar una muneca conocida si la pose desaparece por una
oclusion breve. Solo sigue manos con alcance ya observado y dentro del horizonte activo;
si la muneca fluida permanece en estanteria, extiende `reach_last`. No crea una deteccion,
no inicia un alcance, no afirma que se tomo un producto y no suma puntuacion. El overlay
marca el punto con `flujo mano`, distinto de la muneca roja detectada por pose. Si la
textura, la consistencia o la duracion del flujo no alcanzan los limites, descarta la
posicion y espera a la siguiente pose real. No recupera el ID si ByteTrack asigna otro.

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
3. Respaldo local incluso con revision pendiente: `probable` si salio, no paso por caja y score >= `LOCAL_PROBABLE_SCORE` (0.90); `posible` si score >= `LOCAL_POSSIBLE_SCORE` (0.70).
4. Aviso preventivo: una sola secuencia sostenida de alcance y mano hacia bolsillo, pecho o cintura que no se devolvio en 3 s produce `posible` si score >= 0.50. Una muneca perdida sin posicion de ocultamiento no basta.
5. Si no se cumple lo anterior, `observacion` mientras se revisa o ante un ocultamiento sin confirmar. Un dictamen posterior que descarte el evento resuelve la alerta preventiva.

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
- Camaras: `data/cameras.json`. Las contrasenas van al llavero del sistema (`credentials.py`, paquete `keyring`, servicio `tukuru`, usuario `camara:<id>`) y el archivo queda con `password: ""`; sin llavero quedan en el archivo (chmod 600). `CREDENTIALS_STORE=auto|keyring|file`. Un archivo antiguo en texto plano se migra al cargar. Se siembran desde `CAMERA_n_*` del `.env` solo si el archivo no existe. Cada camara lleva `zones` y puede sobrescribir `yolo_conf` (0.05-0.95) y `min_person_height` (proporcion 0.01-0.50). Si quedan vacios, heredan la confianza global y el filtro de altura actual (10%). Son filtros de personas, no un detector de cabezas; reducirlos puede recuperar detecciones pequeñas y aumentar falsos positivos. Guardar la camara reinicia su pipeline y sus IDs activos.
- Zonas: `{"estanteria": [poligono,...], "caja": [...], "salida": [...], "personal": [...]}`; cada poligono 3-16 puntos `[x, y]` normalizados; hasta 8 por tipo. `estanteria` se mide con **munecas**; `caja`, `salida`, `personal` con los **pies** (centro inferior del bbox).

## API web (`web.py`)

`GET /` panel, `/healthz`, `/api/status`, `/stream/<cam>` (MJPEG), `/api/alerts`, `POST /api/alerts/<id>/ack`, `/evidence/<archivo>`, `GET/POST /api/cameras`, `POST /api/cameras/test`, `DELETE /api/cameras/<id>`. Escrituras: JSON + cabecera `X-Requested-With: tukuru`. Basic Auth si `DASHBOARD_USER/PASSWORD`; con `WEB_HOST` expuesto son obligatorios.
