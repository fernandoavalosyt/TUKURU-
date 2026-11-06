# Decisiones, alternativas descartadas y pendientes

Registro de **por que** el sistema es como es. Si cambias una regla que afecte alertas, agrega aqui la decision y su motivo.

## Decisiones

- **Origen de grabacion y revision por lotes (noviembre 2026):** `session_id` solo
  identifica el proceso de Tukuru y no demuestra independencia. Las camaras pueden
  declarar un `recording_group_id` local que se conserva al reiniciar; el replay usa
  SHA-256 del archivo, sin guardar su ruta. Incidentes historicos sin origen quedan
  desconocidos y se excluyen del entrenamiento/evaluacion agrupada. La cola agrupa
  alertas del mismo origen, camara y track separadas por hasta 120 s; el operador
  confirma cada grupo explicitamente. Las revisiones por lote agregan etiquetas
  humanas, pero `no_concluyente` nunca se exporta. La cola incluye alertas y muestras
  de alcances/devoluciones que quedaron debajo del nivel de alerta; aun existe sesgo
  porque no captura eventos que pose/tracking no observaron. Se limita a 20 muestras
  por camara/hora y 30 s por seguimiento/evento; la evidencia es local, usa la
  retencion normal y no se envia a Gemini ni Telegram. Si se carga un modelo en sombra,
  puntajes cerca de 0.5 suben como incertidumbre experimental, no como probabilidad
  calibrada. La prioridad de novedad compara k vecinos normalizados con ejemplos
  `descartado`; requiere 30 episodios y cinco grupos de origen declarados. Solo cambia
  el orden de revision y se apaga si faltan datos; no cambia alertas ni produce etiquetas.
  Episodios del mismo seguimiento separados hasta 120 s se deduplican; etiquetas
  humanas contradictorias dentro del episodio bloquean el entrenamiento. La evaluacion
  de calidad de esa prioridad con una verdad de terreno independiente sigue pendiente.

- **Experimento de objetos por evento (noviembre 2026):** se incorpora solo al replay,
  mediante un modelo local opcional. La app en vivo y las decisiones existentes se conservan.
  Se empieza con YOLOv8n de detección para no exigir otra dependencia ni cambiar la pose.
  El objetivo inicial es medir cobertura y costo con `video2`, no asumir que COCO detecta
  el empaque transparente. Asociación mano-objeto-destino conservadora, sin inferir
  ocultamiento de una desaparición ni devolución por tocar el estante. Las relaciones
  no cambian alertas ni el dictamen de Gemini hasta evaluar su utilidad con evidencia.

1. **Pose barata + juez visual (Gemini).** Con solo YOLO-pose no se ve el producto. La pose dispara la revision y da respaldo; Gemini decide. Gemini no recibe video: solo un clip corto de recortes ante un posible ocultamiento (control de gasto).
2. **La sustraccion es una secuencia corta, no un estado.** (Vigilia medía estados sostenidos de 10-30 s; el robo dura 1-3 s.) Secuencia: alcance -> ocultamiento -> salida. Por eso `INFERENCE_FPS=8`.
3. **Seguimiento por mano.** En `video2` la mujer estaba parada junto al estante con una mano dentro del poligono casi todo el tiempo; la regla "ninguna mano en el estante" la dejaba ciega (cero eventos) mientras marcaba al hombre. Ahora cada mano tiene su alcance y su armado.
4. **Caminar no es alcanzar.** Una persona que pasa frente al poligono generaba "alcances" falsos. Si `speed > 1.0` anchos de hombro/s no hay alcance.
5. **La devolucion anula el ocultamiento.** Si la misma mano vuelve al mismo sitio del estante dentro de `RETURN_WINDOW_SECONDS`, es "tomo y devolvio" (patron del hombre en `video2`). Limite conocido: un ladron que vuelve al mismo sitio por otro producto tambien se anula; lo corrige el siguiente ocultamiento.
6. **Pasar por caja NO anula un ocultamiento confirmado por Gemini.** Ocultar un producto y pagar otros es el patron tipico. Solo evita el nivel "probable" cuando la decision es unicamente local.
7. **Manos juntas a la cintura** tambien cuentan como ocultamiento (sobre un bolso): fue el gesto de la mujer.
8. **Niveles y alertas.** `observacion` no alerta (solo panel). Se alerta al escalar o ante un nuevo ocultamiento tras `ALERT_REPEAT_SECONDS`, para no inundar (antes salian 5 alertas en 10 s).
9. **Un dictamen por ocultamiento.** `verdict_event` liga el dictamen al evento; `normal` con confianza anula ese evento y los anteriores (`clear_through`). Un ocultamiento nuevo se vuelve a evaluar.
10. **Clip anclado al evento**, no a "los ultimos N s". Con los ultimos 10 s y 8 cuadros salian ~1.4 s entre cuadros y el instante clave casi no aparecia; Gemini dijo `normal` dos veces. Ahora: 4 s antes + 3 s despues del ocultamiento, 12 cuadros, recortes ampliados a 448 px. (Ajuste hecho viendo `video2`: puede estar algo ajustado a ese clip.)
11. **Prompt: ante duda `indeterminado`, nunca `normal`** (el respaldo local cubre `indeterminado`). Nunca afirmar "robo".
12. **Modo sombra (`SHADOW_MODE`)** para calibrar en tienda real antes de avisar a nadie.
13. **`tukuru.replay`** para calibrar con video sin camara, con panel dibujado (`dashboard.py`) y el `AlertManager` real en carpeta temporal.
14. **Salida del sistema = "comportamiento sospechoso".** Nunca "robo". La confirmacion es humana.
15. **Contrasenas de camaras en el llavero del sistema, no en `.env` ni en texto plano.** Se agregan desde el panel y el archivo `data/cameras.json` solo guarda la configuracion. `keyring` es opcional: sin llavero (servidor sin escritorio) se cae al archivo con chmod 600 y se avisa en el log; si el llavero falla al guardar, la contrasena NO se pierde (queda en el archivo). Alternativa descartada: cifrar el archivo con una clave guardada en `data/` (la clave junto al candado no protege). En Windows `chmod` no restringe nada: ahi el llavero es la proteccion real. Tambien: la URL completa pegada en el panel manda sobre el protocolo del formulario, y el modelo YOLO de cada camara se carga en su hilo para no frenar la peticion web.
16. **GPU opcional con `DEVICE=auto`.** Si PyTorch ve CUDA, YOLO usa FP16; sin CUDA usa CPU FP32. Un fallo de CUDA reinicia esa camara en CPU (sus IDs activos se pierden). No se cambian los umbrales ni la tasa de 8 fps por defecto: con un solo video no hay evidencia para recalibrar alertas.
17. **Un YOLO/ByteTrack por camara por ahora.** El tracker de Ultralytics conserva estado dentro de cada instancia; compartir `model.track(..., persist=True)` mezclaria los IDs entre camaras. En 40 cuadros del video de referencia, CPU dio 4.6 fps por camara con 2 instancias y 3.7 con 3. Un modelo compartido con lock (solo `predict`, sin IDs) dio 3.1 fps por camara con 2; un lote de `predict` dio 6.9 con 2 y 5.5 con 3. El lote acelera la deteccion, pero falta separar ByteTrack por camara para usarlo en la aplicacion. Se mantiene la implementacion con IDs correctos hasta medir y validar esa migracion en GPU.
18. **Sin subir `MAX_FRAME_SIDE` por defecto.** Con `yolov8n-pose` y 40 cuadros del video de referencia, 640 obtuvo 129/152 munecas con confianza >= 0.40 a 6.0 fps; 960 obtuvo 124/150 a 4.3 fps. Es un indicador de visibilidad, no una medida de exactitud. No justifica cambiar el valor para todos los equipos.
19. **Medicion GPU pendiente.** El entorno inicial tenia `torch 2.15.0+cpu`; la instalacion de la rueda `cu128` no termino por la velocidad de descarga de sus dependencias. No hay mediciones de FPS, VRAM ni equivalencia de eventos en GPU. El replay CPU posterior al cambio con 220 cuadros analizados tardo 18 s (12.2 cuadros analizados/s); los niveles y eventos de las personas 1 y 2 fueron identicos a la linea base CPU. Las cifras de camaras simuladas miden inferencia sobre cuadros precargados, sin captura RTSP ni Gemini.

20. **Retroalimentacion local y Telegram opcional.** Las respuestas `confirmado`, `descartado` y `no_concluyente` son revisiones humanas de comportamiento sospechoso, no prueba de un robo. Se guardan con historial en SQLite y JSONL. El bot solo acepta usuarios numericos autorizados dentro de un chat configurado; no envia alertas en modo sombra. Se usa sondeo de la API para funcionar sin servidor publico. La captura de caracteristicas se activa para nuevos incidentes, sin cambiar los niveles ni entrenar automaticamente con un solo video.
21. **Primer clasificador solo en sombra.** Una regresion logistica tabular entrenada fuera de la aplicacion con scikit-learn exporta coeficientes a JSON; la inferencia usa NumPy. Random Forest se mide como referencia, sin activarse. Se separan sesiones completas para evaluar y se exige representacion de ambas clases. El modelo solo puntua alertas ya generadas: no estima falsos negativos ni cambia niveles. Los casos antiguos sin caracteristicas actuales y los `no_concluyente` quedan fuera. Una sesion de VLC repetida no aporta diversidad real aunque cambie el identificador de sesion.
22. **Sonido y secuencia anulada.** El panel avisa con un tono ante alertas nuevas y activas, despues de una interaccion del usuario por la restriccion de audio del navegador. Agrupa alertas que llegan en el mismo sondeo para evitar tonos superpuestos; respeta el silencio guardado localmente. Si una devolucion o la revision visual anula el ocultamiento que genero una alerta, esta queda visible como resuelta y sale del conteo de pendientes. Esta anulacion automatica no se convierte en etiqueta de ML: el operador debe revisar la evidencia. Alertas antiguas sin referencia al ocultamiento conservan su estado previo.
23. **Gemini bajo control del panel.** Un boton activa o desactiva nuevas revisiones sin reiniciar la camara. Se persiste solo el booleano en `data/`; la clave sigue en `.env`. Las consultas ya enviadas no pueden retirarse de la red, pero un resultado posterior al apagado se ignora. Otra accion explicita toma una sola captura de la camara para proponer zonas; el operador ve los poligonos y debe guardarlos. No se aplican zonas sugeridas automaticamente porque un estante, caja o salida mal ubicado cambia la deteccion.
24. **Modelo inicial de zonas.** Las pruebas previas de este proyecto registraron fallos 503 de `gemini-3.8-flash` en la cuenta local, mientras `gemini-3-flash-preview` respondio con imagenes. El analisis de zonas selecciona este ultimo por defecto y deja elegir el modelo configurado. Un fallo muestra su codigo/causa sin incluir el mensaje bruto del SDK, que podria contener detalles de la solicitud. No se afirma que el modelo alternativo funcione siempre: falta una prueba real de zonas en esta camara.
25. **Editor local de zonas.** El operador puede capturar un cuadro desde el panel sin enviarlo fuera del equipo, dibujar zonas rectangulares y mover poligonos o vertices existentes. El mismo editor corrige propuestas de Gemini. Se guarda mediante la validacion actual de camaras; abrir o mover el dibujo no cambia la vigilancia hasta guardar. Los nuevos poligonos son rectangulos para simplificar el gesto, pero las propuestas irregulares conservan sus puntos editables.
26. **Evidencia contextual para Telegram.** Los albumes de alerta usan cuadros de escena completa, con el ID de la persona y su caja, desde antes del alcance hasta la alerta. Esto permite revisar estante, sustituciones y otras personas sin recortar la escena. Gemini conserva sus recortes individuales; el contexto amplio se envia solo por el canal de alerta ya configurado por el operador. Son capturas espaciadas, no un video continuo, y no prueban por si solas una sustraccion.
26. **Retencion de tracks ante oclusion.** `TRACK_BUFFER_FRAMES` queda configurable, con 60 por defecto (hasta 7,5 s a 8 inferencias/s). En `video/video2.mp4`, comparar 30, 60 y 90 cuadros en los primeros 12 s no cambio los IDs ni los eventos: se observaron IDs 1, 2 y 5 y el mismo timeline. Por ello no afirmamos que el valor 60 mejore la precision medida; es un margen para re-asociar una deteccion de persona despues de una oclusion. Puede asociar mal a personas distintas si se cruzan. La continuidad de una mano ya detectada se complementa con flujo optico limitado (decision 28), pero no se detectan manos nuevas ni se transfieren secuencias si ByteTrack cambia el ID.

27. **Modelo Gemini por defecto: `gemini-3-flash-preview`.** En noviembre de 2026, `gemini-3.8-flash` (el defecto anterior) devolvio 503/504 de forma sistematica en la cuenta del usuario; `gemini-3-flash-preview` fue el unico modelo que completo la consulta con imagenes. `zone_suggestions.py` ya lo usaba. Se unifica el default y se aumentan los reintentos ante 503 (3 intentos con espera de 5/10/15 s, en lugar de 2 con 2/4 s). El valor configurable via `GEMINI_MODEL` no cambia. Ver historial completo en `docs/PRUEBAS.md`.

28. **Continuidad temporal de muñeca durante oclusion.** `WristMotionTracker` usa Lucas-Kanade forward-backward sobre la imagen reducida para prolongar una trayectoria de muñeca que YOLO-pose ya observo durante un alcance. Si el punto sigue dentro del estante, extiende `reach_last` hasta que vuelva la pose; no agrega puntuacion, no crea ocultamiento ni afirma que se tomo un producto. Se elige esta evidencia solo para continuidad porque el flujo optico puede adherirse a textura del estante. Requiere clip real de CAM-01 para medir recuperacion y deriva antes de conectarlo a alertas.

29. **Aviso preventivo tras un ocultamiento observado.** Una puntuacion local que ya supera el umbral puede generar alerta aunque Gemini este revisando o en pausa; su dictamen posterior puede anularla. Para una sola secuencia sostenida de alcance seguida de mano junto a bolsillo, pecho o cintura, se emite `posible_ocultamiento` despues de 3 s sin devolucion observada si la puntuacion sigue >= 0,50. El nivel requiere revision humana y no afirma robo ni presencia de producto; una muneca perdida por si sola queda fuera de esta regla. Motivo: antes el evento solitario solia sumar ~0,58 frente al umbral 0,70 y la revision pendiente podia retener una puntuacion alta hasta 45 s. Se espera mayor sensibilidad y mas falsas alarmas; validar con videos de clientes normales y casos reales de la tienda antes de ajustar la regla.

30. **Tracks retenidos no se presentan como personas visibles.** ByteTrack y `TRACK_TTL=10 s` conservan estado para intentar recuperar un ID tras una oclusion, pero antes la API y el overlay mostraban esos tracks como si siguieran detectados. El panel y la evidencia en vivo solo muestran tracks vistos en el ultimo segundo; el estado interno sigue retenido y puede recuperarse si la persona reaparece. Esto corrige la presentacion de tracks desactualizados, no los falsos positivos de YOLO. No cambia alertas ni umbrales. Falta evaluar el limite de un segundo en oclusiones reales.

31. **Ajustes de persona por camara para vistas cenitales.** Se permite sobrescribir `YOLO_CONF` y el alto minimo de la caja desde la configuracion de una sola camara; los valores vacios conservan exactamente los defaults existentes (0.40 y 10%). En vistas superiores se puede probar menor confianza y 3-5% de alto para admitir detecciones pequeñas. Esto tambien aumenta falsos positivos y no convierte YOLO-pose en detector de cabezas. Guardar reinicia el pipeline de esa camara. La evaluacion del detector especializado de cabezas queda pendiente de cuadros etiquetados del angulo real.

## Alternativas probadas o descartadas

- **Diferencia de pixeles en el estante** (cambio entre fotogramas): en la licoreria (640x360, ojo de pez, botellas diminutas) el cambio estaba al nivel del ruido. No sirve a esa resolucion.
- **Poligono de estanteria "hasta el piso"** en camara alta: la persona parada tiene las manos "dentro" por perspectiva. Dibujar la zona solo donde llega la mano al alcanzar.
- **Detector de objetos (COCO, YOLO-World) para producto/bolso en mano:** no se pudo probar (no hubo red para bajar pesos). Sigue siendo la mejora con mas potencial; ver pendientes.
- **Discriminar mujer/hombre solo con la pose:** no es posible de forma fiable. En los datos, las manos juntas a la cintura del hombre (8-10 s) se parecen a las de ella (15-16 s). Por eso el juez es Gemini.
- **Escribir el `.env` del usuario o cambiar su modelo:** no se toco; solo se uso `GEMINI_MODEL` por variable de entorno en la prueba.

## Pendientes (priorizados)

1. Estado **"descartada/devuelta"** en el panel: hoy una alerta emitida y luego anulada sigue en la lista (en `video2`, la "posible" del hombre).
2. **Validar con video real de la tienda** (2 camaras) en `SHADOW_MODE`; ajustar zonas, `CONCEAL_*`, `RETURN_WINDOW_SECONDS` y umbrales locales. Hoy solo hay un clip util.
3. **Validar el editor visual de zonas con operadores reales** y agregar gestos para poligonos nuevos irregulares si los rectangulos resultan insuficientes.
4. **Deteccion de producto/bolso** con un segundo modelo (primero vocabulario abierto, luego ajustado a los productos de la tienda), sobre recortes, no el cuadro completo.
5. **Seguimiento entre camaras** (hoy el ID se pierde al cambiar de camara u ocluirse).
6. **Gemini:** elegir modelo estable (el por defecto `gemini-3.8-flash` dio 503 en las pruebas), y evaluar cola/reintentos mas largos.
7. Los mensajes de motivo del respaldo local tienen redaccion simple; revisarlos si se muestran a usuarios finales.

## Origen

El repositorio nacio de **Vigilia** (monitor de somnolencia en guardias de un hospital). Se conservaron camara, optimizador, tracking, `GeminiJudge` (motor), alertas y panel; se reescribieron heuristica, decision, prompt, zonas y pruebas. El historial git de Vigilia se borro a proposito.
