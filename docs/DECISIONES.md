# Decisiones, alternativas descartadas y pendientes

Registro de **por que** el sistema es como es. Si cambias una regla que afecte alertas, agrega aqui la decision y su motivo.

## Decisiones

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

## Alternativas probadas o descartadas

- **Diferencia de pixeles en el estante** (cambio entre fotogramas): en la licoreria (640x360, ojo de pez, botellas diminutas) el cambio estaba al nivel del ruido. No sirve a esa resolucion.
- **Poligono de estanteria "hasta el piso"** en camara alta: la persona parada tiene las manos "dentro" por perspectiva. Dibujar la zona solo donde llega la mano al alcanzar.
- **Detector de objetos (COCO, YOLO-World) para producto/bolso en mano:** no se pudo probar (no hubo red para bajar pesos). Sigue siendo la mejora con mas potencial; ver pendientes.
- **Discriminar mujer/hombre solo con la pose:** no es posible de forma fiable. En los datos, las manos juntas a la cintura del hombre (8-10 s) se parecen a las de ella (15-16 s). Por eso el juez es Gemini.
- **Escribir el `.env` del usuario o cambiar su modelo:** no se toco; solo se uso `GEMINI_MODEL` por variable de entorno en la prueba.

## Pendientes (priorizados)

1. Estado **"descartada/devuelta"** en el panel: hoy una alerta emitida y luego anulada sigue en la lista (en `video2`, la "posible" del hombre).
2. **Validar con video real de la tienda** (2 camaras) en `SHADOW_MODE`; ajustar zonas, `CONCEAL_*`, `RETURN_WINDOW_SECONDS` y umbrales locales. Hoy solo hay un clip util.
3. **Editor visual de zonas** sobre el video en el panel (hoy se pega JSON).
4. **Deteccion de producto/bolso** con un segundo modelo (primero vocabulario abierto, luego ajustado a los productos de la tienda), sobre recortes, no el cuadro completo.
5. **Seguimiento entre camaras** (hoy el ID se pierde al cambiar de camara u ocluirse).
6. **Gemini:** elegir modelo estable (el por defecto `gemini-3.8-flash` dio 503 en las pruebas), y evaluar cola/reintentos mas largos.
7. Los mensajes de motivo del respaldo local tienen redaccion simple; revisarlos si se muestran a usuarios finales.

## Origen

El repositorio nacio de **Vigilia** (monitor de somnolencia en guardias de un hospital). Se conservaron camara, optimizador, tracking, `GeminiJudge` (motor), alertas y panel; se reescribieron heuristica, decision, prompt, zonas y pruebas. El historial git de Vigilia se borro a proposito.
