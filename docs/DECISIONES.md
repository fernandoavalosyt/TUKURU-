# Decisiones, alternativas descartadas y pendientes

Registro de **por que** el sistema es como es. Si cambias una regla que afecte alertas, agrega aqui la decision y su motivo.

## Decisiones

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
