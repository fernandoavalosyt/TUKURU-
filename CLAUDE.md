# Tukuru - contexto del proyecto para el modelo

Lee esto primero. Detalle en `docs/` (ver "Documentos"). El idioma del proyecto y de la conversacion con el usuario es **espanol**.

## Que es

Sistema de vision artificial que detecta **robo hormiga** en una tienda de abarrotes (cliente toma un producto del estante, lo oculta en bolsillo/ropa/bolso y sale sin pagar) y emite una alerta con foto de evidencia para que una **persona** la revise. No acusa: señala comportamiento sospechoso.

- Una sola app Python, CPU, sin base de datos ni contenedores. 2 camaras, ampliable a 3 (`MAX_CAMERAS`, tope 4).
- Origen: se construyo a partir de **Vigilia** (deteccion de somnolencia en guardias). Todo el concepto de somnolencia fue reemplazado; no queda codigo de ojos/bostezos/postura de sueno. Si ves referencias a "vigilia" o "guardia", son restos y se pueden corregir.
- Funciona asi: YOLO-pose (persona + 17 puntos + ID) -> heuristica local por mano (alcance -> ocultamiento -> salida) -> **Gemini revisa un clip** solo ante un posible ocultamiento -> decision -> alerta -> panel web.

## Idea clave (no la pierdas)

Con solo la pose **no se ve el producto**. "Meter algo al bolso" y "mirar el celular con las manos juntas" se parecen. Por eso:
- La heuristica local decide **cuando** vale la pena consultar a Gemini y da un respaldo si Gemini no esta disponible (con umbrales mas altos).
- **Gemini es quien separa** a quien oculta de quien solo toma y devuelve.
- Toda mejora debe respetar: pose barata para disparar, juez visual para confirmar.

## Estado actual (resumen)

- Funciona de punta a punta sobre `video/video2.mp4`: la mujer (ID 1) termina en `probable_sustraccion` confirmada por Gemini; el hombre (ID 2), que toma y devuelve, queda sin alerta final (tuvo una "posible" transitoria que se anulo por devolucion).
- Probado con **un** clip y **dos** consultas a Gemini: no esta calibrado ni validado en general.
- Pendiente conocido: estado "descartada" en el panel para alertas anuladas, editor visual de zonas, deteccion de producto/objeto (no hay red para bajar modelos en el entorno de desarrollo), seguimiento entre camaras, calibracion con video real de la tienda. Ver `docs/DECISIONES.md`.

## Comandos

Entorno nuevo (equipo): `python scripts/preparar_entorno.py --solo-pruebas` (o sin flag para todo). Detalle en `docs/ENTORNO.md`.

```
pip install -r requirements.txt            # requirements-test.txt si solo vas a correr pruebas
copy .env.example .env                     # (Windows) y completar GEMINI_API_KEY
python -m tukuru                           # app + panel en http://127.0.0.1:5000
python -m unittest discover -s tests       # pruebas (unittest, no pytest)
python -m tukuru.replay VIDEO --zones ZONAS.json --fps 10 --dashboard --out SALIDA.mp4
python -m tukuru.replay ... --gemini --save-clips CARPETA    # revision REAL con Gemini (ver reglas)
```

`replay` reproduce un video por la misma logica que el pipeline en vivo y puede dibujar el panel sobre el video. Es la herramienta para calibrar. Ver `docs/PRUEBAS.md`.

## Mapa del codigo (`tukuru/`)

| Archivo | Responsabilidad |
|---|---|
| `__main__.py`, `app.py` | Arranque; `Tukuru` administra pipelines de camara en caliente |
| `config.py` | `Settings.from_env()` (validado, errores acumulados), `CameraConfig`, `validate_camera`, zonas, `CameraStore` (`data/cameras.json`) |
| `zones.py` | `ZoneSet`: poligonos normalizados 0-1 por tipo (`estanteria`, `caja`, `salida`, `personal`) |
| `camera.py`, `optimizer.py` | Captura RTSP/HTTP con reconexion; reduce a <=640 px, limita fps, modo nocturno |
| `vision.py` | `PoseTracker` (YOLO-pose + ByteTrack). Indices COCO de munecas/caderas |
| `heuristics.py` | **Nucleo**: `TrackState`, `update_track`, secuencia por mano, puntuacion con decaimiento |
| `decision.py` | `decide()` (niveles), `apply_verdict()` (aplica dictamen de Gemini) |
| `gemini.py` | `GeminiJudge`: prompt, esquema `Verdict`, cooldown, tope por hora, reintentos |
| `clips.py` | `store_crop` / `select_clip`: recortes por persona y clip que se envia a Gemini |
| `alerts.py` | `AlertManager`: deduplicacion, evidencia, JSONL, webhook, modo sombra |
| `pipeline.py` | Hilo por camara que orquesta todo |
| `render.py`, `dashboard.py` | Overlay del video / panel dibujado en OpenCV (solo para replay) |
| `replay.py` | Reproduccion de video con la heuristica (+ Gemini opcional) |
| `web.py`, `static/index.html` | API Flask y panel |

Flujo y estructuras en `docs/ARQUITECTURA.md`.

## Convenciones del codigo

- Comentarios, mensajes y nombres de dominio en espanol, **sin acentos** en codigo y cadenas (el panel dibujado con OpenCV solo admite ASCII). README y docs si llevan acentos.
- Estilo: tipado con `from __future__ import annotations`, dataclasses, lineas de hasta ~120 col. Igualar el estilo vecino.
- El tiempo se **inyecta** (`now` como parametro) en `update_track`/`decide`: asi el replay y los tests son deterministas. No uses `time.time()` dentro de la logica de decision.
- Nuevo parametro configurable = campo en `Settings` + lectura en `from_env` (con rango) + `.env.example` + tabla del README + prueba.
- Pruebas con escenas sinteticas de puntos del cuerpo (ver `Scene` en `tests/test_tukuru.py`): armar la secuencia (alcance, bolsillo, salida) y comprobar eventos/niveles. Agrega una prueba por cada regla nueva.
- Coordenadas de puntos y zonas estan en el cuadro **reducido** (<=640 px); los recortes para Gemini se sacan del cuadro **original** usando la escala.

## Reglas de trabajo (importantes)

1. **No envies imagenes ni video a servicios externos (Gemini) sin autorizacion explicita del usuario** para esa prueba. Ya se autorizo una vez para `video2`; no se extiende a otros videos ni a produccion.
2. **Nunca muestres ni subas secretos.** `.env` tiene una `GEMINI_API_KEY` real y restos de camaras de Vigilia (`CAMERA_1_*`). Para inspeccionarlo, imprime solo nombres y longitudes. Esta en `.gitignore` (tambien `data/`, `models/`, `*.pt`).
3. Se honesto con los resultados: si algo falla o no se pudo probar, dilo. No ajustes umbrales para "hacer que un video salga bien" sin decirlo (riesgo de sobreajuste: hay un solo clip util).
4. Cambios de comportamiento que afecten alertas (niveles, umbrales, reglas) deben quedar anotados en `docs/DECISIONES.md` con su motivo.
5. No hagas commit ni push sin que el usuario lo pida. El repositorio git es nuevo y casi vacio a proposito (se borro el historial de Vigilia).

## Trampas conocidas

- Entorno de desarrollo (Windows, Git Bash): las descargas de modelos desde GitHub se colgaron; solo existe `models/yolov8n-pose.pt`. No asumas que puedes bajar otros pesos.
- En Bash, un heredoc con comillas triples `'''` puede romper el comando: escribe un script a un archivo y ejecutalo.
- Gemini en la cuenta del usuario: `gemini-3.8-flash` (el modelo por defecto) y `3.7`/`flash-latest` dieron 503/timeout; `gemini-2.5-flash` da 404; `gemini-3.6-flash` agoto cuota; `gemini-3-flash-preview` funciono. Usa `GEMINI_MODEL` para cambiarlo y revisa `docs/PRUEBAS.md`.
- Los modelos 3.x gastan tokens "pensando": `max_output_tokens` bajo corta el JSON.
- Una persona parada junto al estante tiene una mano "dentro" de la zona por perspectiva: por eso el seguimiento es **por mano** y caminar no cuenta como alcance.
- Si el seguimiento cambia el ID (oclusion), la secuencia en curso se pierde.

## Documentos

- `README.md`: uso, instalacion, configuracion (orientado al usuario final).
- `docs/ARQUITECTURA.md`: flujo, estructuras de datos, reglas de decision, API, archivos de datos.
- `docs/DECISIONES.md`: por que se hizo cada cosa, alternativas descartadas, pendientes.
- `docs/ENTORNO.md`: preparar el entorno en otra PC, que necesita cada tarea, problemas frecuentes.
- `docs/PRUEBAS.md`: videos de prueba, verdad de terreno, resultados, como reproducir y calibrar.
