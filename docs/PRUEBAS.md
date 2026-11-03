# Pruebas, videos y calibracion

## Pruebas unitarias

```
python -m unittest discover -s tests
```

37 pruebas, sin red ni modelos. Cubren validacion de camaras/zonas/ajustes y escenas sinteticas de puntos del cuerpo (`Scene` en `tests/test_tukuru.py`: persona de pie 640x360, alcance en `REACH`, bolsillo en `POCKET`, zonas `ZONES`). Una prueba por regla: alcance, roce, manos al bolsillo sin alcance, ventana vencida, muneca perdida, mirada lateral, personal, mano por mano, manos juntas, devolucion (mismo sitio / otro sitio / fuera de ventana), caminar, decision con/sin Gemini, pasar por caja, repeticion de alertas.

## Videos de prueba (carpeta `video/`)

| Archivo | Uso | Notas |
|---|---|---|
| `video2.mp4` | **Video de referencia** (768x432, 30 fps, 22 s) | Tienda de conveniencia, camara alta y nitida |
| `zonas_video2.json` | Zonas de `video2` | 2 estanterias + salida inferior. No hay caja visible |
| `Mecheras-Tecnica-de-SnapYT.App.mp4` | **Descartado por el usuario** ("ignoralo") | Licoreria 640x360 ojo de pez; casi nada se ve. No usar como referencia |
| `zonas_licoreria.json` | Zonas del video descartado | Solo historico |
| `clips_gemini_video2/` | Recortes enviados a Gemini y sus dictamenes | Para auditar que se envio |
| `video2_dashboard_gemini.mp4` | Salida del panel con Gemini | Resultado final de la prueba |

### Verdad de terreno de `video2` (dada por el usuario y verificada cuadro a cuadro)

- **Mujer** (chaqueta azul, bolso negro cruzado al frente; track ID 1): **roba**. Toma un producto pequeño en empaque transparente (~14,0 s), lo sostiene a la altura del pecho (14,3-14,9), junta ambas manos sobre el bolso a la cintura (15,2-15,8), ya sin producto a los 15,8 s; luego manos cruzadas sobre el bolso (16,4-18,5) y se va sin volver al estante. Sale por la parte inferior (~19,6 s).
- **Hombre** (suéter beige, mochila; track ID 2, el ID cambia por oclusion): **solo toma y vuelve a dejar**. Revisa productos con las manos y alcanza arriba repetidas veces entre 12,5 y 17,2 s. Se va con ella al final.

### Resultados de referencia

Con zonas de `zonas_video2.json`, `--fps 10`, `gemini-3-flash-preview`:

| Momento | Persona | Resultado |
|---|---|---|
| 12,2 s | Hombre | Gemini `ocultamiento` 0,90 -> alerta "posible" (**falso positivo**) |
| 14,9 s | Hombre | Devolucion (misma mano, mismo sitio) -> anulado, queda en `observacion` |
| 17,7 s | Mujer | Gemini `ocultamiento` 0,85 `oculta_en_bolso` (evento de 14,7 s) -> "posible" |
| 19,6 s | Mujer | Sale sin pasar por caja -> **`probable_sustraccion`** |

Solo heuristica local (sin Gemini): ambos terminan en `probable_sustraccion` (la pose no los separa). Ese es el motivo de ser de Gemini en el diseño.

## Reproducir

```
# solo heuristica local (no usa red)
python -m tukuru.replay video/video2.mp4 --zones video/zonas_video2.json --fps 10 --dashboard --out salida.mp4

# con Gemini de verdad (ENVIA recortes a la API; pedir autorizacion al usuario antes)
set GEMINI_MODEL=gemini-3-flash-preview
set GEMINI_TIMEOUT_SECONDS=90
python -m tukuru.replay video/video2.mp4 --zones video/zonas_video2.json --fps 10 --gemini --save-clips carpeta --dashboard --out salida.mp4
```

(En PowerShell: `$env:GEMINI_MODEL="..."`.) Imprime la linea de tiempo de eventos (alcance, ocultamiento, mirada, salida, devolucion, nivel, gemini). El replay usa `gemini_cooldown=0`, espera cada respuesta y no modela la latencia real.

## Gemini: lo observado (cuenta del usuario, noviembre 2026)

- `gemini-3.8-flash` (defecto), `3.7-flash`, `flash-latest`: 503/504 o timeout con y sin imagenes.
- `gemini-2.5-flash`: 404 (aparece en la lista pero no responde).
- `gemini-3.5-flash`: texto OK, con imagenes dio 503.
- `gemini-3.6-flash`: funciona, pero llego a 429 (cuota) tras pocas consultas.
- `gemini-3-flash-preview`: el que funciono (~12-14k tokens por consulta de 12 recortes).
- Dos errores de configuracion ya resueltos: `max_output_tokens` demasiado bajo cortaba el JSON; clips cortos/pocos cuadros daban `normal`.
- Esto cambia con el tiempo: antes de depurar el codigo, prueba una llamada de texto de una palabra con cada modelo para descartar el servicio.

## Como agregar un video y calibrar

1. Mirar el video cuadro a cuadro (hojas de contacto con OpenCV) y anotar la verdad de terreno con tiempos.
2. Dibujar zonas: `estanteria` solo donde llega la **mano** al alcanzar; `salida`/`caja` donde pisan los pies. Coordenadas normalizadas 0-1 sobre el cuadro reducido.
3. Correr el replay **sin Gemini** y revisar la linea de tiempo contra la verdad de terreno. Mirar el video anotado para ver si las munecas y los IDs son correctos.
4. Si hay eventos falsos o perdidos, diagnosticar con los puntos de pose (se pueden volcar con `PoseTracker`) antes de tocar umbrales.
5. Correr con `--gemini` (con autorizacion) y revisar `--save-clips`: ¿el clip incluye el momento clave y se ven las manos?
6. En tienda real: `SHADOW_MODE=true` varios dias, revisar falsos positivos, ajustar, y solo entonces activar alertas.

Cuidado con el sobreajuste: hay un solo clip util. Cualquier regla nueva debe justificarse por comportamiento general (y tener su prueba sintetica), no por "que este video salga bien".
