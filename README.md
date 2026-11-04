# Tukuru

Sistema de visión artificial que detecta **robo hormiga** en una tienda de abarrotes: un cliente toma un producto del estante, lo oculta en el bolsillo, bajo la ropa o en un bolso, y sale sin pagarlo. Emite una alerta con evidencia fotográfica para que una persona la revise.

Es una versión de producto mínimo viable: una sola aplicación Python, sin bases de datos ni contenedores, que funciona solo con CPU. Está pensado para 2 cámaras, ampliable a 3 (`MAX_CAMERAS`, máximo 4).

> Tukuru **no acusa**: señala comportamiento sospechoso y guarda evidencia. La confirmación y cualquier acción son siempre de una persona.

## Cómo funciona

```
Cámara (RTSP / RTSPS / HTTP)
   |
   v
camera.py       Captura con reconexión; siempre entrega el frame más reciente
   |
   v
optimizer.py    Limita los fps, reduce la resolución y mejora la imagen con poca luz
   |
   v
vision.py       YOLOv8n-pose: persona, ID (ByteTrack) y 17 puntos del cuerpo (muñecas, caderas...)
   |
   v
heuristics.py   Secuencia por persona: alcance al estante -> mano al cuerpo -> salida sin caja
   |
   v
gemini.py       Solo ante un posible ocultamiento: clip de ~8 recortes + resumen numérico
   |
   v
decision.py     Combina el estado local y el dictamen de Gemini
   |
   v
alerts.py       Alerta, foto anotada, registro, webhook opcional y reconocimiento
   |
   v
web.py          Panel web en tiempo real
```

### Zonas de la tienda

Cada cámara puede tener polígonos (coordenadas normalizadas de 0 a 1) de cuatro tipos:

| Zona | Uso |
|---|---|
| `estanteria` | Cara del estante. Se mide con las **muñecas** de la persona |
| `caja` | Zona de pago. Se mide con los **pies**; permanecer `CHECKOUT_MIN_SECONDS` cuenta como "pasó por caja" |
| `salida` | Puerta. Se mide con los pies |
| `personal` | Área de empleados. Quien permanece ahí 2 s queda excluido del análisis |

Se pegan como JSON en el campo "Zonas" del panel, por ejemplo:

```
{"estanteria": [[[0.0, 0.2], [0.4, 0.2], [0.4, 0.8], [0.0, 0.8]]],
 "salida":     [[[0.8, 0.6], [1.0, 0.6], [1.0, 1.0], [0.8, 1.0]]]}
```

Cada tipo admite hasta 8 polígonos. Sin zona `estanteria` no hay detección de alcances; sin `salida` el nivel "probable" solo lo alcanza un dictamen de Gemini con salida confirmada.

### La heurística local

El robo dura 1 a 3 segundos, así que no se mide un estado sostenido sino una **secuencia**, que se sigue **mano por mano** (una persona parada junto al estante suele tener una mano en la zona y la otra en la cadera):

1. **Alcance**: una muñeca entra a la zona `estanteria` entre 0,25 y 6 s, con la persona parada (si camina, solo pasa junto al estante). Menos tiempo es un roce; más es revisar o reponer.
2. **Ocultamiento**: en los `CONCEAL_WINDOW_SECONDS` siguientes, esa mano queda al menos `CONCEAL_MIN_SECONDS` junto a la cadera/bolsillo/bolso, al pecho o abdomen, o la muñeca se pierde junto al cuerpo; o bien **las dos manos se juntan a la cintura**. Una mano al cuerpo **sin** alcance previo no cuenta.
3. **Devolución**: si la misma mano vuelve al mismo sitio del estante en `RETURN_WINDOW_SECONDS`, se interpreta como "tomó y lo devolvió" y el ocultamiento se anula.
4. **Salida**: los pies entran a la zona `salida` con un ocultamiento sin descartar.
5. **Señales de apoyo**: mirar a los lados repetidamente (giros de cabeza) y repetir ocultamientos en pocos minutos.

Cada evento suma a una puntuación que decae con el tiempo (vida media de unos 2 minutos). Con solo la pose **no se ve el producto**: "meter algo al bolso" y "mirar el celular con las manos juntas" se parecen mucho. Por eso el objetivo local es decidir **cuándo** consultar a Gemini y dar un respaldo si no está disponible; quien separa a quien robó de quien solo miró es Gemini.

### Uso de Gemini

Gemini no recibe el video completo. 3 s después del ocultamiento se envían hasta `GEMINI_FRAMES` recortes de esa persona (desde `CLIP_SECONDS` s antes del ocultamiento hasta ese instante, ampliados si la persona está lejos), con un resumen numérico. Dictamina `ocultamiento`, `normal` o `indeterminado`, e indica la acción (oculta en ropa o bolso, pone en canasta, lleva en mano, devuelve al estante, sin toma). Si descarta el evento con confianza suficiente, ese ocultamiento y todo lo anterior se anulan; un ocultamiento nuevo se vuelve a evaluar. Hay límites de gasto: cooldown por persona, tope por hora y pausa automática ante límite de cuota.

Si Gemini no está disponible, el sistema sigue con la heurística local, que exige una puntuación más alta para alertar.

## Niveles

| Nivel | Significado | Quién lo decide | ¿Alerta? |
|---|---|---|---|
| `observacion` | Comportamiento inusual o verificación en curso | Heurística local | No (solo panel) |
| `posible_ocultamiento` | Tomó algo y lo ocultó | Gemini con confianza ≥ `MIN_CONF_CONFIRM`, o puntuación local ≥ `LOCAL_POSSIBLE_SCORE` sin Gemini | Sí |
| `probable_sustraccion` | Ocultó y salió | Lo anterior más cruce de `salida`. Sin Gemini, además, no haber pasado por caja y puntuación ≥ `LOCAL_PROBABLE_SCORE` | Sí |
| `sistema` | Cámara sin señal por más de 60 s | Automático | Sí |

Pasar por caja **no anula** un ocultamiento confirmado por Gemini: ocultar un producto y pagar otros es justo el patrón típico. Solo evita el nivel "probable" cuando la decisión es únicamente local.

## Requisitos

- Python 3.10 o superior.
- Equipo con CPU convencional; no requiere GPU.
- Cámara IP con acceso RTSP o HTTP, o un archivo de video para pruebas.
- Clave de API de Google AI Studio para activar Gemini (opcional pero recomendada).

## Instalación

Guía para trabajar en equipo y problemas frecuentes: [`docs/ENTORNO.md`](docs/ENTORNO.md). Atajo: `python scripts/preparar_entorno.py` (o con `--solo-pruebas` para solo correr las pruebas).

```
git clone <url-del-repositorio>
cd <carpeta>
pip install -r requirements.txt
copy .env.example .env      (Windows)
cp .env.example .env        (Linux / macOS)
```

Los pesos de YOLO se descargan automáticamente en `models/` la primera vez.

## Uso

```
python -m tukuru
```

Abra el panel: http://127.0.0.1:5000

El panel separa **Monitoreo** (cámaras en vivo, personas e indicadores, alertas) de
**Configuración** (alta, edición y conexión de cámaras). Las alertas se pueden filtrar
por pendientes o reconocidas; pulsa la miniatura para ampliar la evidencia.
**Eliminar** quita una alerta y **Limpiar vista** quita todas, conservando el registro
y las fotos. Las alertas retiradas de la vista no reaparecen al reiniciar.

### Agregar una cámara

Desde el panel, en "Agregar / editar cámara", complete IP, usuario, contraseña y marca, pegue las zonas y use "Probar conexión" antes de guardar. También puede definirla en `.env` (`CAMERA_1_URL`, `CAMERA_1_USER`, ...; ver `.env.example`).

Para probar sin cámara: `CAMERA_1_PROTOCOL=file` y `CAMERA_1_PATH=C:\ruta\al\video.mp4`.

## Configuración

Todas las opciones están en `.env.example`. Las principales:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `GEMINI_API_KEY` | vacío | Clave de Google AI Studio |
| `GEMINI_MAX_CALLS_PER_HOUR` | 120 | Tope de consultas por hora |
| `GEMINI_FRAMES` | 12 | Recortes por consulta |
| `INFERENCE_FPS` | 8 | Análisis por segundo con personas en escena |
| `IDLE_FPS` | 2 | Análisis por segundo con la tienda vacía |
| `MAX_FRAME_SIDE` | 640 | Lado máximo del frame analizado |
| `CONCEAL_WINDOW_SECONDS` | 8 | Ventana tras el alcance en que la mano al cuerpo cuenta |
| `CONCEAL_MIN_SECONDS` | 0.6 | Permanencia de la mano junto al cuerpo |
| `RETURN_WINDOW_SECONDS` | 6 | Si la misma mano vuelve al mismo sitio del estante en este tiempo, se anula |
| `CHECKOUT_MIN_SECONDS` | 3 | Permanencia en caja para considerar que pagó |
| `SHADOW_MODE` | false | Registra eventos sin avisar ni enviar webhook |
| `MAX_CAMERAS` | 3 | Número máximo de cámaras |
| `CREDENTIALS_STORE` | auto | Dónde se guardan las contraseñas de las cámaras: `auto` (llavero del sistema si existe, si no `data/cameras.json`), `keyring` (exige llavero) o `file` |
| `ALERT_WEBHOOK_URL` | vacío | URL que recibe cada alerta como JSON |

Los valores se validan al iniciar; si alguno es inválido, el programa lista todos los errores juntos y no arranca.

## Seguridad y privacidad

- El panel escucha solo en `127.0.0.1` por defecto. `WEB_HOST=0.0.0.0` exige definir `DASHBOARD_USER` y `DASHBOARD_PASSWORD`. El panel no cifra el tráfico: use una red interna o un proxy con HTTPS.
- Las contraseñas de las cámaras no se devuelven por la API ni se escriben en los registros.
- Las cámaras se agregan desde el panel (no hace falta `.env`). Sus contraseñas van al llavero del sistema (Administrador de credenciales de Windows, Keychain, Secret Service) y `data/cameras.json` no las contiene. Sin llavero (servidor sin escritorio) quedan en ese archivo con permisos 600: proteja `data/`. Un `cameras.json` antiguo con contraseñas en texto plano se migra solo al arrancar.
- Las imágenes de clientes se envían a Gemini. Confirme los requisitos legales de videovigilancia y datos personales de su país (aviso en el local, retención, acceso) antes de usarlo en producción.
- No suba `.env`, `data/` ni `models/` al repositorio; ya están en `.gitignore`.

## Calibración

Los umbrales son un punto de partida y **no están calibrados con video real**. Active `SHADOW_MODE=true` unos días: registra eventos con su foto sin avisar a nadie, y permite revisar cuáles fueron falsos positivos (clientes que guardan el celular o la cartera, que ponen algo en su canasta, empleados reponiendo) antes de activar las alertas. Ajuste las zonas, `CONCEAL_*` y los umbrales locales con esos casos. Funciona mejor con la cámara de frente al estante, a la altura del pecho o ligeramente elevada.

### Reproducir un video

Para calibrar sin cámara, `tukuru.replay` pasa un video por la misma heurística (sin Gemini) e imprime la línea de tiempo de eventos:

```
python -m tukuru.replay video.mp4 --zones zonas.json --out salida.mp4
```

`zonas.json` usa el mismo formato que el panel (ver `video/zonas_video2.json`). `salida.mp4` muestra zonas, esqueleto y nivel de cada persona. Opciones útiles:

- `--dashboard`: el video sale como el panel web (video, tabla de personas y alertas con evidencia).
- `--gemini`: cada posible ocultamiento se revisa con Gemini de verdad. **Envía recortes del video a la API** con la clave de `.env`; `--save-clips carpeta` guarda copia de lo enviado y del dictamen.

## Estructura del proyecto

```
tukuru/
  __main__.py     Punto de entrada
  config.py       Configuración, validación, cámaras y zonas
  zones.py        Polígonos de estantería, caja, salida y personal
  camera.py       Conexión y captura de video
  optimizer.py    Reducción de resolución y mejora con poca luz
  vision.py       YOLO-pose con tracking
  heuristics.py   Secuencia alcance -> ocultamiento -> salida (por mano) y puntuación
  clips.py        Recortes por persona y clip que se envía a Gemini
  dashboard.py    Dibuja el panel sobre un video (para replay)
  gemini.py       Cliente de Gemini Vision con control de gasto
  decision.py     Reglas de decisión final
  alerts.py       Alertas, evidencia, registro y modo sombra
  pipeline.py     Flujo completo por cámara
  render.py       Dibujo de zonas, ID, esqueleto y nivel sobre el video
  replay.py       Reproduce un video con la heurística para calibrar
  app.py          Administración de cámaras en ejecución
  web.py          API y panel web
  static/         Interfaz del panel
tests/            Pruebas unitarias
data/             Cámaras, alertas y evidencia (se crea al ejecutar)
models/           Pesos descargados (se crea al ejecutar)
```

## Pruebas

### Experimento opcional de objetos (solo replay)

Con un detector nano local en `models/yolov8n.pt`:

```powershell
python -m tukuru.replay video/video2.mp4 --zones video/zonas_video2.json --fps 10 --object-model models/yolov8n.pt --object-report data/prueba_objetos/objetos.json --out data/prueba_objetos/video2_objetos.mp4
```

La carpeta de salida del informe se crea automáticamente. Los pesos se obtienen de los
[assets oficiales de Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt).
No se requieren paquetes adicionales. Sin `--object-model` no se carga ni ejecuta el detector.
El experimento observa objetos junto a manos, estantes, cuerpo y contenedores; exporta
las detecciones, transiciones y tiempo de inferencia en JSON y dibuja cajas en el video.
No cambia puntuaciones, alertas ni el contexto de Gemini. El comando anterior es local y no usa Gemini.
Detalles y límites en [docs/PRUEBAS.md](docs/PRUEBAS.md).

```
python -m unittest discover -s tests
```

## Limitaciones conocidas

- Con solo la pose no se ve el producto: el filtro de falsos positivos (producto en canasta, objetos propios) depende de Gemini. Sin Gemini habrá más falsos positivos.
- Una persona que sale del encuadre de una cámara y entra a otra tiene un ID nuevo; la secuencia no se sigue entre cámaras. Conviene que la zona `estanteria` y la `salida` queden en la misma cámara, o cubrir el recorrido con zonas por cámara.
- Una oclusión larga puede cambiar el ID de la persona y reiniciar su secuencia.
- Las zonas se pegan como JSON; falta un editor visual sobre el video.
- En cámaras altas o de ojo de pez, la mano sobre el polígono de la estantería no implica contacto: una persona parada frente al estante tiene las manos "dentro" de la zona por perspectiva. Dibuje la zona solo sobre la parte donde la mano llega al alcanzar, no hasta el piso.
- Si la persona queda tapada por una góndola, el seguimiento se corta y la secuencia en curso se pierde.
- Los umbrales no están calibrados con video de la tienda real.
