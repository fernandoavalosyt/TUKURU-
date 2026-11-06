# Preparar el entorno (trabajo en equipo)

Una persona nueva clona el repositorio y debe poder correr las pruebas sin adivinar nada. El codigo llega con `git clone`; **no llegan** las dependencias de Python, el modelo YOLO (`*.pt` esta en `.gitignore`) ni el `.env` (secretos).

## Un comando

Desde la raiz del repositorio:

```
python scripts/preparar_entorno.py --solo-pruebas   # rapido: numpy + opencv (~100 MB)
python scripts/preparar_entorno.py                  # completo con PyTorch CPU
python scripts/preparar_entorno.py --gpu            # completo con PyTorch CUDA para NVIDIA
```

Crea `.venv`, instala dependencias, crea `.env` desde `.env.example` (sin pisar uno existente) y baja el modelo. Se puede repetir. `--gpu` consulta `nvidia-smi`, elige la rueda CUDA compatible con el driver y comprueba `torch.cuda.is_available()`. La descarga puede ocupar varios GB. Sin `--gpu`, instala la rueda CPU; el resto del equipo no necesita NVIDIA.

Despues, activar el entorno (o usar su Python directamente):

```
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
python -m unittest discover -s tests
```

## Que necesita cada tarea

| Tarea | Instalar | Extra |
|---|---|---|
| Pruebas unitarias | `requirements-test.txt` (numpy, OpenCV, Flask, requests y Pydantic) | nada: ni red, ni modelo, ni clave |
| Replay local (`python -m tukuru.replay ...`) | `requirements.txt` completo | `models/yolov8n-pose.pt` |
| Replay con Gemini (`--gemini`) | `requirements.txt` | `GEMINI_API_KEY` propia en `.env`; **envia imagenes a la API: pedir autorizacion** |
| Aplicacion (`python -m tukuru`) | `requirements.txt` | modelo + camaras (panel o `.env`) |

En el panel, **Activar Gemini** permite alternar el juez sin reiniciar; requiere
`GEMINI_API_KEY` en `.env` y envia recortes cuando haya un posible incidente. La
preferencia se conserva en `data/gemini_state.json` y tiene prioridad sobre
`GEMINI_ENABLED` al siguiente arranque. El boton **Sugerir zonas con Gemini** es
independiente del juez de incidentes: envia una captura solo al pulsarlo y no guarda
la propuesta hasta que el operador la revise y guarde la camara. **Dibujar zonas con
mouse** obtiene una captura local sin llamar a Gemini: arrastre para crear rectangulos,
mueva el interior para trasladarlos o arrastre sus puntos para ajustarlos. Guarde la
camara para aplicar los cambios.

## Revision de alertas por Telegram

Telegram es opcional. Cree un bot con BotFather y guarde el token solo en `.env` como
`TELEGRAM_BOT_TOKEN`. Con Tukuru detenido, envie `/start` al bot y ejecute
`python scripts/telegram_ids.py`: muestra el ID numerico del chat y del usuario sin
imprimir el token ni el contenido de los mensajes. Configure en `.env`
`TELEGRAM_CHAT_ID` y `TELEGRAM_ALLOWED_USER_IDS` (uno o varios IDs de operadores,
separados por coma). Reinicie Tukuru. Si el bot ya usa un webhook, `getUpdates` no
funcionara hasta retirarlo; este modulo usa sondeo, sin exponer un servidor a Internet.

Ante una alerta de persona, Tukuru envia al chat autorizado una foto de evidencia si
existe y botones para marcar comportamiento sospechoso confirmado, descartado o no
concluyente. Solo acepta pulsaciones de los IDs autorizados dentro del chat elegido.
La revision queda en `data/alerts/alerts.jsonl` y `data/incidents.sqlite3`; reconocer
o eliminar una alerta del panel no equivale a clasificarla. En `SHADOW_MODE` no se
envian mensajes. Telegram recibe la foto cuando esta opcion se configura; Gemini
sigue siendo independiente.

Los incidentes nuevos guardan caracteristicas numericas versionadas en SQLite. Aun no
se entrena ni se activa un clasificador con las alertas actuales sin etiquetas humanas.
Cuando haya revisiones, `python -m tukuru.ml.dataset --out data/dataset.csv` exporta
solo incidentes con el esquema actual y resultado confirmado/descartado. El CSV queda
local e ignorado por Git. No mezcle cuadros del mismo video entre entrenamiento y prueba.

Antes de exportar/entrenar, audite procedencia y etiquetas con
`python -m tukuru.ml.quality`. El reporte no muestra IDs, rutas ni evidencia y no entrena
ni crea archivos. Para camaras en vivo, asigne en Configuracion un `Grupo de grabacion`
estable a cada grabacion/sesion y repitalo tras reiniciar Tukuru si sigue siendo el mismo
material. Cree otro grupo para una grabacion independiente. Sin este dato, los casos
siguen disponibles para revision, pero no se incluyen en el dataset. En replay el
archivo queda identificado por su hash SHA-256 local, no por la ruta. Si son fragmentos
o versiones re-codificadas de la misma grabacion, pase el mismo `--recording-id` a cada
replay para que el evaluador los mantenga en un solo grupo.

El pipeline agrega muestras locales cuando observa un `alcance` o una `devolucion` que
no llega a nivel de alerta. Se limitan a 20 por camara/hora y una por seguimiento/tipo
en 30 segundos. Aparecen en **Alertas -> Cola agrupada** como muestras que no generaron
alerta; pueden revisarse por lote y no se notifican por Telegram ni se envian a Gemini.
Sus imagenes usan la misma retencion local que la evidencia de alertas. El muestreo no
crea etiquetas: el operador debe confirmar, descartar o dejar no concluyente.

Para preparar un candidato local, instale `pip install -r requirements-ml.txt` y ejecute
`python -m tukuru.ml.train`. El entrenador exige varias sesiones y suficientes revisiones
de ambas clases, conserva las sesiones finales para evaluacion y guarda un JSON en
`data/models_ml/candidato.json`. Si los datos no alcanzan, termina con un mensaje y no
crea modelo. Repetir el mismo video en varios arranques no equivale a reunir sesiones
independientes: revise el origen de los datos antes de entrenar. La comparacion es entre
alertas revisadas, no mide incidentes que las reglas nunca alertaron. El informe compara
la regresion logistica con un Random Forest de referencia; solo exporta el primer modelo.

Para observar un candidato sin cambiar decisiones, defina
`ML_SHADOW_MODEL=data/models_ml/candidato.json` en `.env` y reinicie Tukuru. La
puntuacion aparece en la alerta y se guarda en SQLite; las reglas siguen determinando
niveles y avisos. El archivo es JSON con coeficientes, no un pickle ejecutable. No active
un umbral de ML para ocultar alertas sin una evaluacion independiente.

## GPU NVIDIA

Primero ejecute `nvidia-smi` en el mismo sistema donde corre Tukuru. En WSL debe mostrar el modelo, el driver y la version CUDA admitida. Luego:

```
python scripts/preparar_entorno.py --gpu
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

En Windows use `.venv\Scripts\python` en lugar de `.venv/bin/python`. El ultimo valor debe ser `True`. `DEVICE=auto` elige `cuda:0` si PyTorch ve la GPU y `cpu` en otro caso. Tambien se puede fijar `DEVICE=cpu`, `DEVICE=cuda:0` o `DEVICE=mps`. Si el `.env` local ya tiene `DEVICE=cpu`, cambielo a `auto` manualmente o use `DEVICE=auto python -m tukuru`; el script nunca modifica un `.env` existente.

La RTX 4050 Laptop con driver 591.66 (CUDA maxima 13.1) usa la rueda `cu128` del indice oficial de PyTorch. El driver puede ejecutar una rueda CUDA anterior a su version maxima. La inferencia CUDA usa FP16 (`half=True` en Ultralytics anterior; `quantize=16` en versiones nuevas que deprecan `half`). Si CUDA falla durante la inferencia, Tukuru avisa y reintenta en CPU; ese reinicio del modelo pierde los IDs de seguimiento activos de esa camara.

Para medir rendimiento sin Gemini, ejecute `python scripts/medir_rendimiento.py --device cpu --cameras 1` y repita con `--device cuda:0` y `--cameras 2` o `3`. El modo `trackers` usa una instancia de YOLO/ByteTrack por camara. `--mode predict-lock` y `--mode predict-batch` comparan solo la deteccion: no conservan IDs y no equivalen al pipeline completo. `munecas_visibles` cuenta puntos con confianza >= 0.40; no es una medida de exactitud contra anotaciones humanas.

## Instalacion manual (si el script no sirve)

```
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-test.txt     # o requirements.txt
copy .env.example .env                                           # cp en Linux/macOS
```

Requiere **Python 3.10 o superior**.

## Modelo YOLO

El programa baja `yolov8n-pose.pt` a `models/` la primera vez que arranca (`ensure_model` en `tukuru/vision.py`), si hay acceso a GitHub. Sin red o con proxy: descargarlo a mano de https://github.com/ultralytics/assets/releases y copiarlo a `models/yolov8n-pose.pt`. En el equipo del autor esa descarga se colgo una vez, asi que conviene tenerlo a mano.

## Secretos y archivos locales

- `.env`: **cada persona usa su propia `GEMINI_API_KEY`**. No se sube ni se comparte por chat. `.gitignore` ya lo excluye junto con `data/`, `models/` y `*.pt`.
- `data/` (camaras, alertas, evidencia) se crea al ejecutar; no es compartido.
- `video/` esta versionado e incluye videos y recortes de prueba. Si crece o contiene imagenes que no deban salir del equipo, pasarlo a Git LFS o a `.gitignore`.

## Problemas frecuentes

| Sintoma | Causa y solucion |
|---|---|
| `pip: command not found` / `No module named pip` | Instalar pip: `python -m ensurepip --upgrade`. En Debian/Ubuntu: `sudo apt install python3-pip python3-venv` |
| `No module named numpy` / `cv2` | No se instalo en el Python que se usa. Usar el de `.venv` (`.venv\Scripts\python`) o activar el entorno |
| `No module named tukuru` | Ejecutar desde la raiz del repositorio (`python -m ...` no funciona desde otra carpeta) |
| Falla `venv` | Debian/Ubuntu: `sudo apt install python3-venv` |
| Falta el modelo YOLO | Ver "Modelo YOLO" arriba |
| `Configuracion invalida: ...` al arrancar | Valor fuera de rango en `.env`; el mensaje lista todos los errores juntos. Comparar con `.env.example` |
| Git dice que `README.md` tiene cambios locales distintos | Es una edicion local sin subir. Ver `git diff README.md`; conservarla (`git add`/commit), descartarla (`git checkout -- README.md`) o guardarla (`git stash`) antes de `git pull` |
| `torch` tarda o pesa mucho | Normal (varios GB). Para solo pruebas usar `--solo-pruebas` |
| `torch.version.cuda` es `None` o `torch.cuda.is_available()` es `False` | Se instalo PyTorch CPU o WSL no expone la GPU. Ejecute `nvidia-smi`, luego `python scripts/preparar_entorno.py --gpu` y repita la comprobacion |
