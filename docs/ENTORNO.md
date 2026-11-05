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
| Pruebas unitarias | `requirements-test.txt` (numpy, opencv-python-headless) | nada: ni red, ni modelo, ni clave |
| Replay local (`python -m tukuru.replay ...`) | `requirements.txt` completo | `models/yolov8n-pose.pt` |
| Replay con Gemini (`--gemini`) | `requirements.txt` | `GEMINI_API_KEY` propia en `.env`; **envia imagenes a la API: pedir autorizacion** |
| Aplicacion (`python -m tukuru`) | `requirements.txt` | modelo + camaras (panel o `.env`) |

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
