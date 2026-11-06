"""Prepara el entorno de trabajo: .venv, dependencias, .env y modelo YOLO.

    python scripts/preparar_entorno.py                 # todo (instala torch: varios GB)
    python scripts/preparar_entorno.py --solo-pruebas  # solo numpy + opencv: suficiente para las pruebas
    python scripts/preparar_entorno.py --gpu            # instala PyTorch CUDA segun el driver NVIDIA

Se puede repetir sin problema. No toca un .env ni un modelo que ya existan.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlretrieve
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
MODELO = "yolov8n-pose.pt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(*cmd: object) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def cuda_index_for(version: str) -> str:
    """El driver reporta la mayor version CUDA que acepta, no la instalada."""
    match = re.fullmatch(r"(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"Version CUDA invalida: {version}")
    supported = tuple(map(int, match.groups()))
    for minimum, wheel in (((13, 2), "cu132"), ((12, 8), "cu128"), ((12, 6), "cu126")):
        if supported >= minimum:
            return f"https://download.pytorch.org/whl/{wheel}"
    raise ValueError(f"El driver solo admite CUDA {version}; se requiere 12.6 o superior")


def gpu_index() -> str:
    try:
        report = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"No se pudo consultar la GPU con nvidia-smi: {exc}") from exc
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", report)
    if match is None:
        raise RuntimeError("nvidia-smi no reporto la version CUDA del driver")
    return cuda_index_for(match.group(1))


def installed_cuda(python: Path) -> str:
    try:
        return subprocess.check_output([str(python), "-c", "import torch; print(torch.version.cuda or '')"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--solo-pruebas", action="store_true", help="instala solo lo necesario para las pruebas")
    mode.add_argument("--gpu", action="store_true", help="instala PyTorch con CUDA segun el driver NVIDIA")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print(f"Se requiere Python 3.10 o superior (esta es {sys.version.split()[0]}).")
        return 1

    print("1/4 Entorno virtual")
    if not venv_python().exists():
        try:
            run(sys.executable, "-m", "venv", VENV)
        except subprocess.CalledProcessError:
            print("  venv sin ensurepip; creando entorno sin pip")
            run(sys.executable, "-m", "venv", "--without-pip", VENV)
    python = venv_python()
    if subprocess.call([str(python), "-m", "pip", "--version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        try:
            run(python, "-m", "ensurepip", "--upgrade")
        except subprocess.CalledProcessError:
            print("  ensurepip no esta disponible; instalando pip desde bootstrap.pypa.io")
            with tempfile.TemporaryDirectory() as temp:
                instalador = Path(temp) / "get-pip.py"
                urlretrieve("https://bootstrap.pypa.io/get-pip.py", instalador)
                run(python, instalador)
    run(python, "-m", "pip", "install", "--upgrade", "pip")

    print("2/4 Dependencias")
    if args.gpu:
        index = gpu_index()
        expected_cuda = index.rsplit("cu", 1)[-1]
        expected_cuda = f"{expected_cuda[:-1]}.{expected_cuda[-1]}"
        if installed_cuda(python) != expected_cuda:
            run(python, "-m", "pip", "install", "--force-reinstall", "torch", "torchvision", "--index-url", index)
        else:
            print(f"  PyTorch CUDA {expected_cuda} ya instalado")
    elif not args.solo_pruebas:
        run(python, "-m", "pip", "install", "torch", "torchvision", "--index-url",
            "https://download.pytorch.org/whl/cpu")
    run(python, "-m", "pip", "install", "-r", "requirements-test.txt" if args.solo_pruebas else "requirements.txt")
    if args.gpu:
        available = subprocess.check_output([str(python), "-c", "import torch; print(torch.cuda.is_available())"],
                                            text=True).strip()
        if available != "True":
            raise RuntimeError("PyTorch CUDA se instalo, pero torch.cuda.is_available() es False")

    print("3/4 Configuracion (.env)")
    env, ejemplo = ROOT / ".env", ROOT / ".env.example"
    if env.exists():
        print("  .env ya existe: no se toca")
    else:
        shutil.copyfile(ejemplo, env)
        print("  .env creado desde .env.example: agregue su GEMINI_API_KEY (cada persona usa la suya)")

    print("4/4 Modelo YOLO")
    if args.solo_pruebas:
        print("  omitido (--solo-pruebas)")
    elif (ROOT / "models" / MODELO).is_file():
        print(f"  models/{MODELO} ya existe")
    else:
        try:
            run(python, "-c", "from pathlib import Path; from tukuru.vision import ensure_model; "
                f"print(ensure_model(Path('models'), '{MODELO}'))")
        except subprocess.CalledProcessError:
            print(f"  No se pudo descargar. Bajelo a mano de https://github.com/ultralytics/assets/releases "
                  f"y copielo en models/{MODELO}")

    print("\nListo. Compruebe con:")
    print(f"  {python} -m unittest discover -s tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
