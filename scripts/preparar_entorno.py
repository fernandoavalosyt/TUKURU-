"""Prepara el entorno de trabajo: .venv, dependencias, .env y modelo YOLO.

    python scripts/preparar_entorno.py                 # todo (instala torch: varios GB)
    python scripts/preparar_entorno.py --solo-pruebas  # solo numpy + opencv: suficiente para las pruebas

Se puede repetir sin problema. No toca un .env ni un modelo que ya existan.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
MODELO = "yolov8n-pose.pt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(*cmd: object) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo-pruebas", action="store_true", help="instala solo lo necesario para las pruebas")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print(f"Se requiere Python 3.10 o superior (esta es {sys.version.split()[0]}).")
        return 1

    print("1/4 Entorno virtual")
    if not venv_python().exists():
        try:
            run(sys.executable, "-m", "venv", VENV)
        except subprocess.CalledProcessError:
            print("No se pudo crear el entorno. En Debian/Ubuntu: sudo apt install python3-venv python3-pip")
            return 1
    python = venv_python()
    run(python, "-m", "ensurepip", "--upgrade")
    run(python, "-m", "pip", "install", "--upgrade", "pip")

    print("2/4 Dependencias")
    run(python, "-m", "pip", "install", "-r", "requirements-test.txt" if args.solo_pruebas else "requirements.txt")

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
