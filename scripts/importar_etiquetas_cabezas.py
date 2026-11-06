"""Convierte las anotaciones locales del navegador al formato YOLO de una sesion."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "head_dataset"
SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa cajas de cabezas anotadas localmente a etiquetas YOLO")
    parser.add_argument("--session", required=True, help="ID de la sesion muestreada")
    parser.add_argument("--annotations", type=Path, required=True, help="JSON descargado desde etiquetar_cabezas.html")
    args = parser.parse_args()
    if not SESSION_RE.fullmatch(args.session):
        parser.error("ID de sesion invalido")
    session_dir = DATA_ROOT / args.session
    images_dir, labels_dir = session_dir / "images", session_dir / "labels"
    manifest_path = session_dir / "manifest.json"
    if not images_dir.is_dir():
        parser.error("no existe la sesion muestreada; ejecuta primero muestrear_cabezas.py")
    try:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"no se pudo leer el manifiesto o las etiquetas ({type(exc).__name__})")
    if annotations.get("session_id") != args.session:
        parser.error("el ID de sesion del JSON no coincide con --session")

    if manifest is None:
        # Una camara puede perder senal despues de guardar cuadros y antes de cerrar
        # el manifiesto. Recupera nombres desde la carpeta sin inventar metadatos.
        known = {path.name: {} for path in images_dir.iterdir()
                 if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")}
        recovered = True
    else:
        known = {row["filename"]: row for row in manifest.get("frames", [])}
        recovered = False
    rows = annotations.get("images")
    if not isinstance(rows, list) or not rows:
        parser.error("el JSON no contiene cuadros revisados")
    seen, positives, negatives, head_count = set(), 0, 0, 0
    labels_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row.get("filename")
        if not isinstance(name, str) or Path(name).name != name or name not in known:
            parser.error("el JSON contiene un nombre de imagen que no pertenece a esta sesion")
        if name in seen:
            parser.error("el JSON contiene una imagen repetida")
        seen.add(name)
        image_path = images_dir / name
        if not image_path.is_file():
            parser.error(f"falta una imagen muestreada: {name}")
        width, height = row.get("width"), row.get("height")
        if (not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0):
            parser.error(f"resolucion invalida en {name}")
        expected_width, expected_height = known[name].get("width"), known[name].get("height")
        if ((expected_width is not None and width != expected_width) or
                (expected_height is not None and height != expected_height)):
            parser.error(f"la resolucion de {name} no coincide con el manifiesto")
        boxes = row.get("boxes")
        if not isinstance(boxes, list):
            parser.error(f"cajas invalidas en {name}")
        lines = []
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                parser.error(f"caja invalida en {name}")
            try:
                x1, y1, x2, y2 = map(float, box)
            except (TypeError, ValueError):
                parser.error(f"coordenadas invalidas en {name}")
            if (not all(map(math.isfinite, (x1, y1, x2, y2))) or
                    not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)):
                parser.error(f"caja fuera del cuadro en {name}")
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            box_width, box_height = x2 - x1, y2 - y1
            lines.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        (labels_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                              encoding="utf-8")
        if boxes:
            positives += 1
            head_count += len(boxes)
        else:
            negatives += 1

    destination = session_dir / "annotations.json"
    if args.annotations.resolve() != destination.resolve():
        shutil.copyfile(args.annotations, destination)
    print(f"Etiquetas YOLO guardadas para {len(seen)} cuadros: {positives} positivos, {negatives} sin cabeza.")
    print(f"Cajas de cabeza anotadas: {head_count}")
    if recovered:
        print("Aviso: faltaba manifest.json; se verificaron las imagenes por nombre y existencia.")
    print("No combines cuadros de esta misma sesion entre entrenamiento y validacion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
