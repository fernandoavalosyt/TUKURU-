"""Entrena un candidato local YOLO de cabeza usando cuadros ya etiquetados."""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "head_dataset"
SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def make_split(rows: list[tuple[Path, Path, bool]], fraction: float, seed: int):
    rng = random.Random(seed)
    groups = {True: [], False: []}
    for row in rows:
        groups[row[2]].append(row)
    train, val = [], []
    for group in groups.values():
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * fraction))) if len(group) >= 2 else 0
        val.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def label_box_count(path: Path) -> int:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for values in lines:
        if len(values) != 5 or values[0] != "0":
            raise ValueError(f"Etiqueta YOLO invalida en {path.name}")
        try:
            numbers = [float(value) for value in values[1:]]
        except ValueError:
            raise ValueError(f"Coordenadas invalidas en {path.name}") from None
        if not all(0 <= number <= 1 for number in numbers) or numbers[2] <= 0 or numbers[3] <= 0:
            raise ValueError(f"Caja fuera de rango en {path.name}")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena localmente un detector experimental de cabezas cenitales")
    parser.add_argument("--session", required=True, help="Sesion etiquetada dentro de data/head_dataset")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8n-pose.pt",
                        help="Pesos locales de inicio (default: models/yolov8n-pose.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not SESSION_RE.fullmatch(args.session):
        parser.error("ID de sesion invalido")
    if not args.weights.is_file():
        parser.error("no existe el modelo base local indicado")
    if not 1 <= args.epochs <= 300 or not 320 <= args.imgsz <= 1280 or not 1 <= args.batch <= 32:
        parser.error("epochs, imgsz o batch fuera de rango")
    if not 0.1 <= args.val_fraction <= 0.4:
        parser.error("val-fraction debe estar entre 0.1 y 0.4")

    session_dir = DATA_ROOT / args.session
    images_dir, labels_dir = session_dir / "images", session_dir / "labels"
    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = next((images_dir / f"{label_path.stem}{ext}"
                           for ext in (".jpg", ".jpeg", ".png")
                           if (images_dir / f"{label_path.stem}{ext}").is_file()), None)
        if image_path is None:
            parser.error(f"falta imagen para la etiqueta {label_path.name}")
        try:
            boxes = label_box_count(label_path)
        except ValueError as exc:
            parser.error(str(exc))
        rows.append((image_path, label_path, boxes > 0))
    if len(rows) < 10 or not any(row[2] for row in rows) or all(row[2] for row in rows):
        parser.error("se requieren al menos 10 cuadros etiquetados, positivos y negativos")

    train, val = make_split(rows, args.val_fraction, args.seed)
    if not val or not any(row[2] for row in val) or all(row[2] for row in val):
        parser.error("la particion necesita positivos y negativos en validacion")

    run_root = DATA_ROOT / "training" / f"head_{args.session}_{datetime.now():%Y%m%d_%H%M%S}"
    dataset_root = run_root / "dataset"
    for split_name, split_rows in (("train", train), ("val", val)):
        target_images, target_labels = dataset_root / "images" / split_name, dataset_root / "labels" / split_name
        target_images.mkdir(parents=True, exist_ok=False)
        target_labels.mkdir(parents=True, exist_ok=False)
        for image_path, label_path, _ in split_rows:
            shutil.copy2(image_path, target_images / image_path.name)
            shutil.copy2(label_path, target_labels / label_path.name)

    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset_root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: cabeza\n",
        encoding="utf-8")
    metadata = {"session_id": args.session, "source_weights": str(args.weights.resolve()),
                "train_images": len(train), "validation_images": len(val),
                "train_boxes": sum(label_box_count(row[1]) for row in train),
                "validation_boxes": sum(label_box_count(row[1]) for row in val),
                "validation_warning": "split aleatorio de una sola grabacion; no es evaluacion independiente",
                "seed": args.seed, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch}
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    from ultralytics import YOLO
    import ultralytics

    model_yaml = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "v8" / "yolov8.yaml"
    if not model_yaml.is_file():
        parser.error("no se encontro la arquitectura YOLOv8 local en Ultralytics")
    model = YOLO(str(model_yaml), task="detect")
    model.load(str(args.weights))
    print("Pesos locales de pose transferidos a la arquitectura YOLO de deteccion.")
    print(f"Datos: {len(train)} entrenamiento, {len(val)} validacion; sesion unica {args.session}.")
    print("Advertencia: la validacion es aleatoria dentro del mismo video y puede ser optimista.")
    result = model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                         device="cpu", workers=0, seed=args.seed, patience=8, plots=False,
                         project=str(run_root), name="run", exist_ok=True, verbose=False)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"Candidato guardado: {best}")
    metrics = getattr(result, "results_dict", {})
    for name in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if name in metrics:
            print(f"{name}: {float(metrics[name]):.4f}")
    print("El candidato queda local y no se activa en Tukuru.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
