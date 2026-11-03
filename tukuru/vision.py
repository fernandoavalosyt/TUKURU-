"""Vision local: YOLO-pose (deteccion de personas + tracking con ID + 17 puntos
del cuerpo en un solo modelo). Las munecas y caderas alimentan la heuristica de robo."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger("tukuru.vision")

# Indices COCO de los 17 puntos
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = range(13)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
KP_CONF = 0.40


@dataclass
class Person:
    track_id: int
    bbox: np.ndarray        # x1, y1, x2, y2 (coordenadas del frame reducido)
    conf: float
    kpts: np.ndarray        # (17, 3): x, y, confianza


def ensure_model(models_dir: Path, name: str) -> str:
    """Devuelve la ruta del modelo; si falta, lo descarga (asset oficial de Ultralytics)."""
    target = models_dir / name
    if target.is_file():
        return str(target)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando modelo %s ...", name)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = attempt_download_asset(str(target))
    if downloaded and Path(downloaded).is_file() and Path(downloaded).resolve() != target.resolve():
        shutil.copyfile(downloaded, target)
    if not target.is_file():
        raise RuntimeError(f"No se pudo obtener el modelo '{name}'. Copielo manualmente en {models_dir}.")
    return str(target)


class PoseTracker:
    """YOLO-pose + ByteTrack. Una instancia por camara (el tracker conserva estado)."""

    def __init__(self, model_path: str, device: str, conf: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def infer(self, frame: np.ndarray, imgsz: int) -> List[Person]:
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=self.conf,
                                   imgsz=imgsz, device=self.device, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return []
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()
        return [Person(int(i), boxes[n], float(confs[n]), kpts[n]) for n, i in enumerate(ids)]
