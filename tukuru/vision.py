"""Vision local: YOLO-pose (deteccion de personas + tracking con ID + 17 puntos
del cuerpo en un solo modelo). Las munecas y caderas alimentan la heuristica de robo."""

from __future__ import annotations

import logging
import gc
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


def resolve_device(requested: str, cuda_available: bool, mps_available: bool) -> str:
    """Elige un dispositivo sin importar torch al leer la configuracion."""
    if requested == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if requested == "cuda:0" and not cuda_available:
        return "cpu"
    if requested == "mps" and not mps_available:
        return "cpu"
    return requested


def fp16_arguments() -> dict:
    """Ultralytics nuevo usa quantize=16; versiones anteriores usan half=True."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError:
        return {"half": True}
    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


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

    def __init__(self, model_path: str, device: str, conf: float, imgsz: int = 640) -> None:
        import torch
        from ultralytics import YOLO

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.device = resolve_device(device, torch.cuda.is_available(), torch.backends.mps.is_available())
        if device in ("cuda:0", "mps") and self.device == "cpu":
            logger.warning("Dispositivo %s no disponible; se usara CPU", device)
        self.conf = conf
        self.imgsz = imgsz
        self.half = self.device.startswith("cuda")
        self._precision = fp16_arguments() if self.half else {}
        try:
            self._warmup()
        except Exception as exc:
            if not self.half:
                raise
            self._fallback_cpu(exc)
        logger.info("YOLO-pose en %s (%s)", self.device, "FP16" if self.half else "FP32")

    def _warmup(self) -> None:
        # predict inicia los kernels sin crear un track ficticio en ByteTrack.
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False, **self._precision)

    def _fallback_cpu(self, exc: Exception) -> None:
        import torch
        from ultralytics import YOLO

        logger.warning("CUDA fallo (%s); reiniciando YOLO-pose en CPU", exc)
        self.model = YOLO(self.model_path)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("No se pudo liberar la cache de CUDA", exc_info=True)
        self.device, self.half, self._precision = "cpu", False, {}
        self._warmup()

    def infer(self, frame: np.ndarray, imgsz: int) -> List[Person]:
        try:
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=self.conf,
                                       imgsz=imgsz, device=self.device, verbose=False, **self._precision)
        except Exception as exc:
            if not self.half:
                raise
            self._fallback_cpu(exc)
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=self.conf,
                                       imgsz=imgsz, device=self.device, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return []
        boxes = result.boxes.cpu().numpy()
        kpts = result.keypoints.cpu().numpy().data
        return [Person(int(i), boxes.xyxy[n], float(boxes.conf[n]), kpts[n])
                for n, i in enumerate(boxes.id.astype(int))]
