"""Vector fijo e inferencia local de un clasificador logistico en modo sombra."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .features import FEATURE_VERSION

FEATURE_NAMES = (
    "score", "seg_visible", "alcances_2min", "ocultamientos_activos",
    "seg_desde_ultimo_alcance", "falta_ultimo_alcance", "giros_cabeza",
    "velocidad_cuerpo", "conf_muneca_min", "conf_muneca_max",
    "paso_por_caja", "salio", "lugar_bolsillo", "lugar_pecho_o_abdomen",
    "lugar_mano_oculta",
)


def feature_vector(features: dict) -> np.ndarray:
    if features.get("version") != FEATURE_VERSION:
        raise ValueError("version de caracteristicas incompatible")
    since = features.get("seg_desde_ultimo_alcance")
    place = features.get("lugar_ocultamiento")
    values = [features.get("score", 0), features.get("seg_visible", 0),
              features.get("alcances_2min", 0), features.get("ocultamientos_activos", 0),
              since or 0, since is None, features.get("giros_cabeza", 0),
              features.get("velocidad_cuerpo", 0), features.get("conf_muneca_min", 0),
              features.get("conf_muneca_max", 0), bool(features.get("paso_por_caja")),
              bool(features.get("salio")), place == "bolsillo",
              place == "pecho_o_abdomen", place == "mano_oculta"]
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("caracteristicas no finitas")
    return vector


class ShadowModel:
    def __init__(self, artifact: dict) -> None:
        if (artifact.get("feature_version") != FEATURE_VERSION
                or tuple(artifact.get("feature_names", ())) != FEATURE_NAMES
                or artifact.get("algorithm") != "logistic_regression"):
            raise ValueError("modelo incompatible con el extractor actual")
        self.version = str(artifact["version"])
        self.mean = np.asarray(artifact["mean"], dtype=np.float64)
        self.scale = np.asarray(artifact["scale"], dtype=np.float64)
        self.coef = np.asarray(artifact["coef"], dtype=np.float64)
        self.intercept = float(artifact["intercept"])
        if (any(len(v) != len(FEATURE_NAMES) for v in (self.mean, self.scale, self.coef))
                or np.any(self.scale <= 0)
                or not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.coef))
                or not math.isfinite(self.intercept)):
            raise ValueError("parametros del modelo invalidos")

    @classmethod
    def load(cls, path: Path) -> "ShadowModel":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def score(self, features: dict) -> float:
        vector = (feature_vector(features) - self.mean) / self.scale
        logit = float(vector @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
