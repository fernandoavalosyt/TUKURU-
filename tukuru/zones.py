"""Zonas de la tienda por camara: poligonos normalizados (0-1) por tipo.

    estanteria  cara del estante donde el cliente alcanza productos
    caja        zona de pago (se mide con los pies de la persona)
    salida      puerta o linea de salida (pies)
    personal    area de empleados: quien este ahi se excluye del analisis
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

ZONE_KINDS = ("estanteria", "caja", "salida", "personal")
Polygon = List[List[float]]


class ZoneSet:
    """Poligonos de una camara ya escalados al tamano del frame que se analiza."""

    def __init__(self, zones: Optional[Dict[str, List[Polygon]]], width: int, height: int) -> None:
        self._polys: Dict[str, List[np.ndarray]] = {
            kind: [np.array([[x * width, y * height] for x, y in poly], np.float32) for poly in polys]
            for kind, polys in (zones or {}).items() if polys
        }

    def has(self, kind: str) -> bool:
        return kind in self._polys

    def contains(self, kind: str, point) -> bool:
        x, y = float(point[0]), float(point[1])
        return any(cv2.pointPolygonTest(poly, (x, y), False) >= 0 for poly in self._polys.get(kind, ()))

    def polygons(self, kind: str) -> List[np.ndarray]:
        return [poly.astype(np.int32) for poly in self._polys.get(kind, ())]
