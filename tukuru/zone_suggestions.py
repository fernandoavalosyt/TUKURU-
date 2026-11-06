"""Propuesta de zonas desde una sola captura, solicitada por el operador."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .config import Settings, _validate_zones

DEFAULT_ZONE_MODEL = "gemini-3-flash-preview"


class ZoneProposal(BaseModel):
    estanteria: list[list[list[float]]] = Field(default_factory=list)
    caja: list[list[list[float]]] = Field(default_factory=list)
    salida: list[list[list[float]]] = Field(default_factory=list)
    personal: list[list[list[float]]] = Field(default_factory=list)
    notas: str = Field(default="", max_length=400)


PROMPT = """Analiza una sola imagen fija de una camara de tienda para proponer zonas de vigilancia.
Devuelve coordenadas normalizadas entre 0 y 1, con origen arriba a la izquierda.
estanteria: delimita solamente la cara de los estantes donde una mano podria alcanzar productos;
no incluyas el pasillo, suelo ni el cuerpo de las personas. caja: region visible donde se paga.
salida: suelo o umbral visible por el que salen los pies. personal: zona claramente restringida
a empleados. Una zona no visible o dudosa debe quedar como lista vacia; no inventes puertas,
cajas ni areas de personal. Cada poligono debe tener 3 a 16 puntos. Maximo 8 por tipo.
En notas explica brevemente lo que no se pudo identificar. Estas zonas son una propuesta
que el operador revisara antes de guardar; no afirmes que detectan un robo."""


def encode_frame(frame: np.ndarray) -> bytes:
    """Reduce el cuadro completo para limitar datos y costo de la consulta."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("La camara no entrego una imagen valida")
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        raise ValueError("No se pudo preparar la imagen de la camara")
    return encoded.tobytes()


def validate_proposal(value: ZoneProposal | dict) -> tuple[dict, str]:
    proposal = ZoneProposal.model_validate(value)
    raw = proposal.model_dump(exclude={"notas"})
    zones = _validate_zones(raw) or {}
    for kind, polygons in zones.items():
        for polygon in polygons:
            area = abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                           - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                           for i in range(len(polygon)))) / 2
            if area < 0.001:
                raise ValueError(f"Gemini devolvio una zona {kind} demasiado pequena")
    if not zones:
        raise ValueError("Gemini no identifico zonas visibles; pruebe otro encuadre")
    return zones, proposal.notas


def suggest_zones(jpeg: bytes, settings: Settings, model: str = DEFAULT_ZONE_MODEL) -> tuple[dict, str]:
    if not settings.gemini_api_key:
        raise ValueError("Falta GEMINI_API_KEY en .env")
    if model not in (DEFAULT_ZONE_MODEL, settings.gemini_model):
        raise ValueError("Modelo de zonas no permitido")
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as exc:
        raise ValueError("Falta el paquete google-genai") from exc
    client = genai.Client(api_key=settings.gemini_api_key,
                          http_options=types.HttpOptions(timeout=int(max(60, settings.gemini_timeout) * 1000)))
    try:
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ZoneProposal,
                                              temperature=0, max_output_tokens=4096),
        )
    except errors.APIError as exc:
        code = getattr(exc, "code", None)
        if code in (401, 403):
            detail = "clave sin permisos"
        elif code == 404:
            detail = "modelo no disponible para esta cuenta"
        elif code == 429:
            detail = "cuota agotada; espere o use otro modelo"
        elif code in (500, 502, 503, 504):
            detail = "servicio temporalmente ocupado; intente mas tarde"
        elif code == 400:
            detail = "solicitud o esquema rechazado por el modelo"
        else:
            detail = "error del servicio"
        raise ValueError(f"Gemini {model}: {detail} (HTTP {code})") from None
    except Exception as exc:  # noqa: BLE001 - no incluir URLs/clave del SDK en el error visible
        kind = type(exc).__name__
        if "Timeout" in kind:
            raise ValueError(f"Gemini {model}: tiempo de espera agotado") from None
        if "Connect" in kind or "Network" in kind:
            raise ValueError(f"Gemini {model}: no se pudo conectar al servicio") from None
        raise ValueError(f"Gemini {model}: fallo local al solicitar analisis ({kind})") from None
    result = response.parsed if response.parsed is not None else ZoneProposal.model_validate_json(response.text or "")
    return validate_proposal(result)
