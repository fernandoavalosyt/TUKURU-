"""Juez Gemini Vision: revisa un clip corto (recortes cronologicos) de UNA persona en una
tienda de abarrotes y dictamina si oculto un producto o si es una conducta normal.

Ahorro de tokens: nunca se envia el video completo. Solo se consulta cuando la heuristica
local detecta un posible ocultamiento, con pocos recortes pequenos y un resumen numerico.
Hay cooldown por persona, tope de llamadas por hora y pausa automatica ante limites de cuota.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .config import Settings

logger = logging.getLogger("tukuru.gemini")

SERVER_RETRIES = 3


class _DisabledDuringReview(Exception):
    """Una consulta en cola ya no debe salir hacia Gemini."""

SYSTEM_PROMPT = """Eres un analista de prevencion de perdidas de una tienda de abarrotes. Recibes una secuencia cronologica de recortes de UNA persona tomados por una camara, y un resumen numerico de sensores locales que ya detectaron: la mano llego al estante y despues se quedo junto al cuerpo.
Decide si la persona OCULTO un producto de la tienda (bolsillo, bajo la ropa, bolso o mochila propios) o si la conducta es NORMAL.

Indicios de ocultamiento: toma un producto del estante y desaparece de la mano; la mano entra al bolsillo, a la cintura, bajo la chaqueta o a un bolso con algo; mira alrededor antes o despues; se da vuelta o se agacha para tapar la accion; quita empaque o etiquetas.
Conducta normal: lleva el producto a la vista en la mano; lo pone en una canasta o carrito de la tienda; lo devuelve al estante; solo revisa la etiqueta; saca o guarda su celular, cartera o llaves; se acomoda la ropa o se rasca; es empleado reponiendo mercancia.
Un producto en canasta, carrito o mano visible NO es ocultamiento. Guardar algo propio NO es ocultamiento. Si no se ve que tomo un producto, usa accion 'sin_toma' y state 'normal'.
Los recortes son pequenos y el producto puede ser diminuto. Fijate en las MANOS y en el bolso: si el producto esta en la mano y en un recorte posterior ya no se ve mientras las manos estan sobre el bolso, la ropa o el bolsillo, eso es ocultamiento aunque no se vea el instante exacto de entrar. Responde 'normal' solo si ves claramente que el producto queda a la vista, se devuelve al estante o no hubo toma; si no se ven las manos o el producto en los momentos clave, responde 'indeterminado' y no 'normal'. Usa 'ocultamiento' solo con evidencia clara en los recortes; si las imagenes no bastan, 'indeterminado'. Nunca afirmes que es un robo: describe lo que se ve. El resumen numerico es orientativo: manda lo que ves."""


class Verdict(BaseModel):
    state: Literal["ocultamiento", "normal", "indeterminado"]
    confidence: float = Field(ge=0.0, le=1.0)
    action: Literal["oculta_en_ropa", "oculta_en_bolso", "pone_en_canasta", "lleva_en_mano", "devuelve_al_estante",
                    "sin_toma", "no_visible"]
    product: str = Field(description="Producto que se ve, en pocas palabras; vacio si no se ve.")
    place: Literal["bolsillo", "bajo_la_ropa", "bolso_o_mochila", "ninguno", "no_visible"]
    looked_around: bool
    reasoning: str = Field(description="Maximo 2 frases en espanol, solo lo observable.")


VerdictCallback = Callable[[str, Optional[Dict[str, Any]], Optional[str]], None]


class GeminiJudge:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._client = None
        self._types = None
        self._errors_mod = None
        self.disabled_reason: Optional[str] = None
        self._paused_until = 0.0
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._last_request: Dict[str, float] = {}
        self._call_times: Deque[float] = deque()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._generation = 0
        self.stats = {"calls": 0, "ocultamiento": 0, "normal": 0, "indeterminado": 0,
                      "errors": 0, "tokens": 0, "skipped_budget": 0}
        self.last_error: Optional[str] = None

        if not settings.gemini_enabled:
            self.disabled_reason = "GEMINI_ENABLED=false"
        elif not settings.gemini_api_key:
            self.disabled_reason = "falta GEMINI_API_KEY"
        else:
            try:
                self._create_client()
            except ImportError:
                self.disabled_reason = "paquete google-genai no instalado"
            except Exception as exc:  # noqa: BLE001
                self.disabled_reason = f"no se pudo crear el cliente: {exc.__class__.__name__}"
        if self.disabled_reason:
            logger.warning("Gemini DESACTIVADO (%s): se usara solo la heuristica local.", self.disabled_reason)
        else:
            logger.info("Gemini activo | modelo=%s | maximo %d llamadas/hora", settings.gemini_model,
                        settings.gemini_max_calls_per_hour)

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.disabled_reason is None and time.time() >= self._paused_until

    def _create_client(self) -> None:
        from google import genai
        from google.genai import errors, types

        self._client = genai.Client(api_key=self.s.gemini_api_key,
                                    http_options=types.HttpOptions(timeout=int(self.s.gemini_timeout * 1000)))
        self._types, self._errors_mod = types, errors
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini")

    def set_enabled(self, enabled: bool) -> None:
        """Cambia la admision de nuevas consultas sin reiniciar las camaras."""
        if not enabled:
            with self._lock:
                if self.disabled_reason == "desactivado desde el panel":
                    return
                self.disabled_reason = "desactivado desde el panel"
                self._generation += 1
            logger.info("Gemini desactivado desde el panel")
            return
        if not self.s.gemini_api_key:
            raise ValueError("Falta GEMINI_API_KEY en .env")
        with self._lock:
            if self.disabled_reason is None:
                return
            if self._executor is None:
                try:
                    self._create_client()
                except ImportError as exc:
                    raise ValueError("Falta el paquete google-genai") from exc
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"No se pudo crear el cliente Gemini ({exc.__class__.__name__})") from exc
            self.disabled_reason = None
            self._generation += 1
        logger.info("Gemini activado desde el panel | modelo=%s", self.s.gemini_model)

    def can_review(self, key: str, now: float) -> bool:
        if not self.available:
            return False
        with self._lock:
            if key in self._pending or now - self._last_request.get(key, 0.0) < self.s.gemini_cooldown:
                return False
            while self._call_times and now - self._call_times[0] > 3600:
                self._call_times.popleft()
            return len(self._call_times) < self.s.gemini_max_calls_per_hour

    def review(self, key: str, frames: List[Tuple[float, bytes]], context: Dict[str, Any],
               callback: VerdictCallback) -> bool:
        """Encola la revision (no bloquea). `frames` = [(segundos_atras, jpeg)]. Devuelve False si se rechaza."""
        now = time.time()
        if not frames or self._executor is None or not self.can_review(key, now):
            if self.available and frames:
                self.stats["skipped_budget"] += 1
            return False
        with self._lock:
            if self.disabled_reason is not None or self._executor is None:
                return False
            generation = self._generation
            self._pending.add(key)
            self._last_request[key] = now
            self._call_times.append(now)
        try:
            self._executor.submit(self._run, key, frames, context, callback, generation)
        except RuntimeError:
            with self._lock:
                self._pending.discard(key)
            return False
        return True

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def health(self) -> Dict[str, Any]:
        return {"active": self.disabled_reason is None, "disabled_reason": self.disabled_reason,
                "model": self.s.gemini_model, "paused_seconds": max(0, round(self._paused_until - time.time())),
                "pending": len(self._pending), "last_error": self.last_error, **self.stats}

    # ------------------------------------------------------------------
    def _run(self, key: str, frames: List[Tuple[float, bytes]], context: Dict[str, Any],
             callback: VerdictCallback, generation: int) -> None:
        with self._lock:
            cancelled = generation != self._generation or self.disabled_reason is not None
            if cancelled:
                self._pending.discard(key)
        if cancelled:
            try:
                callback(key, None, "gemini_desactivado")
            except Exception:  # noqa: BLE001
                logger.exception("Callback de Gemini fallo")
            return
        types = self._types
        started = time.time()
        verdict: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        tokens = 0
        try:
            parts: List[Any] = []
            for age, jpeg in frames:
                parts.append(f"Recorte tomado hace {age:.0f} s:")
                parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
            summary = ", ".join(f"{k}={v}" for k, v in context.items() if v is not None)
            parts.append(f"Resumen de sensores locales (orientativo): {summary}. Evalua si hubo ocultamiento.")
            config: Dict[str, Any] = {
                "system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json",
                "response_schema": Verdict, "temperature": 0.0, "max_output_tokens": 2048,
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if self.s.gemini_thinking_level:
                config["thinking_config"] = types.ThinkingConfig(thinking_level=self.s.gemini_thinking_level)
            for attempt in range(SERVER_RETRIES + 1):
                if generation != self._generation or self.disabled_reason is not None:
                    raise _DisabledDuringReview()
                try:
                    response = self._client.models.generate_content(
                        model=self.s.gemini_model, contents=parts, config=types.GenerateContentConfig(**config))
                    break
                except self._errors_mod.ServerError as exc:       # 503/504: servicio sobrecargado, se reintenta
                    if attempt == SERVER_RETRIES:
                        raise
                    logger.info("Gemini %s: %s, reintento %d/%d", key, exc.code, attempt + 1, SERVER_RETRIES)
                    time.sleep(5.0 * (attempt + 1))
            parsed = response.parsed
            verdict = (parsed if isinstance(parsed, Verdict)
                       else Verdict.model_validate_json(response.text or "")).model_dump()
            usage = getattr(response, "usage_metadata", None)
            tokens = int(getattr(usage, "total_token_count", 0) or 0)
        except _DisabledDuringReview:
            error = "gemini_desactivado"
        except self._errors_mod.ClientError as exc:
            error = self._client_error(exc)
        except self._errors_mod.ServerError as exc:
            error = f"server_error_{exc.code}"
        except Exception as exc:  # noqa: BLE001
            error = "timeout" if "Timeout" in exc.__class__.__name__ else f"{exc.__class__.__name__}: {str(exc)[:100]}"

        latency = (time.time() - started) * 1000
        with self._lock:
            self._pending.discard(key)
            stale = generation != self._generation or self.disabled_reason is not None
            if verdict is not None:
                self.stats["calls"] += 1
                self.stats[verdict["state"]] += 1
                self.stats["tokens"] += tokens
            elif error != "gemini_desactivado":
                self.stats["errors"] += 1
                self.last_error = error
        if verdict is not None:
            logger.info("Gemini %s -> %s (conf %.2f, accion=%s, %d tokens, %.0f ms)", key, verdict["state"],
                        verdict["confidence"], verdict["action"], tokens, latency)
        else:
            logger.warning("Gemini %s fallo (%s, %.0f ms); se conserva la heuristica local", key, error, latency)
        try:
            callback(key, None if stale else verdict, "gemini_desactivado" if stale else error)
        except Exception:  # noqa: BLE001
            logger.exception("Callback de Gemini fallo")

    def _client_error(self, exc: Any) -> str:
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", "") or exc)
        if code == 429:
            self._paused_until = time.time() + 60
            logger.warning("Gemini: limite de cuota (429). Pausa de 60 s.")
            return "rate_limited"
        if code in (401, 403) or (code == 400 and "API key" in message):
            self.disabled_reason = f"API key invalida o sin permisos (HTTP {code})"
            logger.error("Gemini: %s. Desactivado hasta reiniciar.", self.disabled_reason)
            return "invalid_api_key"
        if code == 404:
            self.disabled_reason = f"modelo '{self.s.gemini_model}' no disponible (HTTP 404). Revise GEMINI_MODEL"
            logger.error("Gemini: %s", self.disabled_reason)
            return "model_not_found"
        return f"client_error_{code}"
