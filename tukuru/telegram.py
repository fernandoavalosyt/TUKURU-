"""Avisos y revision humana por Telegram, opcionales y sin bloquear las camaras."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("tukuru.telegram")
LABELS = {"c": "confirmado", "d": "descartado", "n": "no_concluyente"}


def parse_review(update: dict, chat_id: str, allowed_users: tuple[int, ...]) -> tuple[str, str, int] | None:
    """Valida origen y datos antes de aceptar una pulsacion."""
    query = update.get("callback_query") or {}
    if str((query.get("message") or {}).get("chat", {}).get("id")) != chat_id:
        return None
    reviewer = (query.get("from") or {}).get("id")
    if reviewer not in allowed_users:
        return None
    parts = str(query.get("data") or "").split(":")
    if len(parts) != 3 or parts[0] != "review" or parts[2] not in LABELS:
        return None
    alert_id = parts[1]
    if len(alert_id) != 8 or any(ch not in "0123456789abcdef" for ch in alert_id):
        return None
    return alert_id, LABELS[parts[2]], reviewer


class TelegramBridge:
    def __init__(self, token: str, chat_id: str, allowed_users: tuple[int, ...],
                 review: Callable[[str, str, int], bool], evidence_dir: Path) -> None:
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        self._url = f"https://api.telegram.org/bot{token}/"
        self.chat_id = chat_id
        self.allowed_users = allowed_users
        self.review = review
        self.evidence_dir = evidence_dir
        self._pending: queue.Queue[dict] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._offset: int | None = None

    def start(self) -> None:
        for target, name in ((self._send_loop, "telegram-send"), (self._poll_loop, "telegram-review")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)

    def notify(self, alert: dict) -> None:
        try:
            self._pending.put_nowait(alert)
        except queue.Full:
            logger.warning("Cola de Telegram llena; alerta %s permanece en el registro local", alert["id"])

    def _call(self, method: str, data: dict, *, files: dict | None = None, timeout: int = 10) -> dict:
        import requests

        # No registrar excepciones completas: requests incluiria el token en la URL.
        response = requests.post(self._url + method, data=data, files=files, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError("Telegram rechazo la solicitud")
        return payload

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._send(alert)
            except Exception as exc:  # noqa: BLE001 - nunca detener el pipeline por la red
                logger.warning("No se pudo enviar alerta %s a Telegram (%s)", alert["id"], type(exc).__name__)
            finally:
                self._pending.task_done()

    def _send(self, alert: dict) -> None:
        import json
        keyboard = {"inline_keyboard": [
            [{"text": "Sospechoso confirmado", "callback_data": f"review:{alert['id']}:c"}],
            [{"text": "Descartado", "callback_data": f"review:{alert['id']}:d"}],
            [{"text": "No concluyente", "callback_data": f"review:{alert['id']}:n"}],
        ]}
        caption = ("Alerta para revision humana; no confirma un robo.\n"
                   f"Tukuru | {alert['camera_name']} | persona {alert['track_id']}\n"
                   f"Nivel: {alert['level']}\n{alert['reason']}")[:1024]

        clip_frames: list = alert.get("clip_frames") or []
        if len(clip_frames) >= 2:
            self._send_clip_album(clip_frames, caption, keyboard)
            return

        # Fallback: foto unica o solo texto
        reply_markup = json.dumps(keyboard)
        full_caption = (caption + "\nClasifique el comportamiento observado.")[:1024]
        photo_name = alert.get("snapshot")
        photo = self.evidence_dir / photo_name if photo_name else None
        if photo and photo.is_file() and photo.resolve().parent == self.evidence_dir.resolve():
            with photo.open("rb") as image:
                self._call("sendPhoto", {"chat_id": self.chat_id, "caption": full_caption,
                                         "reply_markup": reply_markup}, files={"photo": image})
        else:
            self._call("sendMessage", {"chat_id": self.chat_id, "text": full_caption,
                                       "reply_markup": reply_markup})

    def _send_clip_album(self, frames: list, caption: str, keyboard: dict) -> None:
        """Album de hasta 10 recortes del clip + mensaje separado con botones de revision."""
        import json
        media = []
        files = {}
        for i, jpeg in enumerate(frames[:10]):
            key = f"f{i}"
            files[key] = (f"{key}.jpg", jpeg, "image/jpeg")
            item: dict = {"type": "photo", "media": f"attach://{key}"}
            if i == 0:
                item["caption"] = caption
            media.append(item)
        self._call("sendMediaGroup", {"chat_id": self.chat_id, "media": json.dumps(media)}, files=files)
        self._call("sendMessage", {"chat_id": self.chat_id,
                                   "text": "Clasifique el comportamiento observado.",
                                   "reply_markup": json.dumps(keyboard)})

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data: dict[str, Any] = {"timeout": 20, "allowed_updates": '["callback_query"]'}
                if self._offset is not None:
                    data["offset"] = self._offset
                updates = self._call("getUpdates", data, timeout=25).get("result", [])
                for update in updates:
                    self._handle(update)
                    self._offset = update["update_id"] + 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudieron leer respuestas de Telegram (%s)", type(exc).__name__)
                self._stop.wait(5)

    def _handle(self, update: dict) -> None:
        query = update.get("callback_query") or {}
        query_id = query.get("id")
        parsed = parse_review(update, self.chat_id, self.allowed_users)
        message = "No autorizado o respuesta invalida"
        if parsed:
            alert_id, label, reviewer = parsed
            message = "Revision guardada" if self.review(alert_id, label, reviewer) else "Alerta no disponible"
        if query_id:
            try:
                self._call("answerCallbackQuery", {"callback_query_id": query_id, "text": message})
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudo confirmar pulsacion de Telegram (%s)", type(exc).__name__)
