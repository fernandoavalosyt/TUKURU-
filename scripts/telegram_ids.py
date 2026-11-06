"""Muestra IDs de chat y usuario del bot, sin imprimir token ni mensajes.

Detenga Tukuru, envie /start al bot y ejecute: python scripts/telegram_ids.py
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Falta TELEGRAM_BOT_TOKEN en .env")
        return 1
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/getUpdates",
                                 data={"timeout": 0, "allowed_updates": '["message"]'}, timeout=10)
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            raise RuntimeError
    except Exception:  # noqa: BLE001 - nunca imprimir la URL que contiene el token
        print("No se pudieron consultar IDs. Detenga Tukuru y revise si el bot tiene un webhook activo.")
        return 1
    pairs = {(str(item["message"]["chat"]["id"]), item["message"]["from"]["id"])
             for item in result.get("result", []) if "message" in item and "from" in item["message"]}
    for chat, user in sorted(pairs):
        print(f"TELEGRAM_CHAT_ID={chat} TELEGRAM_ALLOWED_USER_IDS={user}")
    if not pairs:
        print("Sin mensajes. Envie /start al bot y repita este comando.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
