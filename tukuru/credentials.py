"""Contrasenas de camaras fuera de data/cameras.json.

Si el sistema operativo tiene llavero (Windows: Administrador de credenciales, macOS: Keychain,
Linux: Secret Service) la contrasena se guarda ahi, cifrada y ligada al usuario del sistema.
Sin llavero (servidor sin escritorio, contenedor) se queda en cameras.json con permisos 600.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("tukuru.credentials")

SERVICE = "tukuru"
MODES = ("auto", "keyring", "file")


class CredentialError(RuntimeError):
    pass


class KeyringCredentials:
    """Una contrasena por camara, bajo el servicio 'tukuru' y el usuario 'camara:<id>'."""

    name = "keyring"

    def __init__(self, backend: Any) -> None:
        self._kr = backend

    @staticmethod
    def _user(cam_id: str) -> str:
        return f"camara:{cam_id}"

    def get(self, cam_id: str) -> Optional[str]:
        try:
            return self._kr.get_password(SERVICE, self._user(cam_id))
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del backend es "no disponible"
            raise CredentialError(f"no se pudo leer la contrasena de {cam_id}: {exc.__class__.__name__}") from exc

    def set(self, cam_id: str, password: str) -> None:
        try:
            self._kr.set_password(SERVICE, self._user(cam_id), password)
        except Exception as exc:  # noqa: BLE001
            raise CredentialError(f"no se pudo guardar la contrasena de {cam_id}: {exc.__class__.__name__}") from exc

    def delete(self, cam_id: str) -> None:
        try:
            self._kr.delete_password(SERVICE, self._user(cam_id))
        except Exception as exc:  # noqa: BLE001 - borrar lo que no existe no es un error
            if exc.__class__.__name__ != "PasswordDeleteError":
                logger.warning("No se pudo borrar la contrasena de %s del llavero: %s", cam_id, exc.__class__.__name__)


def open_credentials(mode: str = "auto", backend: Any = None) -> Optional[KeyringCredentials]:
    """Devuelve el llavero o None si las contrasenas deben ir en el archivo.

    auto: llavero si funciona, si no archivo.  keyring: exige llavero (error si no hay).  file: nunca llavero."""
    if mode == "file":
        return None
    try:
        kr = backend
        if kr is None:
            import keyring  # dependencia opcional
            kr = keyring
            if kr.get_keyring().__class__.__module__.startswith("keyring.backends.fail"):
                raise CredentialError("el sistema no tiene llavero disponible")
        kr.get_password(SERVICE, "__sonda__")     # falla ya aqui si el backend no sirve
        return KeyringCredentials(kr)
    except ImportError:
        reason = "falta el paquete 'keyring' (pip install keyring)"
    except Exception as exc:  # noqa: BLE001
        reason = str(exc) if isinstance(exc, CredentialError) else f"llavero no utilizable ({exc.__class__.__name__})"
    if mode == "keyring":
        raise CredentialError(f"CREDENTIALS_STORE=keyring pero {reason}")
    logger.warning("Contrasenas de camaras en data/cameras.json (permisos 600): %s", reason)
    return None
