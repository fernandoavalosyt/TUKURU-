import json
import tempfile
import unittest
from pathlib import Path

from tukuru.config import CameraStore, validate_camera
from tukuru.credentials import CredentialError, KeyringCredentials, open_credentials


class FakeKeyring:
    """Llavero en memoria con la misma interfaz que el paquete keyring."""

    def __init__(self, broken=False):
        self.items, self.broken = {}, broken

    def get_password(self, service, user):
        if self.broken:
            raise RuntimeError("sin backend")
        return self.items.get((service, user))

    def set_password(self, service, user, password):
        if self.broken:
            raise RuntimeError("sin backend")
        self.items[(service, user)] = password

    def delete_password(self, service, user):
        self.items.pop((service, user), None)


def cam(**extra):
    data = {"host": "10.0.0.5", "user": "admin", "password": "secreto", "id": "CAM-01"}
    data.update(extra)
    return validate_camera(data)


class CredentialStorage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "cameras.json"
        self.kr = FakeKeyring()
        self.creds = KeyringCredentials(self.kr)

    def tearDown(self):
        self._tmp.cleanup()

    def raw(self):
        return self.path.read_text(encoding="utf-8")

    def test_password_goes_to_keyring_not_to_file(self):
        store = CameraStore(self.path, 3, self.creds)
        store.upsert(cam())
        self.assertNotIn("secreto", self.raw())
        self.assertEqual(self.kr.items[("tukuru", "camara:CAM-01")], "secreto")
        self.assertEqual(store.where_password("CAM-01"), "keyring")

    def test_reload_recovers_password_from_keyring(self):
        CameraStore(self.path, 3, self.creds).upsert(cam())
        again = CameraStore(self.path, 3, self.creds)
        self.assertEqual(again.get("CAM-01").password, "secreto")
        self.assertIn("admin:secreto@", again.get("CAM-01").source())

    def test_plaintext_file_is_migrated(self):
        self.path.write_text(json.dumps({"cameras": [cam().to_json()]}), encoding="utf-8")
        self.assertIn("secreto", self.raw())
        store = CameraStore(self.path, 3, self.creds)
        self.assertNotIn("secreto", self.raw())
        self.assertEqual(store.get("CAM-01").password, "secreto")

    def test_without_keyring_password_stays_in_file(self):
        store = CameraStore(self.path, 3, None)
        store.upsert(cam())
        self.assertIn("secreto", self.raw())
        self.assertEqual(store.where_password("CAM-01"), "file")
        self.assertEqual(store.mode, "file")

    def test_keyring_failure_keeps_password_in_file(self):
        store = CameraStore(self.path, 3, KeyringCredentials(FakeKeyring(broken=True)))
        store.upsert(cam())                                  # no debe perder la contrasena
        self.assertIn("secreto", self.raw())
        self.assertEqual(store.where_password("CAM-01"), "file")

    def test_remove_deletes_secret(self):
        store = CameraStore(self.path, 3, self.creds)
        store.upsert(cam())
        self.assertTrue(store.remove("CAM-01"))
        self.assertEqual(self.kr.items, {})

    def test_clearing_user_deletes_secret(self):
        store = CameraStore(self.path, 3, self.creds)
        store.upsert(cam())
        store.upsert(validate_camera({"host": "10.0.0.5", "id": "CAM-01"}))
        self.assertEqual(self.kr.items, {})

    def test_mode_selection(self):
        self.assertIsNone(open_credentials("file"))
        self.assertIsNotNone(open_credentials("auto", backend=self.kr))
        with self.assertRaises(CredentialError):
            open_credentials("keyring", backend=FakeKeyring(broken=True))
        self.assertIsNone(open_credentials("auto", backend=FakeKeyring(broken=True)))


class CameraForm(unittest.TestCase):
    def test_url_overrides_form_protocol(self):
        # El formulario siempre manda un protocolo; la URL pegada debe mandar sobre el.
        c = validate_camera({"protocol": "rtsp", "url": "http://u:p@10.0.0.9:8080/video", "brand": "generico"})
        self.assertEqual((c.protocol, c.host, c.port, c.user, c.password, c.path),
                         ("http", "10.0.0.9", 8080, "u", "p", "/video"))

    def test_blank_password_keeps_existing_for_same_user(self):
        c = validate_camera({"host": "10.0.0.5", "user": "admin"}, existing_password="secreto")
        self.assertEqual(c.password, "secreto")


if __name__ == "__main__":
    unittest.main()
