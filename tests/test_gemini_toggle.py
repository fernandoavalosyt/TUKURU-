"""Control de Gemini sin red ni envio de imagenes."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tukuru.app import Tukuru, saved_gemini_enabled
from tukuru.config import Settings
from tukuru.gemini import GeminiJudge, Verdict
from tukuru.web import create_web


class GeminiToggleTests(unittest.TestCase):
    def test_saved_preference_overrides_startup_setting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gemini_state.json"
            self.assertFalse(saved_gemini_enabled(path, False))
            path.write_text('{"enabled": true}', encoding="utf-8")
            self.assertTrue(saved_gemini_enabled(path, False))
            path.write_text('{"enabled": false}', encoding="utf-8")
            self.assertFalse(saved_gemini_enabled(path, True))
            path.write_text('{"enabled": "si"}', encoding="utf-8")
            self.assertTrue(saved_gemini_enabled(path, True))

    def test_panel_changes_state_and_persists_only_boolean(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), gemini_enabled=False, gemini_api_key="x" * 24)
            judge = GeminiJudge(settings)
            core = Tukuru.__new__(Tukuru)
            core.s = settings
            core.judge = judge
            core._gemini_state_path = Path(temp) / "gemini_state.json"
            client = create_web(core).test_client()
            headers = {"X-Requested-With": "tukuru"}
            self.assertEqual(client.post("/api/gemini", json={"enabled": "true"}, headers=headers).status_code, 400)

            def fake_client(instance):
                instance._executor = Mock()

            with patch.object(GeminiJudge, "_create_client", fake_client):
                response = client.post("/api/gemini", json={"enabled": True}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["gemini"]["active"])
            self.assertEqual(json.loads(core._gemini_state_path.read_text()), {"enabled": True})
            self.assertNotIn(settings.gemini_api_key, core._gemini_state_path.read_text())

            response = client.post("/api/gemini", json={"enabled": False}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json["gemini"]["active"])
            self.assertEqual(json.loads(core._gemini_state_path.read_text()), {"enabled": False})

    def test_cannot_enable_without_key(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(Settings.from_env(), gemini_enabled=False, gemini_api_key=None)
            judge = GeminiJudge(settings)
            core = Tukuru.__new__(Tukuru)
            core.s = settings
            core.judge = judge
            core._gemini_state_path = Path(temp) / "gemini_state.json"
            response = create_web(core).test_client().post(
                "/api/gemini", json={"enabled": True}, headers={"X-Requested-With": "tukuru"})
            self.assertEqual(response.status_code, 400)
            self.assertFalse(core._gemini_state_path.exists())

    def test_result_from_before_disable_is_not_applied(self):
        settings = replace(Settings.from_env(), gemini_enabled=False, gemini_api_key=None)
        judge = GeminiJudge(settings)
        judge._types = SimpleNamespace(
            Part=SimpleNamespace(from_bytes=lambda **kw: object()),
            AutomaticFunctionCallingConfig=lambda **kw: None,
            GenerateContentConfig=lambda **kw: None,
        )
        verdict = Verdict(state="normal", confidence=0.9, action="sin_toma", product="", place="ninguno",
                          looked_around=False, reasoning="Sin toma visible")
        def finish_after_disable(**kw):
            judge.set_enabled(False)
            return SimpleNamespace(parsed=verdict, usage_metadata=None)

        generate = Mock(side_effect=finish_after_disable)
        judge._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        judge._errors_mod = SimpleNamespace(ClientError=type("ClientError", (Exception,), {}),
                                            ServerError=type("ServerError", (Exception,), {}))
        callback = Mock()
        judge.disabled_reason = None
        judge._run("CAM-01-1", [(0.0, b"imagen_falsa")], {}, callback, 0)
        callback.assert_called_once_with("CAM-01-1", None, "gemini_desactivado")
        generate.assert_called_once()

    def test_queued_review_does_not_send_after_disable(self):
        settings = replace(Settings.from_env(), gemini_enabled=False)
        judge = GeminiJudge(settings)
        generate = Mock()
        judge._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        callback = Mock()
        judge._run("CAM-01-1", [(0.0, b"imagen_falsa")], {}, callback, 0)
        generate.assert_not_called()
        callback.assert_called_once_with("CAM-01-1", None, "gemini_desactivado")


if __name__ == "__main__":
    unittest.main()
