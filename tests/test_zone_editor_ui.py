"""Prueba de gestos del editor con Node cuando esta disponible."""

import shutil
import subprocess
import unittest
from pathlib import Path


class ZoneEditorUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node no esta instalado")
    def test_draw_move_and_resize_with_pointer(self):
        script = Path(__file__).with_name("zone_editor_smoke.js")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("draw, move, resize, select, delete: OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
