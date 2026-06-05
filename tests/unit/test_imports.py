from __future__ import annotations

import subprocess
import sys
import unittest


class ImportSafetyTest(unittest.TestCase):
    def test_import_atlas3r_does_not_import_heavy_packages(self) -> None:
        code = (
            "import sys\n"
            "import atlas3r\n"
            "heavy={'torch','cv2','open3d','vggt','depth_pro','segment_anything'}\n"
            "loaded=heavy & set(sys.modules)\n"
            "print(sorted(loaded))\n"
            "raise SystemExit(1 if loaded else 0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_help_works(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("offline", result.stdout)
        self.assertIn("teachers", result.stdout)


if __name__ == "__main__":
    unittest.main()
