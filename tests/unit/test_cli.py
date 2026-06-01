import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import atlas3r  # noqa: E402


class Atlas3RCliTest(unittest.TestCase):
    def test_package_import_and_cli_help(self) -> None:
        self.assertIsInstance(atlas3r.__version__, str)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)

        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Atlas3R", result.stdout)
        self.assertIn("smoke", result.stdout)
        self.assertIn("profile", result.stdout)


if __name__ == "__main__":
    unittest.main()
