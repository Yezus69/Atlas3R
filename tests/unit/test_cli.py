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

        recording_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "recording", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(recording_result.returncode, 0, recording_result.stderr)
        self.assertIn("validate", recording_result.stdout)
        self.assertIn("from-tum", recording_result.stdout)
        self.assertIn("from-clip-cache", recording_result.stdout)
        self.assertIn("from-sensor-folder", recording_result.stdout)

        fuse_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "fuse-recording", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(fuse_result.returncode, 0, fuse_result.stderr)
        self.assertIn("--recording", fuse_result.stdout)
        self.assertIn("--export-mesh", fuse_result.stdout)
        self.assertIn("--mode", fuse_result.stdout)
        self.assertIn("--backend", fuse_result.stdout)
        self.assertIn("cpu-sparse", fuse_result.stdout)

        stress_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "sparse-tsdf-stress", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(stress_result.returncode, 0, stress_result.stderr)
        self.assertIn("--room-size-m", stress_result.stdout)
        self.assertIn("--voxel-size-m", stress_result.stdout)

        invalid_backend_combo = subprocess.run(
            [
                sys.executable,
                "-m",
                "atlas3r",
                "runtime",
                "fuse-recording",
                "--recording",
                str(ROOT / "missing-recording"),
                "--output",
                str(ROOT / "missing-output"),
                "--backend",
                "cpu-persistent",
            ],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(invalid_backend_combo.returncode, 2)
        self.assertIn("backend: only valid with mode incremental", invalid_backend_combo.stderr)


if __name__ == "__main__":
    unittest.main()
