from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTest(unittest.TestCase):
    def test_small_cli_surface_works(self) -> None:
        smoke = subprocess.run(
            [sys.executable, "-m", "atlas3r", "smoke", "contracts"],
            check=False,
            capture_output=True,
            text=True,
        )
        teachers = subprocess.run(
            [sys.executable, "-m", "atlas3r", "teachers", "list"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertIn("contracts ok", smoke.stdout)
        self.assertEqual(teachers.returncode, 0, teachers.stderr)
        self.assertIn("depth_pro", teachers.stdout)

    def test_offline_inspect_video_writes_skeleton_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "capture.mp4"
            output = root / "run"
            video.write_bytes(b"not a real mp4")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "inspect-video",
                    "--input",
                    str(video),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "video_inspection.json").is_file())
            self.assertTrue((output / "quality_report_skeleton.json").is_file())

    def test_old_commands_are_not_advertised(self) -> None:
        help_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotIn("smgt-tiny", help_result.stdout)
        self.assertNotIn("map-rgb-student", help_result.stdout)
        self.assertNotIn("live-replay-recording", help_result.stdout)

    def test_offline_build_world_help_works(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "offline", "build-world", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--debug-geometry-mode", result.stdout)
        self.assertIn("--enable-vggt", result.stdout)
        self.assertIn("--vggt-proposal-cache", result.stdout)
        self.assertIn("--vggt-stitch-mode", result.stdout)
        self.assertIn("--enable-depth-pro", result.stdout)
        self.assertIn("--depth-pro-proposal-cache", result.stdout)
        self.assertIn("--export-world-map", result.stdout)
        self.assertIn("--map-point-stride", result.stdout)
        self.assertIn("--map-depth-source", result.stdout)
        self.assertIn("--map-write-observed-mesh", result.stdout)


if __name__ == "__main__":
    unittest.main()
