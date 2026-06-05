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

        live_replay_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "live-replay-recording", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(live_replay_result.returncode, 0, live_replay_result.stderr)
        self.assertIn("--target-fps", live_replay_result.stdout)
        self.assertIn("--max-capture-queue", live_replay_result.stdout)
        self.assertIn("--max-map-queue", live_replay_result.stdout)
        self.assertIn("--drop-policy", live_replay_result.stdout)
        self.assertIn("--pixel-stride", live_replay_result.stdout)
        self.assertIn("--export-mesh-chunks", live_replay_result.stdout)
        self.assertIn("--mesh-format", live_replay_result.stdout)
        self.assertIn("--mesh-update-interval-frames", live_replay_result.stdout)
        self.assertIn("--mesh-min-weight", live_replay_result.stdout)
        self.assertIn("cpu-sparse", live_replay_result.stdout)

        rgb_teacher_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "map-rgb-teacher", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(rgb_teacher_result.returncode, 0, rgb_teacher_result.stderr)
        self.assertIn("--input", rgb_teacher_result.stdout)
        self.assertIn("--teacher", rgb_teacher_result.stdout)
        self.assertIn("--teacher-window-size", rgb_teacher_result.stdout)
        self.assertIn("--export-mesh-chunks", rgb_teacher_result.stdout)

        rgb_student_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "map-rgb-student", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(rgb_student_result.returncode, 0, rgb_student_result.stderr)
        self.assertIn("--checkpoint", rgb_student_result.stdout)
        self.assertIn("--clip-length", rgb_student_result.stdout)
        self.assertIn("--rgb-only", rgb_student_result.stdout)

        teacher_temporal_cache_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atlas3r",
                "inspect",
                "teacher-temporal-cache",
                "--help",
            ],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            teacher_temporal_cache_result.returncode,
            0,
            teacher_temporal_cache_result.stderr,
        )
        self.assertIn("--cache", teacher_temporal_cache_result.stdout)

        capture_adapters_result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "capture-adapters", "list"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(capture_adapters_result.returncode, 0, capture_adapters_result.stderr)
        self.assertIn("opencv-camera", capture_adapters_result.stdout)
        self.assertIn("replay-recording", capture_adapters_result.stdout)

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
