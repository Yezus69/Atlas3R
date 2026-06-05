import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    recording_truth_boundary,
    write_recording_files,
)
from atlas3r.runtime.capture_adapters import (
    CaptureAdapterConfigError,
    CaptureAdapterDependencyError,
    OpenCVCameraAdapter,
    ReplayRecordingAdapter,
    list_capture_adapters,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class CaptureAdaptersTest(unittest.TestCase):
    def test_runtime_capture_module_import_does_not_require_cv2(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "import atlas3r\n"
            "import atlas3r.runtime.capture_adapters\n"
            "unexpected = {'cv2'} & set(sys.modules)\n"
            "message = 'unexpected imports: ' + ', '.join(unexpected)\n"
            "raise SystemExit(message if unexpected else 0)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_opencv_status_and_missing_dependency_error_are_explicit(self) -> None:
        adapter = OpenCVCameraAdapter()
        with patch("atlas3r.runtime.capture_adapters.find_spec", return_value=None):
            status = adapter.status()
            self.assertFalse(status.available)
            self.assertIn("cv2", status.reason or "")
            self.assertIn("opencv-python", status.install_hint or "")
            with self.assertRaises(CaptureAdapterDependencyError):
                adapter.open({"camera_index": 0})

    def test_invalid_camera_config_fails_before_hardware_access(self) -> None:
        adapter = OpenCVCameraAdapter()

        with self.assertRaisesRegex(CaptureAdapterConfigError, "camera_index"):
            adapter.open({"camera_index": -1})

    def test_replay_recording_adapter_uses_recording_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp) / "recording"
            _write_capture_recording(recording, frame_count=2)

            adapter = ReplayRecordingAdapter()
            with adapter.open({"recording": recording}) as stream:
                frames = list(stream.frames())

        self.assertEqual(adapter.status().available, True)
        self.assertEqual([frame.frame_id for frame in frames], [0, 1])
        self.assertTrue(all(frame.frame_packet is None for frame in frames))
        self.assertTrue(all(frame.has_measured_depth for frame in frames))
        self.assertTrue(all(frame.has_measured_pose for frame in frames))

    def test_capture_adapter_registry_and_cli_listing(self) -> None:
        statuses = list_capture_adapters()
        names = {status.name for status in statuses}
        self.assertIn("opencv-camera", names)
        self.assertIn("replay-recording", names)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "capture-adapters", "list"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("name\tavailable\tdetails", result.stdout)
        self.assertIn("opencv-camera", result.stdout)
        self.assertIn("replay-recording", result.stdout)


def _write_capture_recording(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.ones((3, 3), dtype=np.float32)
    valid = np.ones((3, 3), dtype=np.bool_)
    frames = []
    for frame_id in range(frame_count):
        np.savez(root / "rgb" / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
        np.savez(
            root / "depth" / f"frame_{frame_id:06d}.npz",
            depth_m=depth,
            valid_depth_mask=valid,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = np.float32(frame_id * 0.05)
        frames.append(
            {
                "K": [[4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]],
                "T_world_camera": transform.tolist(),
                "camera_center_world_m": transform[:3, 3].tolist(),
                "depth_path": f"depth/frame_{frame_id:06d}.npz",
                "frame_id": frame_id,
                "rgb_path": f"rgb/frame_{frame_id:06d}.npz",
                "source_metadata": {"fixture": True},
                "timestamp_s": frame_id / 30.0,
            }
        )
    write_recording_files(
        root,
        manifest={
            "capture_metadata": {"source": "capture_adapter_unit"},
            "coordinate_frame": RECORDING_COORDINATE_FRAME,
            "depth_present": True,
            "external_roots": {},
            "format_name": RECORDING_FORMAT_NAME,
            "format_version": RECORDING_FORMAT_VERSION,
            "frame_count": len(frames),
            "height": 3,
            "known_calibration_metadata": {"source": "unit"},
            "pose_present": True,
            "source_dataset": "unit",
            "source_sequence": "capture_adapter_unit",
            "truth_boundary": recording_truth_boundary(depth_present=True, pose_present=True),
            "width": 3,
        },
        frames=frames,
    )


if __name__ == "__main__":
    unittest.main()
