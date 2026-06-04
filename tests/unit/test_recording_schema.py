import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    load_recording,
    recording_truth_boundary,
    validate_recording_folder,
    write_recording_files,
)


class RecordingSchemaTest(unittest.TestCase):
    def test_validate_recording_manifest_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_recording(root)

            loaded = load_recording(root)
            summary = validate_recording_folder(root)

        self.assertEqual(len(loaded.frames), 2)
        self.assertEqual(summary["format_name"], "atlas3r_recording_validation")
        self.assertEqual(summary["frame_count"], 2)
        self.assertTrue(summary["depth_present"])
        self.assertTrue(summary["pose_present"])

    def test_rejects_frame_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_assets(root)
            manifest = _manifest(frame_count=1)
            frame = _frame(0)
            frame["rgb_path"] = "../escape.npz"

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                write_recording_files(root, manifest=manifest, frames=[frame])


def _write_fixture_recording(root: Path) -> None:
    _write_assets(root)
    write_recording_files(root, manifest=_manifest(frame_count=2), frames=[_frame(0), _frame(1)])


def _write_assets(root: Path) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    valid = np.ones((4, 4), dtype=np.bool_)
    for frame_id in (0, 1):
        np.savez(root / "rgb" / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
        np.savez(
            root / "depth" / f"frame_{frame_id:06d}.npz", depth_m=depth, valid_depth_mask=valid
        )


def _manifest(*, frame_count: int) -> dict[str, object]:
    return {
        "capture_metadata": {"source": "unit"},
        "coordinate_frame": RECORDING_COORDINATE_FRAME,
        "depth_present": True,
        "external_roots": {},
        "format_name": RECORDING_FORMAT_NAME,
        "format_version": RECORDING_FORMAT_VERSION,
        "frame_count": frame_count,
        "height": 4,
        "known_calibration_metadata": {"source": "unit"},
        "pose_present": True,
        "source_dataset": "unit",
        "source_sequence": "unit_sequence",
        "truth_boundary": recording_truth_boundary(depth_present=True, pose_present=True),
        "width": 4,
    }


def _frame(frame_id: int) -> dict[str, object]:
    return {
        "K": [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
        "T_world_camera": [
            [1.0, 0.0, 0.0, float(frame_id) * 0.02],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "camera_center_world_m": [float(frame_id) * 0.02, 0.0, 0.0],
        "depth_path": f"depth/frame_{frame_id:06d}.npz",
        "frame_id": frame_id,
        "rgb_path": f"rgb/frame_{frame_id:06d}.npz",
        "source_metadata": {"fixture": True},
        "timestamp_s": float(frame_id) * 0.033,
    }


if __name__ == "__main__":
    unittest.main()
