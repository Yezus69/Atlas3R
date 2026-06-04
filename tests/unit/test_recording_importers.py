import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.forge.clip_cache import (
    CLIP_CACHE_FORMAT_NAME,
    CLIP_CACHE_FORMAT_VERSION,
    CLIP_CACHE_MANIFEST_FILENAME,
    write_clip_payload,
    write_json_file,
)
from atlas3r.recording.importers import (
    ClipCacheRecordingImportConfig,
    TumRecordingImportConfig,
    recording_from_clip_cache,
    recording_from_tum_manifest,
)
from atlas3r.recording.schema import load_recording


class RecordingImportersTest(unittest.TestCase):
    def test_from_clip_cache_writes_valid_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            _write_clip_cache(cache)
            output = root / "recording"

            result = recording_from_clip_cache(
                ClipCacheRecordingImportConfig(
                    clip_cache=cache,
                    output=output,
                    dedupe_frame_id=True,
                )
            )
            recording = load_recording(output)
            depth_exists = (output / "depth" / "frame_000001.npz").is_file()

        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(len(recording.frames), 3)
        self.assertTrue(depth_exists)

    def test_from_tum_manifest_references_external_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tum_root = root / "tum"
            tum_root.mkdir()
            (tum_root / "rgb").mkdir()
            (tum_root / "depth").mkdir()
            (tum_root / "rgb" / "0.png").write_bytes(b"not-used-by-import")
            (tum_root / "depth" / "0.png").write_bytes(b"not-used-by-import")
            manifest_path = root / "tum_manifest.json"
            manifest_path.write_text(json.dumps(_tum_manifest(tum_root)), encoding="utf-8")
            output = root / "recording"

            result = recording_from_tum_manifest(
                TumRecordingImportConfig(
                    manifest=manifest_path,
                    output=output,
                    split="val",
                    max_frames=1,
                    width=4,
                    height=3,
                )
            )
            recording = load_recording(output)

        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(recording.frames[0].depth_scale, 5000.0)
        self.assertIn("tum_root", recording.manifest["external_roots"])  # type: ignore[operator]


def _write_clip_cache(cache: Path) -> None:
    payload = _clip_payload()
    write_clip_payload(
        cache / "clips" / "clip_000000.npz", payload, clip_length=3, height=4, width=4
    )
    manifest = {
        "clip_count": 1,
        "clip_length": 3,
        "clips": [
            {
                "center_index": 1,
                "clip_id": 0,
                "frame_ids": [0, 1, 2],
                "payload_path": "clips/clip_000000.npz",
                "timestamps_s": [0.0, 0.033, 0.066],
            }
        ],
        "format_name": CLIP_CACHE_FORMAT_NAME,
        "format_version": CLIP_CACHE_FORMAT_VERSION,
        "image_height": 4,
        "image_width": 4,
        "max_frame_gap_s": 0.12,
        "source_dataset_name": "unit",
        "source_manifest_path": "unit_manifest.json",
        "source_sequence_name": "unit_sequence",
        "split": "val",
        "stride": 1,
        "teacher_source_metadata": {"depth_source": "unit"},
        "truth_boundary": {
            "accuracy_report": False,
            "diagnostic_only": True,
            "performance_report": False,
            "teacher_source": "tum_rgbd_sensor_depth_pose",
        },
    }
    write_json_file(cache / CLIP_CACHE_MANIFEST_FILENAME, manifest)


def _clip_payload() -> dict[str, np.ndarray]:
    K = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {
        "K": np.repeat(K[np.newaxis, :, :], 3, axis=0),
        "T_world_camera": np.repeat(np.eye(4, dtype=np.float32)[np.newaxis, :, :], 3, axis=0),
        "center_index": np.array(1, dtype=np.int32),
        "depth_m": np.ones((3, 4, 4), dtype=np.float32),
        "frame_ids": np.array([0, 1, 2], dtype=np.int32),
        "images_rgb_u8": np.zeros((3, 4, 4, 3), dtype=np.uint8),
        "timestamps_s": np.array([0.0, 0.033, 0.066], dtype=np.float64),
        "valid_depth_mask": np.ones((3, 4, 4), dtype=np.bool_),
    }


def _tum_manifest(root: Path) -> dict[str, object]:
    return {
        "coordinate_frame": "x_right_y_down_z_forward",
        "dataset_name": "TUM RGB-D unit",
        "depth": {"scale": 5000.0},
        "format_name": "atlas3r_tum_rgbd_manifest",
        "format_version": 1,
        "frame_count": 1,
        "frames": [
            {
                "T_world_camera": np.eye(4, dtype=np.float32).tolist(),
                "depth_path": "depth/0.png",
                "depth_timestamp_s": 0.0,
                "frame_id": 0,
                "pose_timestamp_s": 0.0,
                "rgb_path": "rgb/0.png",
                "rgb_timestamp_s": 0.0,
                "split": "val",
            }
        ],
        "image": {"height": 480, "width": 640},
        "intrinsics": {
            "K": [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
            "source": "unit",
        },
        "root_path": str(root),
        "sequence_name": "unit_sequence",
    }


if __name__ == "__main__":
    unittest.main()
