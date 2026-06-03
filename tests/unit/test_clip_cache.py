import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.forge.clip_cache import (
    CLIP_CACHE_FORMAT_NAME,
    CLIP_CACHE_FORMAT_VERSION,
    read_clip_payload,
    validate_clip_cache_manifest,
    write_clip_payload,
)


class ClipCacheTest(unittest.TestCase):
    def test_writer_reader_validates_shapes_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload_path = root / "clips" / "clip_000000.npz"
            payload = _valid_payload()
            write_clip_payload(payload_path, payload, clip_length=3, height=2, width=4)
            manifest = _valid_manifest()
            validate_clip_cache_manifest(manifest, cache_root=root, validate_payloads=True)

            manifest_path = root / "atlas3r_clip_cache_manifest.json"
            manifest_path.write_text(
                __import__("json").dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            loaded = read_clip_payload(manifest_path)

        self.assertEqual(tuple(loaded["images_rgb_u8"].shape), (3, 2, 4, 3))
        self.assertEqual(loaded["valid_depth_mask"].dtype, np.bool_)

    def test_manifest_rejects_payload_path_escape(self) -> None:
        manifest = _valid_manifest()
        manifest["clips"][0]["payload_path"] = "../bad.npz"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "unsafe path"):
            validate_clip_cache_manifest(manifest, cache_root=".", validate_payloads=False)


def _valid_manifest() -> dict[str, object]:
    return {
        "format_name": CLIP_CACHE_FORMAT_NAME,
        "format_version": CLIP_CACHE_FORMAT_VERSION,
        "source_dataset_name": "fixture",
        "source_sequence_name": "fixture",
        "source_manifest_path": "manifest.json",
        "split": "train",
        "clip_length": 3,
        "stride": 1,
        "image_width": 4,
        "image_height": 2,
        "max_frame_gap_s": 0.12,
        "clip_count": 1,
        "clips": [
            {
                "clip_id": 0,
                "payload_path": "clips/clip_000000.npz",
                "frame_ids": [0, 1, 2],
                "timestamps_s": [1.0, 1.03, 1.06],
                "center_index": 1,
            }
        ],
        "truth_boundary": {
            "diagnostic_only": True,
            "accuracy_report": False,
            "performance_report": False,
            "teacher_source": "tum_rgbd_sensor_depth_pose",
        },
    }


def _valid_payload() -> dict[str, np.ndarray]:
    K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {
        "images_rgb_u8": np.zeros((3, 2, 4, 3), dtype=np.uint8),
        "depth_m": np.ones((3, 2, 4), dtype=np.float32),
        "valid_depth_mask": np.ones((3, 2, 4), dtype=np.bool_),
        "K": np.repeat(K[np.newaxis, :, :], 3, axis=0),
        "T_world_camera": np.repeat(np.eye(4, dtype=np.float32)[np.newaxis, :, :], 3, axis=0),
        "frame_ids": np.array([0, 1, 2], dtype=np.int32),
        "timestamps_s": np.array([1.0, 1.03, 1.06], dtype=np.float64),
        "center_index": np.array(1, dtype=np.int32),
    }


if __name__ == "__main__":
    unittest.main()
