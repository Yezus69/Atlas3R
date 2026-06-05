import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.training.teacher_temporal_cache import (
    MANIFEST_FILENAME,
    TeacherTemporalCacheDataset,
    inspect_teacher_temporal_cache,
    write_teacher_temporal_cache,
)
from tests.unit.test_rgb_teacher_stitching import _observation


class TeacherTemporalCacheTest(unittest.TestCase):
    def test_cache_manifest_clips_inspector_and_dataset_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = [
                _observation(index, np.asarray([index * 0.1, 0.0, 0.0], dtype=np.float32))
                for index in range(4)
            ]
            report = write_teacher_temporal_cache(
                root / "cache",
                observations,
                source_rgb_teacher_run=root / "run",
                teacher_metadata={
                    "model_source": "fixture_vggt",
                    "checkpoint": None,
                    "resolved_device": "cpu",
                },
                stitch_mode="sim3-overlap",
                metric_scale_source="teacher_scale_unverified",
                clip_length=2,
                clip_stride=1,
            )
            manifest = json.loads((root / "cache" / MANIFEST_FILENAME).read_text("utf-8"))
            clip_path = root / "cache" / "clips" / "clip_000000.npz"
            with np.load(clip_path, allow_pickle=False) as payload:
                frame_ids = np.asarray(payload["frame_ids"])
                rgb = np.asarray(payload["rgb_u8"])
                depth = np.asarray(payload["depth_m"])
            inspection = inspect_teacher_temporal_cache(root / "cache")
            dataset = TeacherTemporalCacheDataset(root / "cache")
            sample = dataset[0]

        self.assertTrue(report["validation_passed"])
        self.assertEqual(manifest["format_name"], "atlas3r_teacher_temporal_cache")
        self.assertEqual(manifest["clip_count"], 3)
        self.assertEqual(frame_ids.tolist(), [0, 1])
        self.assertEqual(rgb.shape, (2, 4, 4, 3))
        self.assertEqual(depth.shape, (2, 4, 4))
        self.assertEqual(inspection["clip_count"], 3)
        self.assertEqual(len(dataset), 3)
        self.assertIn("T_world_camera", sample)
        self.assertLessEqual(sample["depth_target_weight"], 0.25)
        self.assertLessEqual(sample["pose_target_weight"], 0.25)
        self.assertFalse(sample["truth_flags"]["measured_depth_used"])  # type: ignore[index]
        self.assertFalse(sample["truth_flags"]["measured_pose_used"])  # type: ignore[index]

    def test_cache_rejects_unsafe_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = [
                _observation(index, np.asarray([index * 0.1, 0.0, 0.0], dtype=np.float32))
                for index in range(2)
            ]
            write_teacher_temporal_cache(
                root / "cache",
                observations,
                source_rgb_teacher_run=root / "run",
                teacher_metadata={"model_source": "fixture_vggt", "resolved_device": "cpu"},
                stitch_mode="none",
                metric_scale_source="teacher_scale_unverified",
                clip_length=2,
                clip_stride=1,
            )
            manifest_path = root / "cache" / MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["source_paths_relative"] = ["../outside"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "safe relative path"):
                inspect_teacher_temporal_cache(root / "cache")


if __name__ == "__main__":
    unittest.main()
