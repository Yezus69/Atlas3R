import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.training.smgt_tiny_dataset import (
    SMGTTinyTeacherCacheDataset,
    smgt_tiny_train_val_indices,
)
from atlas3r.training.teacher_temporal_cache import write_teacher_temporal_cache
from atlas3r.training.torch_runtime import torch_available
from tests.unit.test_rgb_teacher_stitching import _observation


class SMGTTinyDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")

    def test_teacher_cache_sample_converts_to_smgt_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _write_cache(Path(tmp))
            dataset = SMGTTinyTeacherCacheDataset(cache)
            sample = dataset[0]

        self.assertEqual(sample["images_rgb"].shape, (2, 3, 4, 4))
        self.assertEqual(sample["K"].shape, (2, 3, 3))
        self.assertEqual(sample["target"]["depth_m"].shape, (2, 4, 4))
        self.assertLessEqual(float(sample["depth_target_weight"]), 0.25)
        self.assertLessEqual(float(sample["pose_target_weight"]), 0.25)

    def test_debug_override_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _write_cache(Path(tmp))
            dataset = SMGTTinyTeacherCacheDataset(
                cache, debug_subset_clips=2, pseudo_weight_override=1.0
            )
            train, val = smgt_tiny_train_val_indices(len(dataset), val_split=0.5)
            sample = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(train, (0,))
        self.assertEqual(val, (1,))
        self.assertEqual(float(sample["depth_target_weight"]), 1.0)
        self.assertEqual(float(sample["pose_target_weight"]), 1.0)

    def test_bad_arrays_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _write_cache(Path(tmp))
            clip = cache / "clips" / "clip_000000.npz"
            with np.load(clip, allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            arrays["depth_m"] = -np.ones_like(arrays["depth_m"], dtype=np.float32)
            np.savez_compressed(clip, **arrays)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                dataset = SMGTTinyTeacherCacheDataset(cache)
                _ = dataset[0]


def _write_cache(root: Path) -> Path:
    observations = [
        _observation(index, np.asarray([index * 0.1, 0.0, 0.0], dtype=np.float32))
        for index in range(4)
    ]
    cache = root / "cache"
    write_teacher_temporal_cache(
        cache,
        observations,
        source_rgb_teacher_run=root / "run",
        teacher_metadata={"model_source": "fixture_vggt", "resolved_device": "cpu"},
        stitch_mode="sim3-overlap",
        metric_scale_source="teacher_scale_unverified",
        clip_length=2,
        clip_stride=1,
    )
    return cache


if __name__ == "__main__":
    unittest.main()
