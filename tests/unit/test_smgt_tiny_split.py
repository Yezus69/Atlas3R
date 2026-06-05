import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.training.smgt_tiny_dataset import SMGTTinyTeacherCacheDataset
from atlas3r.training.smgt_tiny_split import (
    build_smgt_tiny_split_manifest,
    load_smgt_tiny_split_manifest,
    split_indices,
    validate_smgt_tiny_split_manifest,
    write_smgt_tiny_split_manifest,
)
from atlas3r.training.teacher_temporal_cache import write_teacher_temporal_cache
from atlas3r.training.torch_runtime import torch_available
from tests.unit.test_rgb_teacher_stitching import _observation


class SMGTTinySplitTest(unittest.TestCase):
    def test_split_manifest_is_deterministic_and_non_overlapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _write_cache(Path(tmp), frame_count=30, clip_length=4, clip_stride=2)
            first = build_smgt_tiny_split_manifest(cache, val_split=0.2, heldout_split=0.2)
            second = build_smgt_tiny_split_manifest(cache, val_split=0.2, heldout_split=0.2)

        self.assertEqual(first, second)
        splits = first["splits"]  # type: ignore[index]
        train = set(splits["train"]["frame_ids"])  # type: ignore[index]
        val = set(splits["val"]["frame_ids"])  # type: ignore[index]
        heldout = set(splits["heldout"]["frame_ids"])  # type: ignore[index]
        self.assertFalse(train & val)
        self.assertFalse(train & heldout)
        self.assertFalse(val & heldout)
        self.assertGreater(splits["heldout"]["clip_count"], 0)  # type: ignore[index]

    def test_split_manifest_serializes_loads_and_heldout_clips_load(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = _write_cache(root, frame_count=30, clip_length=4, clip_stride=2)
            manifest = write_smgt_tiny_split_manifest(
                root / "split",
                cache,
                val_split=0.2,
                heldout_split=0.2,
            )
            loaded = load_smgt_tiny_split_manifest(root / "split" / "smgt_tiny_split_manifest.json")
            heldout = SMGTTinyTeacherCacheDataset(cache, indices=split_indices(loaded, "heldout"))
            sample = heldout[0]

        self.assertEqual(loaded, manifest)
        self.assertGreater(len(heldout), 0)
        self.assertEqual(sample["images_rgb"].shape[0], 4)

    def test_bad_split_overlap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _write_cache(Path(tmp), frame_count=30, clip_length=4, clip_stride=2)
            manifest = build_smgt_tiny_split_manifest(
                cache,
                val_split=0.2,
                heldout_split=0.2,
            )
            splits = manifest["splits"]  # type: ignore[index]
            train_frame = splits["train"]["frame_ids"][0]  # type: ignore[index]
            splits["heldout"]["frame_ids"].append(train_frame)  # type: ignore[index]
            splits["heldout"]["frame_count"] += 1  # type: ignore[index]

            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_smgt_tiny_split_manifest(manifest)

    def test_too_small_heldout_split_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _write_cache(Path(tmp), frame_count=8, clip_length=4, clip_stride=2)
            with self.assertRaisesRegex(ValueError, "heldout split has no clips"):
                build_smgt_tiny_split_manifest(cache, val_split=0.0, heldout_split=0.2)


def _write_cache(root: Path, *, frame_count: int, clip_length: int, clip_stride: int) -> Path:
    observations = [
        _observation(index, np.asarray([index * 0.05, 0.0, 0.0], dtype=np.float32))
        for index in range(frame_count)
    ]
    cache = root / "teacher_temporal_cache"
    write_teacher_temporal_cache(
        cache,
        observations,
        source_rgb_teacher_run=root / "run",
        teacher_metadata={"model_source": "fixture_vggt", "resolved_device": "cpu"},
        stitch_mode="sim3-overlap",
        metric_scale_source="teacher_scale_unverified",
        clip_length=clip_length,
        clip_stride=clip_stride,
    )
    return cache


if __name__ == "__main__":
    unittest.main()
