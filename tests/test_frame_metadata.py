from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas3r.input.metadata import ImageMetadata, metadata_summary
from atlas3r.offline.frame_cache import build_frame_cache
from tests.helpers import write_ppm_sequence


class FrameMetadataTest(unittest.TestCase):
    def test_missing_exif_summary_is_explicit(self) -> None:
        summary = metadata_summary(
            (
                ImageMetadata(
                    source_uri="frame.jpg",
                    width=640,
                    height=480,
                    metadata_status="no_useful_exif",
                ),
            )
        )

        self.assertEqual(summary["metadata_quality"], "missing")
        self.assertEqual(summary["frame_count_with_exif"], 0)
        self.assertFalse(summary["physical_accuracy_claim"])
        self.assertTrue(summary["intrinsics_proposal_only"])

    def test_fake_focal_metadata_is_intrinsics_proposal_only(self) -> None:
        summary = metadata_summary(
            (
                ImageMetadata(
                    source_uri="frame.jpg",
                    width=4032,
                    height=3024,
                    focal_length_mm=6.8,
                    focal_length_35mm=26.0,
                    camera_make="Example",
                    camera_model="Phone",
                    exif_available=True,
                    metadata_status="available",
                ),
            )
        )

        self.assertEqual(summary["metadata_quality"], "focal_metadata_available")
        self.assertEqual(summary["camera_make_models"], ["Example Phone"])
        self.assertEqual(summary["focal_mm_values"], [6.8])
        self.assertTrue(summary["intrinsics_proposal_only"])
        self.assertFalse(summary["physical_accuracy_claim"])

    def test_frame_cache_writes_metadata_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = root / "run"
            write_ppm_sequence(input_dir, count=2, width=4, height=3)
            failures = []

            result = build_frame_cache(input_dir, run_dir, max_frames=2, failure_points=failures)

            self.assertEqual(result.metadata_summary_path, "frames/metadata_summary.json")
            summary_path = run_dir / result.metadata_summary_path
            self.assertTrue(summary_path.is_file())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["frame_count"], 2)
            self.assertEqual(payload["metadata_quality"], "missing")


if __name__ == "__main__":
    unittest.main()
