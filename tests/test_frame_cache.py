from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.frame_cache import InputDataError, build_frame_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from tests.helpers import write_ppm_sequence


class FrameCacheTest(unittest.TestCase):
    def test_frame_index_writes_stable_ids_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=3)
            failures = []

            result = build_frame_cache(input_dir, run_dir, max_frames=3, failure_points=failures)

            self.assertEqual(result.status, "complete")
            self.assertEqual([record.frame_id for record in result.records], [0, 1, 2])
            self.assertEqual(result.records[0].frame_path, "frames/images/frame_000000.ppm")
            self.assertEqual(result.records[0].camera.source, "guessed_from_image_size")
            for record in result.records:
                quality = record.quality
                self.assertTrue(math.isfinite(quality.blur_score))
                self.assertTrue(math.isfinite(quality.exposure_score))
                self.assertTrue(math.isfinite(quality.visual_change_score))

    def test_missing_input_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = ensure_run_tree(Path(tmp) / "run")
            with self.assertRaisesRegex(InputDataError, "input path does not exist"):
                build_frame_cache(
                    Path(tmp) / "missing",
                    run_dir,
                    max_frames=1,
                    failure_points=[],
                )


if __name__ == "__main__":
    unittest.main()
