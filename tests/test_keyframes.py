from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.run_manifest import ensure_run_tree
from tests.helpers import write_ppm_sequence


class KeyframeSelectorTest(unittest.TestCase):
    def test_keyframe_selection_records_reasons_and_no_pose_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=5)
            failures = []
            frames = build_frame_cache(input_dir, run_dir, max_frames=5, failure_points=failures)

            result = select_keyframes(
                frames.records,
                run_dir,
                keyframe_stride=2,
                keyframe_max_count=3,
                failure_points=failures,
            )

            self.assertEqual(result.status, "complete")
            self.assertGreaterEqual(len(result.keyframes), 1)
            for keyframe in result.keyframes:
                self.assertIn("no_pose_assumed", keyframe.reasons)
                self.assertEqual(keyframe.pose_assumption, "none")


if __name__ == "__main__":
    unittest.main()
