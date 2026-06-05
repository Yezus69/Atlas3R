from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from tests.helpers import write_ppm_sequence


class ProposalCacheTest(unittest.TestCase):
    def test_unavailable_teachers_are_recorded_and_manifest_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=2)
            failures = []
            frames = build_frame_cache(input_dir, run_dir, max_frames=2, failure_points=failures)
            keyframes = select_keyframes(
                frames.records,
                run_dir,
                keyframe_stride=1,
                keyframe_max_count=2,
                failure_points=failures,
            )
            statuses = write_teacher_statuses(run_dir, failures)

            result = write_proposal_cache(
                run_dir,
                teacher_statuses=statuses,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                debug_geometry_mode="none",
            )

            manifest = json.loads((run_dir / result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["format_name"], "atlas3r_teacher_proposal_cache")
            self.assertTrue((run_dir / "proposals" / "depth_pro_proposals.jsonl").is_file())
            self.assertFalse(result.depth_proposal_available)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("cv2", sys.modules)


if __name__ == "__main__":
    unittest.main()
