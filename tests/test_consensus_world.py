from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.camera_scale_ledger import write_camera_scale_ledgers
from atlas3r.offline.consensus_world import write_consensus_world_state
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from tests.helpers import write_ppm_sequence


class ConsensusWorldTest(unittest.TestCase):
    def test_unresolved_statuses_and_scale_claim_blocker_are_explicit(self) -> None:
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
            proposals = write_proposal_cache(
                run_dir,
                teacher_statuses=statuses,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                debug_geometry_mode="none",
            )
            ledgers = write_camera_scale_ledgers(
                run_dir,
                camera=frames.camera,
                frame_count=len(frames.records),
                debug_geometry_mode="none",
            )

            result = write_consensus_world_state(
                run_dir,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                proposal_cache=proposals,
                ledgers=ledgers,
            )

            world = json.loads((run_dir / result.world_state_path).read_text(encoding="utf-8"))
            scale = json.loads((run_dir / ledgers.scale_ledger_path).read_text(encoding="utf-8"))
            self.assertEqual(world["pose_status"], "unknown")
            self.assertEqual(world["depth_status"], "missing")
            self.assertEqual(world["map_status"], "none")
            self.assertFalse(scale["physical_accuracy_allowed"])
            self.assertEqual(scale["scale_source"], "depth_pro_vggt_soft_metric_prior")
            self.assertEqual(scale["legacy_scale_source"], "unanchored_rgb_prior")
            self.assertEqual(scale["scale_status"], "soft_metric_unanchored")


if __name__ == "__main__":
    unittest.main()
