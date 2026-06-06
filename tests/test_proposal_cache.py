from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.depth_pro_witness import load_depth_pro_proposal_cache
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_depth_pro_cache, write_fake_vggt_cache, write_ppm_sequence


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

    def test_vggt_replay_cache_writes_manifest_counts_and_truth_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            cache_dir = write_fake_vggt_cache(root / "cache", frame_ids=(0, 1))
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
            vggt = load_vggt_proposal_cache(cache_dir)
            statuses = write_teacher_statuses(run_dir, failures, vggt_result=vggt)

            result = write_proposal_cache(
                run_dir,
                teacher_statuses=statuses,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                debug_geometry_mode="none",
                vggt_result=vggt,
            )

            manifest = json.loads((run_dir / result.manifest_path).read_text(encoding="utf-8"))
            self.assertTrue((run_dir / "proposals" / "vggt_cameras.jsonl").is_file())
            self.assertTrue((run_dir / "proposals" / "vggt_depths.npz").is_file())
            self.assertEqual(manifest["teacher_counts"]["vggt_camera_proposals"], 2)
            self.assertEqual(manifest["teacher_counts"]["vggt_depth_proposals"], 2)
            self.assertEqual(manifest["truth_boundary"]["label_type"], "teacher_pseudo")
            self.assertFalse(manifest["truth_boundary"]["measured_geometry"])
            self.assertTrue(result.depth_proposal_available)
            self.assertEqual(result.geometry_source, "vggt")

    def test_depth_pro_replay_cache_writes_streams_counts_and_truth_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            cache_dir = write_fake_depth_pro_cache(root / "cache", frame_ids=(0, 1))
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
            depth_pro = load_depth_pro_proposal_cache(cache_dir)
            statuses = write_teacher_statuses(run_dir, failures, depth_pro_result=depth_pro)

            result = write_proposal_cache(
                run_dir,
                teacher_statuses=statuses,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                debug_geometry_mode="none",
                depth_pro_result=depth_pro,
            )

            manifest = json.loads((run_dir / result.manifest_path).read_text(encoding="utf-8"))
            self.assertTrue((run_dir / "proposals" / "depth_pro_cameras.jsonl").is_file())
            self.assertTrue((run_dir / "proposals" / "depth_pro_depths.npz").is_file())
            self.assertEqual(manifest["teacher_counts"]["depth_pro_camera_proposals"], 2)
            self.assertEqual(manifest["teacher_counts"]["depth_pro_depth_proposals"], 2)
            self.assertEqual(manifest["truth_boundary"]["label_type"], "teacher_pseudo")
            self.assertFalse(manifest["truth_boundary"]["measured_geometry"])
            self.assertFalse(result.depth_proposal_available)
            self.assertEqual(result.geometry_source, "depth_pro_diagnostic_no_global_pose")


if __name__ == "__main__":
    unittest.main()
