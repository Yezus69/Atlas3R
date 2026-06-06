from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.best_map_selection import write_best_world_map
from atlas3r.offline.depth_pro_witness import load_depth_pro_proposal_cache
from atlas3r.offline.disagreement import write_teacher_disagreement
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.fused_world_map import FusedWorldMapOptions, write_fused_world_map
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.map_consistency_optimizer import MapConsistencyOptimizerResult
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_depth_pro_cache, write_fake_vggt_cache, write_ppm_sequence


class BestMapSelectionTest(unittest.TestCase):
    def test_raw_map_selected_when_optimizer_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _prepared_run(root)
            options = FusedWorldMapOptions(
                export_world_map=True,
                point_stride=1,
                min_confidence=0.0,
                max_relative_disagreement=1.0,
                write_occupancy=True,
                write_observed_mesh=True,
            )
            raw = write_fused_world_map(
                run["run_dir"],
                input_path=str(run["input_dir"]),
                frame_cache=run["frames"],
                keyframes=run["keyframes"].keyframes,
                proposal_cache=run["proposals"],
                disagreement=run["disagreement"],
                options=options,
                failure_points=run["failures"],
            )

            best = write_best_world_map(
                run["run_dir"],
                input_path=str(run["input_dir"]),
                frame_cache=run["frames"],
                keyframes=run["keyframes"].keyframes,
                proposal_cache=run["proposals"],
                disagreement=run["disagreement"],
                raw_world_map=raw,
                optimizer=MapConsistencyOptimizerResult(status="disabled"),
                map_options=options,
                export_best_world_map=True,
                failure_points=run["failures"],
            )

            self.assertEqual(best.selected_source, "raw_consensus")
            self.assertGreater(best.point_count, 0)
            self.assertGreater(best.occupied_voxel_count, 0)
            self.assertGreater(best.mesh_triangle_count, 0)
            manifest = json.loads(
                (run["run_dir"] / "world_map_best" / "world_map_manifest.json").read_text()
            )
            self.assertEqual(manifest["selected_best_map_source"], "raw_consensus")
            self.assertFalse(manifest["truth_boundary"]["physical_accuracy_claim"])
            self.assertFalse(manifest["truth_boundary"]["training_quality"])
            self.assertTrue((run["run_dir"] / "world_map_best" / "topdown_preview.svg").is_file())


def _prepared_run(root: Path) -> dict[str, object]:
    input_dir = root / "input"
    run_dir = ensure_run_tree(root / "run")
    write_ppm_sequence(input_dir, count=2, width=6, height=4)
    failures = []
    frames = build_frame_cache(input_dir, run_dir, max_frames=2, failure_points=failures)
    keyframes = select_keyframes(
        frames.records,
        run_dir,
        keyframe_stride=1,
        keyframe_max_count=2,
        failure_points=failures,
    )
    vggt = load_vggt_proposal_cache(
        write_fake_vggt_cache(root / "vggt_cache", frame_ids=(0, 1), width=6, height=4)
    )
    depth_pro = load_depth_pro_proposal_cache(
        write_fake_depth_pro_cache(root / "depth_pro_cache", frame_ids=(0, 1), width=6, height=4)
    )
    statuses = write_teacher_statuses(
        run_dir, failures, vggt_result=vggt, depth_pro_result=depth_pro
    )
    proposals = write_proposal_cache(
        run_dir,
        teacher_statuses=statuses,
        frame_records=frames.records,
        keyframes=keyframes.keyframes,
        debug_geometry_mode="none",
        vggt_result=vggt,
        depth_pro_result=depth_pro,
    )
    disagreement = write_teacher_disagreement(run_dir, proposal_cache=proposals)
    return {
        "input_dir": input_dir,
        "run_dir": run_dir,
        "frames": frames,
        "keyframes": keyframes,
        "proposals": proposals,
        "disagreement": disagreement,
        "failures": failures,
    }


if __name__ == "__main__":
    unittest.main()
