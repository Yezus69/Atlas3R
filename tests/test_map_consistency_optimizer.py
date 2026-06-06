from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.depth_pro_witness import load_depth_pro_proposal_cache
from atlas3r.offline.disagreement import write_teacher_disagreement
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.map_consistency_optimizer import (
    MapConsistencyOptimizerOptions,
    compute_projection_consistency,
    fit_depth_scale_bias,
)
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_depth_pro_cache, write_fake_vggt_cache, write_ppm_sequence


class MapConsistencyOptimizerTest(unittest.TestCase):
    def test_scale_bias_recovers_inverse_depth_pro_alignment(self) -> None:
        vggt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        depth_pro = 2.0 * vggt + 0.4
        valid = np.ones(vggt.shape, dtype=np.bool_)

        estimate = fit_depth_scale_bias(
            vggt,
            depth_pro.astype(np.float32),
            valid,
            options=MapConsistencyOptimizerOptions(min_overlap_pixels=1),
        )

        self.assertTrue(estimate.accepted)
        self.assertAlmostEqual(estimate.scale, 0.5, places=3)
        self.assertAlmostEqual(estimate.bias_m, -0.2, places=3)
        self.assertLess(estimate.robust_loss_after, estimate.robust_loss_before)
        aligned = estimate.scale * depth_pro + estimate.bias_m
        self.assertFalse(np.isnan(aligned).any())

    def test_low_overlap_is_rejected_and_bounds_are_enforced(self) -> None:
        vggt = np.full((2, 2), 2.0, dtype=np.float32)
        depth_pro = np.full((2, 2), 8.0, dtype=np.float32)

        rejected = fit_depth_scale_bias(
            vggt,
            depth_pro,
            np.eye(2, dtype=np.bool_),
            options=MapConsistencyOptimizerOptions(min_overlap_pixels=3),
        )
        bounded = fit_depth_scale_bias(
            vggt,
            depth_pro,
            np.ones((2, 2), dtype=np.bool_),
            options=MapConsistencyOptimizerOptions(
                min_overlap_pixels=1,
                depth_scale_min=0.8,
                depth_scale_max=0.9,
                depth_bias_max_m=0.1,
            ),
        )

        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.rejection_reason, "valid_overlap_below_minimum")
        self.assertGreaterEqual(bounded.scale, 0.8)
        self.assertLessEqual(bounded.scale, 0.9)
        self.assertLessEqual(abs(bounded.bias_m), 0.1)

    def test_projection_diagnostics_returns_finite_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _proposal_run(Path(tmp), vggt_depth=2.0, depth_pro_depth=3.0)

            diagnostics = compute_projection_consistency(
                proposal_cache=run["proposals"],
                depth_records=run["disagreement"].consensus_depth_records,
                depth_arrays=run["disagreement"].consensus_depth_arrays,
                min_overlap_pixels=1,
            )

            self.assertGreater(diagnostics.projection_count, 0)
            self.assertTrue(np.isfinite(diagnostics.residuals_m).all())

    def test_build_world_writes_raw_and_optimized_map_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "run"
            write_ppm_sequence(input_dir, count=2, width=6, height=4)
            vggt_cache = write_fake_vggt_cache(
                root / "vggt_cache", frame_ids=(0, 1), width=6, height=4, depth_m=2.0
            )
            depth_pro_cache = write_fake_depth_pro_cache(
                root / "depth_pro_cache", frame_ids=(0, 1), width=6, height=4, depth_m=4.0
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "2",
                    "--keyframe-stride",
                    "1",
                    "--keyframe-max-count",
                    "2",
                    "--vggt-proposal-cache",
                    str(vggt_cache),
                    "--depth-pro-proposal-cache",
                    str(depth_pro_cache),
                    "--export-world-map",
                    "--map-depth-source",
                    "consensus",
                    "--map-point-stride",
                    "1",
                    "--map-min-confidence",
                    "0.0",
                    "--map-max-relative-disagreement",
                    "1.0",
                    "--map-write-occupancy",
                    "--map-write-observed-mesh",
                    "--optimize-map-consistency",
                    "--optimizer-min-overlap-pixels",
                    "1",
                    "--export-optimized-world-map",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for relative in (
                "world_map/world_map_manifest.json",
                "world_map_optimized/world_map_manifest.json",
                "optimizer/before_metrics.json",
                "optimizer/after_metrics.json",
                "optimizer/depth_scale_bias.jsonl",
                "diagnostics/projection_consistency_before.json",
                "diagnostics/projection_consistency_after.json",
            ):
                self.assertTrue((output / relative).is_file(), relative)
            before = json.loads((output / "optimizer" / "before_metrics.json").read_text())
            after = json.loads((output / "optimizer" / "after_metrics.json").read_text())
            manifest = json.loads(
                (output / "world_map_optimized" / "world_map_manifest.json").read_text()
            )
            training = json.loads(
                (output / "training_cache" / "training_cache_manifest.json").read_text()
            )

            self.assertLess(
                after["vggt_depthpro_rel_diff_mean"], before["vggt_depthpro_rel_diff_mean"]
            )
            self.assertGreaterEqual(after["retained_point_ratio"], 0.5)
            self.assertEqual(
                manifest["truth_boundary"]["label_type"],
                "teacher_pseudo_optimized_map",
            )
            self.assertFalse(manifest["truth_boundary"]["measured_geometry"])
            self.assertFalse(training["usable_for_training"])
            self.assertEqual(
                training["refs"]["optimized_world_map_manifest"],
                "world_map_optimized/world_map_manifest.json",
            )


def _proposal_run(root: Path, *, vggt_depth: float, depth_pro_depth: float) -> dict[str, object]:
    run_dir = ensure_run_tree(root / "run")
    input_dir = root / "input"
    write_ppm_sequence(input_dir, count=2, width=6, height=4)
    failures = []
    frames = build_frame_cache(input_dir, run_dir, max_frames=2, failure_points=failures)
    vggt = load_vggt_proposal_cache(
        write_fake_vggt_cache(
            root / "vggt_cache", frame_ids=(0, 1), width=6, height=4, depth_m=vggt_depth
        )
    )
    depth_pro = load_depth_pro_proposal_cache(
        write_fake_depth_pro_cache(
            root / "depth_pro_cache", frame_ids=(0, 1), width=6, height=4, depth_m=depth_pro_depth
        )
    )
    statuses = write_teacher_statuses(
        run_dir, failures, vggt_result=vggt, depth_pro_result=depth_pro
    )
    proposals = write_proposal_cache(
        run_dir,
        teacher_statuses=statuses,
        frame_records=frames.records,
        keyframes=(),
        debug_geometry_mode="none",
        vggt_result=vggt,
        depth_pro_result=depth_pro,
    )
    disagreement = write_teacher_disagreement(run_dir, proposal_cache=proposals)
    return {"proposals": proposals, "disagreement": disagreement}


if __name__ == "__main__":
    unittest.main()
