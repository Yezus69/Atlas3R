from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.camera_scale_ledger import update_soft_metric_scale_ledgers
from atlas3r.offline.depth_pro_witness import load_depth_pro_proposal_cache
from atlas3r.offline.disagreement import write_teacher_disagreement
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_depth_pro_cache, write_fake_vggt_cache, write_ppm_sequence


class SoftMetricScaleLedgerTest(unittest.TestCase):
    def test_agreement_and_cross_view_metrics_raise_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _proposal_run(Path(tmp), vggt_depth=2.0, depth_pro_depth=2.0)

            ledger = update_soft_metric_scale_ledgers(
                run["run_dir"],
                frame_metadata_summary={"focal_35mm_values": [26.0]},
                proposal_cache=run["proposals"],
                disagreement=run["disagreement"],
                cross_view_metrics={
                    "cross_view_projection_count": 10,
                    "cross_view_depth_residual_mean_m": 0.01,
                    "cross_view_depth_residual_p95_m": 0.02,
                },
            )

            self.assertEqual(ledger["selected_scale_mode"], "unanchored_soft_metric")
            self.assertEqual(ledger["scale_status"], "soft_metric_unanchored")
            self.assertEqual(ledger["scale_confidence"], "high")
            self.assertFalse(ledger["physical_accuracy_claim"])

    def test_disagreement_lowers_confidence_and_never_claims_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _proposal_run(Path(tmp), vggt_depth=2.0, depth_pro_depth=8.0)

            ledger = update_soft_metric_scale_ledgers(
                run["run_dir"],
                frame_metadata_summary={},
                proposal_cache=run["proposals"],
                disagreement=run["disagreement"],
                cross_view_metrics=None,
            )

            self.assertEqual(ledger["scale_confidence"], "low")
            self.assertTrue(ledger["depth_pro_metric_prior_available"])
            self.assertTrue(ledger["vggt_metric_prior_available"])
            self.assertFalse(ledger["physical_accuracy_claim"])


def _proposal_run(root: Path, *, vggt_depth: float, depth_pro_depth: float) -> dict[str, object]:
    run_dir = ensure_run_tree(root / "run")
    input_dir = root / "input"
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
        write_fake_vggt_cache(
            root / "vggt_cache", frame_ids=(0, 1), width=6, height=4, depth_m=vggt_depth
        )
    )
    depth_pro = load_depth_pro_proposal_cache(
        write_fake_depth_pro_cache(
            root / "depth_pro_cache",
            frame_ids=(0, 1),
            width=6,
            height=4,
            depth_m=depth_pro_depth,
        )
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
    return {"run_dir": run_dir, "proposals": proposals, "disagreement": disagreement}


if __name__ == "__main__":
    unittest.main()
