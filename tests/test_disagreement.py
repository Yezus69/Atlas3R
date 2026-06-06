from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.depth_pro_witness import load_depth_pro_proposal_cache
from atlas3r.offline.disagreement import write_teacher_disagreement
from atlas3r.offline.proposal_cache import ProposalCacheResult, write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_depth_pro_cache, write_fake_vggt_cache


class DisagreementTest(unittest.TestCase):
    def test_identical_depth_maps_have_near_zero_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposals = _proposal_cache(Path(tmp), vggt_depth=2.0, depth_pro_depth=2.0)

            result = write_teacher_disagreement(Path(tmp) / "run", proposal_cache=proposals)

            self.assertEqual(result.status, "available")
            self.assertEqual(result.valid_overlap_count, 4)
            self.assertAlmostEqual(float(result.summary["abs_depth_diff_mean_m"]), 0.0)
            with np.load(Path(tmp) / "run" / result.maps_npz_path, allow_pickle=False) as data:
                self.assertEqual(int(data["valid_overlap_mask"].sum()), 4)

    def test_scaled_depth_maps_have_high_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposals = _proposal_cache(Path(tmp), vggt_depth=2.0, depth_pro_depth=4.0)

            result = write_teacher_disagreement(Path(tmp) / "run", proposal_cache=proposals)

            self.assertEqual(result.status, "available")
            self.assertGreater(float(result.summary["rel_depth_diff_mean"]), 0.4)
            self.assertEqual(float(result.summary["disagreement_high_ratio"]), 1.0)

    def test_invalid_masks_reduce_overlap_and_source_mask_values_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposals = _proposal_cache(Path(tmp), vggt_depth=2.0, depth_pro_depth=2.0)
            proposals.depth_arrays["vggt_valid_mask_000000"] = np.array(
                [[1.0, 1.0], [0.0, 1.0]], dtype=np.float32
            )
            proposals.depth_arrays["depth_pro_depth_000000"] = np.array(
                [[2.0, 4.0], [3.0, 2.0]], dtype=np.float32
            )
            proposals.depth_arrays["depth_pro_valid_mask_000000"] = np.array(
                [[1.0, 1.0], [1.0, 0.0]], dtype=np.float32
            )

            result = write_teacher_disagreement(Path(tmp) / "run", proposal_cache=proposals)

            self.assertEqual(result.valid_overlap_count, 2)
            with np.load(Path(tmp) / "run" / result.consensus_npz_path, allow_pickle=False) as data:
                source_mask = data["source_mask"][0]
                confidence = data["consensus_confidence"][0]
            self.assertEqual(set(source_mask.reshape((-1,)).tolist()), {1, 2, 3, 4})
            self.assertGreater(float(confidence[0, 0]), 0.8)
            self.assertLess(float(confidence[0, 1]), 0.3)

    def test_missing_witness_writes_insufficient_status_and_loadable_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = ensure_run_tree(Path(tmp) / "run")
            proposals = ProposalCacheResult(
                status="partial",
                manifest_path="proposals/proposal_manifest.json",
                streams=(),
                debug_depth_records=(),
                depth_proposal_available=False,
                debug_geometry_mode="none",
            )

            result = write_teacher_disagreement(run_dir, proposal_cache=proposals)

            payload = json.loads((run_dir / result.json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "insufficient_witnesses")
            with np.load(run_dir / result.maps_npz_path, allow_pickle=False) as data:
                self.assertEqual(data["frame_ids"].shape[0], 0)


def _proposal_cache(
    root: Path, *, vggt_depth: float, depth_pro_depth: float
) -> ProposalCacheResult:
    run_dir = ensure_run_tree(root / "run")
    vggt_cache = write_fake_vggt_cache(
        root / "vggt_cache", frame_ids=(0,), width=2, height=2, depth_m=vggt_depth
    )
    depth_pro_cache = write_fake_depth_pro_cache(
        root / "depth_pro_cache",
        frame_ids=(0,),
        width=2,
        height=2,
        depth_m=depth_pro_depth,
    )
    failures = []
    vggt = load_vggt_proposal_cache(vggt_cache)
    depth_pro = load_depth_pro_proposal_cache(depth_pro_cache)
    statuses = write_teacher_statuses(
        run_dir, failures, vggt_result=vggt, depth_pro_result=depth_pro
    )
    return write_proposal_cache(
        run_dir,
        teacher_statuses=statuses,
        frame_records=(),
        keyframes=(),
        debug_geometry_mode="none",
        vggt_result=vggt,
        depth_pro_result=depth_pro,
    )


if __name__ == "__main__":
    unittest.main()
