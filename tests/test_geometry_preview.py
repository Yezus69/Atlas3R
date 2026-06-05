from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.geometry_preview import (
    lift_depth_to_world_points,
    read_geometry_metadata,
    write_geometry_preview,
)
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import ensure_run_tree
from tests.helpers import write_ppm_sequence


class GeometryPreviewTest(unittest.TestCase):
    def test_pixel_depth_pose_lifts_to_world_points(self) -> None:
        K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        T_world_camera = np.eye(4, dtype=np.float32)

        points = lift_depth_to_world_points(K, depth, T_world_camera)

        expected = np.array(
            [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(points, expected)

    def test_empty_geometry_writes_failure_reason_and_loadable_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=1)
            failures = []
            frames = build_frame_cache(input_dir, run_dir, max_frames=1, failure_points=failures)
            proposals = ProposalCacheResult(
                status="partial",
                manifest_path="proposals/proposal_manifest.json",
                streams=(),
                debug_depth_records=(),
                depth_proposal_available=False,
                debug_geometry_mode="none",
            )

            result = write_geometry_preview(
                run_dir,
                frame_cache=frames,
                proposal_cache=proposals,
                write_ply=True,
                failure_points=failures,
            )

            self.assertEqual(result.point_count, 0)
            metadata = read_geometry_metadata(run_dir / result.geometry_npz_path)
            self.assertEqual(metadata["status"], "unavailable")
            self.assertIsNone(result.geometry_ply_path)
            self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
