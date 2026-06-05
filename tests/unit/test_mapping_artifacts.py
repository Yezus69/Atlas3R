from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.contracts import COORDINATE_FRAME_NAME, MapArtifact, TruthBoundary
from atlas3r.mapping import read_npz_artifact_metadata, write_mesh_ply, write_voxel_npz


class MappingArtifactTest(unittest.TestCase):
    def test_voxel_npz_and_mesh_ply_are_loadable(self) -> None:
        truth = TruthBoundary.teacher_pseudo("teacher_scale_unverified")
        artifact = MapArtifact(
            artifact_type="voxel",
            path="map/world_voxels.npz",
            coordinate_frame=COORDINATE_FRAME_NAME,
            source_frame_ids=(0, 1),
            voxel_size_m=0.1,
            observed_coverage_estimate=0.4,
            mean_uncertainty_m=0.05,
            p95_uncertainty_m=0.12,
            truth_boundary=truth,
            metadata={"observed_only": True},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "world_voxels.npz"
            ply_path = root / "world_mesh.ply"

            write_voxel_npz(
                npz_path,
                occupancy=np.ones((2, 2, 2), dtype=np.float32),
                uncertainty_m=np.full((2, 2, 2), 0.05, dtype=np.float32),
                artifact=artifact,
            )
            write_mesh_ply(
                ply_path,
                vertices_world_m=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                triangles=np.array([[0, 1, 2]], dtype=np.uint32),
                artifact=artifact,
            )

            metadata = read_npz_artifact_metadata(npz_path)
            self.assertEqual(metadata["artifact_type"], "voxel")
            self.assertIn("atlas3r_metadata_json", ply_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
