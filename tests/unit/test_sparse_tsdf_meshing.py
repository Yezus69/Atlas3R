import sys
import unittest

import numpy as np

from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper, SparseTSDFConfig
from atlas3r.mapping.sparse_tsdf_meshing import (
    SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
    SparseTSDFMesherConfig,
    mesh_sparse_tsdf_block,
    mesher_backend_status,
)
from tests.unit.test_sparse_tsdf_mapper import _plane_observation


class SparseTSDFMeshingTest(unittest.TestCase):
    def test_fallback_mesher_produces_valid_triangles_from_sparse_surface(self) -> None:
        mapper = _mapper_with_plane()
        stats = mapper.integrate(_plane_observation())
        self.assertGreater(stats.dirty_block_count, 0)

        chunk = mesh_sparse_tsdf_block(
            mapper,
            stats.changed_block_coords_xyz[0],
            version=1,
            mesher_config=SparseTSDFMesherConfig(),
        )

        self.assertIsNotNone(chunk)
        assert chunk is not None
        self.assertEqual(chunk.mesher_backend, SPARSE_TSDF_FALLBACK_MESHER_BACKEND)
        self.assertGreater(chunk.vertex_count, 0)
        self.assertGreater(chunk.triangle_count, 0)
        self.assertTrue(np.all(chunk.triangles < chunk.vertex_count))
        self.assertTrue(np.all(np.isfinite(chunk.vertices_world_m)))
        self.assertTrue(np.all(np.isfinite(chunk.normals_world)))
        normal_norms = np.linalg.norm(chunk.normals_world, axis=1)
        np.testing.assert_allclose(normal_norms, np.ones_like(normal_norms), atol=1e-6)
        bbox_min = np.asarray(chunk.bbox_world_min_m)
        bbox_max = np.asarray(chunk.bbox_world_max_m)
        self.assertTrue(np.all(chunk.vertices_world_m >= bbox_min[None, :] - 1e-6))
        self.assertTrue(np.all(chunk.vertices_world_m <= bbox_max[None, :] + 1e-6))
        self.assertTrue(chunk.metadata_record()["observed_only"])
        self.assertFalse(chunk.metadata_record()["hidden_geometry_measured"])

    def test_empty_unknown_chunk_produces_no_mesh(self) -> None:
        mapper = SparseBlockTSDFMapper(
            SparseTSDFConfig(
                voxel_size_m=0.1,
                truncation_distance_m=0.3,
                block_size_voxels=4,
                coordinate_frame="x_right_y_down_z_forward",
                metric_scale_source="external_pose",
                pixel_stride=1,
            )
        )

        chunk = mesh_sparse_tsdf_block(
            mapper,
            (0, 0, 0),
            version=1,
            mesher_config=SparseTSDFMesherConfig(),
        )

        self.assertIsNone(chunk)

    def test_optional_skimage_backend_status_is_dependency_safe(self) -> None:
        before = "skimage" in sys.modules
        status = mesher_backend_status()
        after = "skimage" in sys.modules

        self.assertEqual(before, after)
        self.assertIn("skimage_available", status)
        self.assertEqual(status["used_backend"], SPARSE_TSDF_FALLBACK_MESHER_BACKEND)


def _mapper_with_plane() -> SparseBlockTSDFMapper:
    return SparseBlockTSDFMapper(
        SparseTSDFConfig(
            voxel_size_m=0.1,
            truncation_distance_m=0.3,
            block_size_voxels=4,
            coordinate_frame="x_right_y_down_z_forward",
            metric_scale_source="external_pose",
            pixel_stride=1,
        )
    )


if __name__ == "__main__":
    unittest.main()
