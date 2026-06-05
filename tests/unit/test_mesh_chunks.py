import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.mapping.mesh_artifacts import MeshChunkArtifactWriter
from atlas3r.mapping.mesh_chunks import (
    MESH_TRUTH_FLAGS,
    ObservedMeshChunk,
    chunk_id_from_block_coord,
    load_mesh_chunk_npz,
    save_mesh_chunk_npz,
    write_mesh_chunk_ply,
)
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper, SparseTSDFConfig
from atlas3r.mapping.sparse_tsdf_meshing import SparseTSDFMesherConfig
from tests.unit.test_sparse_tsdf_mapper import _plane_observation


class MeshChunkSchemaTest(unittest.TestCase):
    def test_chunk_id_is_deterministic_from_block_coords(self) -> None:
        self.assertEqual(chunk_id_from_block_coord((1, -2, 3)), "block_1_-2_3")
        self.assertEqual(chunk_id_from_block_coord((1, -2, 3)), "block_1_-2_3")

    def test_mesh_chunk_npz_serializes_deserializes_and_truth_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunk = _chunk()
            payload = save_mesh_chunk_npz(Path(tmp) / "chunk.npz", chunk)
            loaded = load_mesh_chunk_npz(payload)

        self.assertEqual(loaded.chunk_id, chunk.chunk_id)
        self.assertEqual(loaded.version, 1)
        self.assertEqual(loaded.vertex_count, 4)
        self.assertEqual(loaded.triangle_count, 2)
        self.assertFalse(loaded.metadata_record()["accuracy_report"])
        self.assertTrue(loaded.metadata_record()["observed_only"])
        self.assertFalse(loaded.metadata_record()["predicted_completion"])
        self.assertFalse(MESH_TRUTH_FLAGS["hidden_geometry_measured"])
        self.assertEqual(loaded.bbox_world_min_m, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.bbox_world_max_m, [1.0, 1.0, 0.0])

    def test_mesh_chunk_validates_indices_normals_and_ply_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunk = _chunk()
            ply_path = write_mesh_chunk_ply(Path(tmp) / "chunk.ply", chunk)
            header = ply_path.read_text(encoding="utf-8").split("end_header", maxsplit=1)[0]

        self.assertIn("element vertex 4", header)
        self.assertIn("element face 2", header)
        normal_norms = np.linalg.norm(chunk.normals_world, axis=1)
        np.testing.assert_allclose(normal_norms, np.ones_like(normal_norms))
        with self.assertRaisesRegex(ValueError, "triangles"):
            _chunk(triangles=np.asarray([[0, 1, 4]], dtype=np.uint32))

    def test_artifact_writer_versions_increment_on_repeated_dirty_updates(self) -> None:
        observation = _plane_observation()
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
        first = mapper.integrate(observation)
        second_observation = _plane_observation(frame_id=8)
        second = mapper.integrate(second_observation)
        common_blocks = sorted(
            set(first.changed_block_coords_xyz) & set(second.changed_block_coords_xyz)
        )
        self.assertTrue(common_blocks)

        with tempfile.TemporaryDirectory() as tmp:
            writer = MeshChunkArtifactWriter(
                Path(tmp) / "mesh_chunks",
                mesh_format="npz",
                mesher_config=SparseTSDFMesherConfig(),
            )
            first_events, _ = writer.process_dirty_blocks(
                mapper=mapper,
                dirty_block_coords_xyz=(common_blocks[0],),
                frame_id=observation.frame_id,
                timestamp_ns=0,
                dirty_reason="unit",
                capture_queue_depth=0,
                map_queue_depth=0,
                max_dirty_chunks=None,
            )
            second_events, _ = writer.process_dirty_blocks(
                mapper=mapper,
                dirty_block_coords_xyz=(common_blocks[0],),
                frame_id=second_observation.frame_id,
                timestamp_ns=1,
                dirty_reason="unit",
                capture_queue_depth=0,
                map_queue_depth=0,
                max_dirty_chunks=None,
            )

        self.assertEqual(first_events[0]["version"], 1)
        self.assertEqual(second_events[0]["version"], 2)


def _chunk(**overrides: object) -> ObservedMeshChunk:
    values: dict[str, object] = {
        "active_voxel_count": 1,
        "block_size_voxels": 8,
        "chunk_coord_xyz": (0, 0, 0),
        "chunk_id": "block_0_0_0",
        "confidence_summary": {"max": 1.0, "mean": 1.0, "p50": 1.0, "p95": 1.0},
        "coordinate_frame": "x_right_y_down_z_forward",
        "mesher_backend": "unit",
        "metric_scale_source": "external_pose",
        "normals_world": np.asarray(
            [[0.0, 0.0, 1.0]] * 4,
            dtype=np.float32,
        ),
        "observed_coverage_estimate": 1.0 / 512.0,
        "source_frame_ids": [7],
        "surface_voxel_count": 1,
        "T_world_chunk": np.eye(4, dtype=np.float32),
        "triangles": np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        "truncation_distance_m": 0.15,
        "uncertainty_summary_m": {"max": 0.1, "mean": 0.1, "p50": 0.1, "p95": 0.1},
        "update_type": "upsert",
        "version": 1,
        "vertices_world_m": np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "voxel_size_m": 0.05,
    }
    values.update(overrides)
    return ObservedMeshChunk(**values)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
