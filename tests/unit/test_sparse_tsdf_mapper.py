import unittest

import numpy as np

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import (
    SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    SparseBlockTSDFMapper,
    SparseTSDFConfig,
)


class SparseBlockTSDFMapperTest(unittest.TestCase):
    def test_rejects_invalid_config_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "voxel_size_m"):
            SparseTSDFConfig(voxel_size_m=0.0, truncation_distance_m=0.15)
        with self.assertRaisesRegex(ValueError, "block_size_voxels"):
            SparseTSDFConfig(
                voxel_size_m=0.05,
                truncation_distance_m=0.15,
                block_size_voxels=0,
            )
        with self.assertRaisesRegex(ValueError, "pixel_stride"):
            SparseTSDFConfig(
                voxel_size_m=0.05,
                truncation_distance_m=0.15,
                pixel_stride=0,
            )

    def test_plane_observation_creates_sparse_state_and_surface(self) -> None:
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

        stats = mapper.integrate(observation)
        surface = mapper.extract_surface()
        voxel_coords, tsdf, weight = mapper.active_voxel_arrays()

        self.assertEqual(stats.frame_id, observation.frame_id)
        self.assertGreater(stats.active_block_count, 0)
        self.assertGreater(stats.active_voxel_count, 0)
        self.assertGreater(stats.new_voxel_count, 0)
        self.assertEqual(stats.update_implementation, SPARSE_TSDF_UPDATE_IMPLEMENTATION)
        self.assertEqual(voxel_coords.shape[1], 3)
        self.assertEqual(tsdf.shape, weight.shape)
        self.assertGreater(surface.points_world_m.shape[0], 0)
        self.assertEqual(surface.metadata["duplicate_frame_policy"], "skip")

    def test_duplicate_frame_id_is_skipped_deterministically(self) -> None:
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
        active_voxels = mapper.active_voxel_count
        second = mapper.integrate(observation)

        self.assertFalse(first.skipped_duplicate_frame)
        self.assertTrue(second.skipped_duplicate_frame)
        self.assertEqual(second.new_voxel_count, 0)
        self.assertEqual(second.updated_voxel_count, 0)
        self.assertEqual(mapper.active_voxel_count, active_voxels)
        self.assertEqual(mapper.source_frame_ids, (observation.frame_id,))


def _plane_observation() -> DepthObservation:
    width, height = 8, 8
    K = np.asarray([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera = CameraModel(
        width=width,
        height=height,
        K=K,
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=1.0,
        source="unit",
    )
    T_world_camera = np.eye(4, dtype=np.float32)
    pose = PoseEstimate(
        frame_id=7,
        timestamp_ns=0,
        T_world_camera=T_world_camera,
        q_world_camera_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        camera_center_world_m=T_world_camera[:3, 3].copy(),
        covariance_6x6=None,
        confidence=1.0,
        tracking_state="OK",
        scale_source="external_pose",
        diagnostics={"coordinate_frame": "x_right_y_down_z_forward"},
    )
    depth = np.ones((height, width), dtype=np.float32)
    return DepthObservation(
        frame_id=7,
        camera=camera,
        pose=pose,
        depth_m=depth,
        depth_sigma_m=np.full_like(depth, 0.02),
        confidence=np.ones_like(depth),
        source="unit_plane_depth",
    )


if __name__ == "__main__":
    unittest.main()
