import unittest

import numpy as np

from atlas3r.data.synthetic_cube_room import create_synthetic_cube_room_scene
from atlas3r.data.synthetic_observations import depth_observation_from_synthetic_frame
from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
)
from atlas3r.mapping.incremental_tsdf import (
    PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
    IncrementalTSDFConfig,
    PersistentIncrementalTSDFMapper,
)


class PersistentIncrementalTSDFMapperTest(unittest.TestCase):
    def test_initializes_fixed_grid_and_integrates_one_observation(self) -> None:
        scene = create_synthetic_cube_room_scene()
        observation = depth_observation_from_synthetic_frame(scene.frames[0])
        truncation_distance_m = 0.3
        grid_min, grid_max = grid_bounds_from_observations(
            (observation,),
            voxel_size_m=0.1,
            truncation_distance_m=truncation_distance_m,
        )
        mapper = PersistentIncrementalTSDFMapper(
            IncrementalTSDFConfig(
                grid_min_world_m=grid_min,
                grid_max_world_m=grid_max,
                voxel_size_m=0.1,
                truncation_distance_m=truncation_distance_m,
                coordinate_frame="synthetic_world",
                metric_scale_source="known_anchor",
            )
        )

        stats = mapper.integrate(observation)
        volume = mapper.volume()

        self.assertEqual(stats.frame_id, observation.frame_id)
        self.assertEqual(stats.integrated_observation_count, 1)
        self.assertEqual(stats.unique_source_frame_count, 1)
        self.assertGreater(stats.observed_voxel_count, 0)
        self.assertGreater(stats.centers_array_bytes, 0)
        self.assertEqual(stats.update_implementation, PERSISTENT_CPU_UPDATE_IMPLEMENTATION)
        self.assertEqual(volume.source_frame_ids, (observation.frame_id,))
        self.assertEqual(volume.coordinate_frame, "synthetic_world")
        self.assertTrue(np.any(volume.weight > 0.0))

    def test_final_volume_matches_batch_integration_on_tiny_fixture(self) -> None:
        scene = create_synthetic_cube_room_scene()
        observations = tuple(
            depth_observation_from_synthetic_frame(frame) for frame in scene.frames[:2]
        )
        truncation_distance_m = 0.3
        grid_min, grid_max = grid_bounds_from_observations(
            observations,
            voxel_size_m=0.1,
            truncation_distance_m=truncation_distance_m,
        )
        mapper = PersistentIncrementalTSDFMapper(
            IncrementalTSDFConfig(
                grid_min_world_m=grid_min,
                grid_max_world_m=grid_max,
                voxel_size_m=0.1,
                truncation_distance_m=truncation_distance_m,
                coordinate_frame="synthetic_world",
                metric_scale_source="known_anchor",
            )
        )
        for observation in observations:
            mapper.integrate(observation)

        persistent = mapper.volume()
        batch = integrate_observations_to_volume(
            observations,
            grid_min_world_m=grid_min,
            grid_max_world_m=grid_max,
            voxel_size_m=0.1,
            truncation_distance_m=truncation_distance_m,
        )

        self.assertEqual(persistent.tsdf.shape, batch.tsdf.shape)
        np.testing.assert_allclose(persistent.tsdf, batch.tsdf, atol=1.0e-7)
        np.testing.assert_allclose(persistent.weight, batch.weight, atol=1.0e-7)
        self.assertEqual(persistent.source_frame_ids, batch.source_frame_ids)

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "voxel_size_m"):
            PersistentIncrementalTSDFMapper(
                IncrementalTSDFConfig(
                    grid_min_world_m=np.zeros(3, dtype=np.float64),
                    grid_max_world_m=np.ones(3, dtype=np.float64),
                    voxel_size_m=0.0,
                    truncation_distance_m=0.3,
                    coordinate_frame="synthetic_world",
                    metric_scale_source="known_anchor",
                )
            )


if __name__ == "__main__":
    unittest.main()
