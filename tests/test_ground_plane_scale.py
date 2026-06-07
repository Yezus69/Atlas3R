from __future__ import annotations

import unittest

import numpy as np

from atlas3r.offline.ground_plane_scale import (
    GROUND_PLANE_METRIC_SCALE_SOURCE,
    estimate_ground_plane_camera_height_scale,
    scale_trajectory,
)


class GroundPlaneScaleTest(unittest.TestCase):
    def test_estimates_scale_from_ground_plane_and_camera_height(self) -> None:
        x, y, z = np.meshgrid(
            np.linspace(-0.8, 0.8, 5),
            np.linspace(-0.3, 0.3, 4),
            np.linspace(0.3, 1.2, 4),
            indexing="ij",
        )
        points = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1).astype(np.float32)
        trajectory = [
            {
                "camera_center_world_m": [0.0, 0.0, 0.0],
                "T_world_camera": np.eye(4, dtype=np.float32).tolist(),
            }
        ]

        estimate = estimate_ground_plane_camera_height_scale(
            points, trajectory, camera_height_prior_m=1.5
        )

        self.assertTrue(estimate.available)
        self.assertGreater(estimate.scale_factor, 1.0)
        self.assertEqual(estimate.metric_scale_source, GROUND_PLANE_METRIC_SCALE_SOURCE)
        self.assertFalse(estimate.to_dict()["physical_accuracy_claim"])

    def test_scales_trajectory_translation_without_changing_rotation(self) -> None:
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        rows = [
            {
                "T_world_camera": transform.tolist(),
                "camera_center_world_m": [1.0, 2.0, 3.0],
                "metric_scale_source": "old",
            }
        ]

        scaled = scale_trajectory(
            rows,
            scale_factor=2.0,
            anchor_world_m=(1.0, 1.0, 1.0),
            metric_scale_source="scaled",
        )

        self.assertEqual(scaled[0]["camera_center_world_m"], [1.0, 3.0, 5.0])
        self.assertEqual(scaled[0]["metric_scale_source"], "scaled")
        np.testing.assert_allclose(
            np.asarray(scaled[0]["T_world_camera"], dtype=np.float32)[:3, :3],
            np.eye(3, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
