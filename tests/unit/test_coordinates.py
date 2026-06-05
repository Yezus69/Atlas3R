from __future__ import annotations

import unittest

import numpy as np

from atlas3r.contracts.coordinates import (
    COORDINATE_FRAME_NAME,
    invert_T_A_B,
    project_points,
    transform_points,
    unproject_depth,
    validate_T_A_B,
)


class CoordinateContractTest(unittest.TestCase):
    def test_T_A_B_convention_maps_points_from_B_to_A(self) -> None:
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[0, 3] = 2.0
        point_camera = np.array([[1.0, 0.0, 3.0]], dtype=np.float32)

        point_world = transform_points(T_world_camera, point_camera)

        np.testing.assert_allclose(point_world, np.array([[3.0, 0.0, 3.0]], dtype=np.float32))
        roundtrip = transform_points(invert_T_A_B(T_world_camera), point_world)
        np.testing.assert_allclose(roundtrip, point_camera, atol=1e-6)

    def test_projection_and_unprojection_use_meters(self) -> None:
        self.assertEqual(COORDINATE_FRAME_NAME, "x_right_y_down_z_forward")
        K = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        depth_m = np.full((3, 3), 2.0, dtype=np.float32)

        points = unproject_depth(K, depth_m)
        center = points[1, 1]
        pixels = project_points(K, center.reshape(1, 3))

        np.testing.assert_allclose(center, np.array([0.0, 0.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(pixels, np.array([[1.0, 1.0]], dtype=np.float32))

    def test_invalid_homogeneous_transform_fails(self) -> None:
        T = np.eye(4, dtype=np.float32)
        T[3, 3] = 2.0
        with self.assertRaises(ValueError):
            validate_T_A_B(T)


if __name__ == "__main__":
    unittest.main()
