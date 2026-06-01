import unittest

import numpy as np

from atlas3r.camera.pinhole import (
    project_points_camera,
    scale_intrinsics,
    unproject_depth_map,
    unproject_pixels,
)
from atlas3r.pose.transforms import (
    camera_center_from_T_world_camera,
    compose_transforms,
    invert_transform,
    make_transform,
    quaternion_xyzw_from_rotation_matrix,
    rotation_matrix_from_quaternion_xyzw,
    transform_points,
)


class CoordinateMathTest(unittest.TestCase):
    def test_camera_center_is_transform_translation(self) -> None:
        T_world_camera = make_transform(np.eye(3), np.array([1.0, -2.0, 3.0]))
        np.testing.assert_allclose(
            camera_center_from_T_world_camera(T_world_camera), T_world_camera[:3, 3]
        )

    def test_quaternion_rotation_round_trip_identity_and_90_deg(self) -> None:
        identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        identity_R = rotation_matrix_from_quaternion_xyzw(identity_q)
        np.testing.assert_allclose(quaternion_xyzw_from_rotation_matrix(identity_R), identity_q)

        angle = np.pi / 2.0
        q_z_90 = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
        R_z_90 = rotation_matrix_from_quaternion_xyzw(q_z_90)
        q_round_trip = quaternion_xyzw_from_rotation_matrix(R_z_90)
        np.testing.assert_allclose(q_round_trip, q_z_90, atol=1e-12)

    def test_transform_compose_invert_and_points(self) -> None:
        T_A_B = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
        T_B_C = make_transform(np.eye(3), np.array([0.0, 2.0, 0.0]))
        T_A_C = compose_transforms(T_A_B, T_B_C)
        points_C = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
        np.testing.assert_allclose(transform_points(T_A_C, points_C), [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(invert_transform(T_A_C) @ T_A_C, np.eye(4), atol=1e-12)

    def test_project_unproject_round_trip(self) -> None:
        K = np.array(
            [
                [120.0, 0.0, 3.0],
                [0.0, 110.0, 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points_camera = np.array(
            [[0.1, -0.2, 1.5], [0.5, 0.25, 2.0], [-0.1, 0.3, 3.0]], dtype=np.float64
        )
        pixels_uv, depth_m = project_points_camera(points_camera, K)
        round_trip = unproject_pixels(pixels_uv, depth_m, K)
        max_error = float(np.max(np.abs(points_camera - round_trip)))
        self.assertLess(max_error, 1e-5)

    def test_project_rejects_non_positive_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "points_camera_m"):
            project_points_camera(np.array([[0.0, 0.0, 0.0]], dtype=np.float64), np.eye(3))

    def test_unproject_depth_map_center_ray(self) -> None:
        K = np.array(
            [
                [100.0, 0.0, 1.0],
                [0.0, 100.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        depth_m = np.ones((3, 3), dtype=np.float64) * 2.0
        points = unproject_depth_map(depth_m, K)
        self.assertEqual(points.shape, (3, 3, 3))
        np.testing.assert_allclose(points[1, 1], np.array([0.0, 0.0, 2.0]), atol=1e-12)

    def test_scale_intrinsics(self) -> None:
        K = np.array(
            [
                [100.0, 0.0, 4.0],
                [0.0, 200.0, 6.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            scale_intrinsics(K, 0.5, 2.0),
            np.array([[50.0, 0.0, 2.0], [0.0, 400.0, 12.0], [0.0, 0.0, 1.0]]),
        )


if __name__ == "__main__":
    unittest.main()
