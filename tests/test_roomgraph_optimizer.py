from __future__ import annotations

import unittest

import numpy as np

from atlas3r.offline.roomgraph_core import (
    RoomGraphFrame,
    RoomGraphObservation,
    RoomGraphProblem,
    optimize_roomgraph_variant,
)


class RoomGraphOptimizerTest(unittest.TestCase):
    def test_joint_variant_expands_collapsed_pose_with_tracks(self) -> None:
        K = np.array([[120.0, 0.0, 50.0], [0.0, 120.0, 50.0], [0.0, 0.0, 1.0]], np.float32)
        true_centers = [np.array([0.0, 0.0, 0.0]), np.array([0.8, 0.0, 0.0])]
        prior_centers = [center.astype(np.float32) * 0.25 for center in true_centers]
        points = np.array(
            [
                [0.0, -0.2, 4.0],
                [0.2, 0.1, 4.5],
                [-0.25, 0.15, 5.0],
                [0.35, -0.1, 4.2],
            ],
            dtype=np.float32,
        )
        frames = tuple(
            RoomGraphFrame(
                frame_id=index,
                keyframe_id=index,
                K=K,
                R_world_camera=np.eye(3, dtype=np.float32),
                center_prior=prior_centers[index],
                depth_pro_m=np.ones((100, 100), dtype=np.float32),
                depth_valid=np.ones((100, 100), dtype=np.bool_),
                vggt_depth_m=np.ones((100, 100), dtype=np.float32),
                vggt_valid=np.ones((100, 100), dtype=np.bool_),
                base_depth_scale=1.0,
                base_depth_bias_m=0.0,
            )
            for index in range(2)
        )
        observations = []
        for point_index, point in enumerate(points):
            for frame_index, center in enumerate(true_centers):
                point_camera = point - center.astype(np.float32)
                pixel = (K @ point_camera)[:2] / point_camera[2]
                observations.append(
                    RoomGraphObservation(
                        track_index=point_index,
                        frame_index=frame_index,
                        xy_px=(float(pixel[0]), float(pixel[1])),
                        depth_pro_m=float(point_camera[2]),
                        vggt_depth_m=float(point_camera[2]),
                        confidence=1.0,
                    )
                )
        problem = RoomGraphProblem(
            frames=frames,
            observations=tuple(observations),
            initial_points_world_m=points,
        )

        result = optimize_roomgraph_variant(problem, variant="joint", max_iterations=80)

        self.assertGreater(result.after_metrics.camera_collapse_score_m, 0.4)
        self.assertLess(
            result.after_metrics.reprojection_error_mean_px,
            result.before_metrics.reprojection_error_mean_px,
        )
        self.assertGreater(result.improvement["camera_collapse_score_m"], 0.1)


if __name__ == "__main__":
    unittest.main()
