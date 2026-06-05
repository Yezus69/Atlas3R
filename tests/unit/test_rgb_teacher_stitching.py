import math
import unittest

import numpy as np

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.runtime.rgb_teacher_stitching import (
    Sim3Transform,
    StitchThresholds,
    TeacherWindowPrediction,
    apply_sim3_scale_to_depth,
    apply_sim3_to_camera_pose,
    estimate_sim3_umeyama,
    stitch_teacher_windows,
)


class RGBTeacherStitchingTest(unittest.TestCase):
    def test_umeyama_recovers_known_sim3(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3], [0.4, 0.3, 1.0]],
            dtype=np.float32,
        )
        rotation = _rotation_z(math.radians(30.0))
        scale = 1.7
        translation = np.asarray([0.2, -0.4, 0.8], dtype=np.float32)
        target = scale * (source @ rotation.T) + translation[None, :]

        sim3 = estimate_sim3_umeyama(source, target)

        self.assertAlmostEqual(sim3.scale, scale, places=5)
        np.testing.assert_allclose(sim3.rotation, rotation, atol=1e-5)
        np.testing.assert_allclose(sim3.translation, translation, atol=1e-5)

    def test_apply_sim3_to_pose_and_depth_scales_center_depth_and_sigma(self) -> None:
        T_local = np.eye(4, dtype=np.float32)
        T_local[:3, 3] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        sim3 = Sim3Transform(
            scale=2.0,
            rotation=_rotation_z(math.pi / 2.0),
            translation=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )

        T_global = apply_sim3_to_camera_pose(T_local, sim3)
        depth, sigma = apply_sim3_scale_to_depth(
            np.ones((2, 2), dtype=np.float32),
            np.full((2, 2), 0.1, dtype=np.float32),
            sim3.scale,
        )

        np.testing.assert_allclose(T_global[:3, 3], [0.0, 3.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(T_global[:3, :3], sim3.rotation, atol=1e-6)
        np.testing.assert_allclose(depth, np.full((2, 2), 2.0, dtype=np.float32))
        np.testing.assert_allclose(sigma, np.full((2, 2), 0.2, dtype=np.float32))

    def test_invalid_sim3_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite Nx3"):
            estimate_sim3_umeyama(
                np.asarray([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            )
        bad_pose = np.eye(4, dtype=np.float32)
        bad_pose[3, 3] = 2.0
        with self.assertRaisesRegex(ValueError, "bottom row"):
            apply_sim3_to_camera_pose(bad_pose, Sim3Transform.identity())

    def test_two_windows_with_known_scale_are_stitched(self) -> None:
        window0 = _window(0, range(0, 6), scale_to_global=1.0, translation_to_global=0.0)
        window1 = _window(1, range(3, 9), scale_to_global=1.5, translation_to_global=0.7)

        batch = stitch_teacher_windows(
            (window0, window1),
            overlap_policy="sim3-overlap",
            thresholds=StitchThresholds(min_overlap_frames=3, min_inliers=3),
        )

        fields = batch.diagnostics.summary_fields()
        self.assertEqual(fields["stitch_accepted_edge_count"], 1)
        self.assertEqual(fields["stitch_rejected_edge_count"], 0)
        self.assertLess(fields["stitch_mean_overlap_center_rmse_m"], 1e-5)
        frame8 = {obs.frame_id: obs for obs in batch.observations}[8]
        np.testing.assert_allclose(frame8.pose.camera_center_world_m, [0.8, 0.0, 0.0], atol=1e-5)
        self.assertEqual(_submap_id(frame8), 0)

    def test_insufficient_overlap_rejects_window(self) -> None:
        window0 = _window(0, range(0, 4), scale_to_global=1.0, translation_to_global=0.0)
        window1 = _window(1, range(3, 7), scale_to_global=1.0, translation_to_global=0.0)

        batch = stitch_teacher_windows(
            (window0, window1),
            overlap_policy="sim3-overlap",
            thresholds=StitchThresholds(min_overlap_frames=2, min_inliers=2),
        )

        fields = batch.diagnostics.summary_fields()
        self.assertEqual(fields["stitch_accepted_edge_count"], 0)
        self.assertEqual(fields["stitch_rejected_edge_count"], 1)
        self.assertEqual(len(fields["rejected_windows"]), 1)
        self.assertEqual([obs.frame_id for obs in batch.observations], [0, 1, 2, 3])

    def test_bad_overlap_residual_rejects_edge(self) -> None:
        window0 = _window(0, range(0, 5), scale_to_global=1.0, translation_to_global=0.0)
        window1 = _window(1, range(2, 7), scale_to_global=1.0, translation_to_global=0.0)
        bad_observations = list(window1.observations)
        bad_observations[1] = _observation(3, np.asarray([10.0, 0.0, 0.0], dtype=np.float32))
        window1 = TeacherWindowPrediction(
            window_index=1,
            frame_ids=window1.frame_ids,
            observations=tuple(bad_observations),
            payload={},
            metadata={},
        )

        batch = stitch_teacher_windows(
            (window0, window1),
            overlap_policy="sim3-overlap",
            thresholds=StitchThresholds(
                min_overlap_frames=3,
                min_inliers=3,
                max_center_rmse_m=0.01,
            ),
        )

        self.assertEqual(batch.diagnostics.summary_fields()["stitch_accepted_edge_count"], 0)
        self.assertEqual(len(batch.observations), 5)

    def test_no_stitch_preserves_raw_later_window_coordinates(self) -> None:
        window0 = _window(0, range(0, 4), scale_to_global=1.0, translation_to_global=0.0)
        window1 = _window(1, range(2, 6), scale_to_global=2.0, translation_to_global=0.5)

        batch = stitch_teacher_windows(
            (window0, window1),
            overlap_policy="none",
            thresholds=StitchThresholds(),
        )

        fields = batch.diagnostics.summary_fields()
        frame5 = {obs.frame_id: obs for obs in batch.observations}[5]
        self.assertEqual(fields["stitch_edge_count"], 0)
        self.assertEqual(fields["stitch_submap_count"], 2)
        np.testing.assert_allclose(frame5.pose.camera_center_world_m, [(0.5 - 0.5) / 2.0, 0, 0])


def _window(
    window_index: int,
    frame_ids: range,
    *,
    scale_to_global: float,
    translation_to_global: float,
) -> TeacherWindowPrediction:
    observations = []
    for frame_id in frame_ids:
        global_center = np.asarray([frame_id * 0.1, 0.0, 0.0], dtype=np.float32)
        local_center = (
            global_center - np.asarray([translation_to_global, 0.0, 0.0])
        ) / scale_to_global
        observations.append(_observation(frame_id, local_center.astype(np.float32)))
    return TeacherWindowPrediction(
        window_index=window_index,
        frame_ids=tuple(frame_ids),
        observations=tuple(observations),
        payload={},
        metadata={},
    )


def _observation(frame_id: int, center: np.ndarray) -> DepthObservation:
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = center
    pose = PoseEstimate(
        frame_id=frame_id,
        timestamp_ns=frame_id,
        T_world_camera=T,
        q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T[:3, :3]).astype(np.float32),
        camera_center_world_m=T[:3, 3].astype(np.float32),
        covariance_6x6=np.eye(6, dtype=np.float32) * 0.01,
        confidence=1.0,
        tracking_state="OK",
        scale_source="rgb_prior",
        diagnostics={"truth_boundary": {"measured_depth_used": False, "measured_pose_used": False}},
    )
    return DepthObservation(
        frame_id=frame_id,
        camera=CameraModel(
            width=4,
            height=4,
            K=K,
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="unit",
        ),
        pose=pose,
        depth_m=np.ones((4, 4), dtype=np.float32),
        depth_sigma_m=np.full((4, 4), 0.1, dtype=np.float32),
        confidence=np.ones((4, 4), dtype=np.float32),
        static_mask=np.ones((4, 4), dtype=np.bool_),
        rgb_u8=np.zeros((4, 4, 3), dtype=np.uint8),
        source="unit",
    )


def _rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _submap_id(observation: DepthObservation) -> int:
    return int(observation.pose.diagnostics["stitching"]["pseudo_submap_id"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
