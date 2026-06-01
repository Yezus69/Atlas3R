import unittest

import numpy as np

from atlas3r.api import (
    CameraModel,
    DenseMatchSet,
    FramePrediction,
    MeshChunk,
    ObjectInstance,
    PoseEstimate,
    ScaleSource,
    SurfaceSource,
    TrackingState,
)


def _K() -> np.ndarray:
    return np.array(
        [
            [100.0, 0.0, 2.0],
            [0.0, 100.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _T(translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def _camera() -> CameraModel:
    return CameraModel(
        width=4,
        height=3,
        K=_K(),
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=1.0,
        source="calibrated",
    )


def _pose(T_world_camera: np.ndarray | None = None) -> PoseEstimate:
    T = _T((1.0, 2.0, 3.0)) if T_world_camera is None else T_world_camera
    center = T[:3, 3].copy() if T.shape == (4, 4) else np.zeros(3, dtype=np.float32)
    return PoseEstimate(
        frame_id=7,
        timestamp_ns=123,
        T_world_camera=T,
        q_world_camera_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        camera_center_world_m=center,
        covariance_6x6=np.eye(6, dtype=np.float32) * 0.01,
        confidence=0.9,
        tracking_state=TrackingState.OK.value,
        scale_source=ScaleSource.CALIBRATED_RGB.value,
        diagnostics={},
    )


def _mesh_chunk(**overrides: object) -> MeshChunk:
    values: dict[str, object] = {
        "chunk_id": "chunk_0",
        "version": 1,
        "T_world_chunk": _T(),
        "vertices_m": np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
        ),
        "faces": np.array([[0, 1, 2]], dtype=np.int32),
        "normals": None,
        "colors": None,
        "uvs": None,
        "object_id_per_face": None,
        "surface_source_per_face": np.array([SurfaceSource.OBSERVED_SURFACE], dtype=np.int8),
        "voxel_size_m": 0.05,
        "mean_uncertainty_m": 0.01,
        "p95_uncertainty_m": 0.02,
        "source_frame_ids": [7],
        "scale_source": ScaleSource.CALIBRATED_RGB.value,
        "flags": [],
    }
    values.update(overrides)
    return MeshChunk(**values)  # type: ignore[arg-type]


def _object_instance(**overrides: object) -> ObjectInstance:
    values: dict[str, object] = {
        "object_id": 1,
        "label_candidates": [("box", 0.8)],
        "T_world_object": _T(),
        "oriented_bbox_center_m": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "oriented_bbox_axes": np.eye(3, dtype=np.float32),
        "oriented_bbox_extents_m": np.array([0.5, 0.25, 0.75], dtype=np.float32),
        "mesh_chunk_ids": ["chunk_0"],
        "is_dynamic": False,
        "observed_coverage_ratio": 0.7,
        "confidence": 0.8,
        "uncertainty_m": 0.02,
        "first_seen_frame_id": 3,
        "last_seen_frame_id": 5,
        "metadata": {},
    }
    values.update(overrides)
    return ObjectInstance(**values)  # type: ignore[arg-type]


class ContractValidationTest(unittest.TestCase):
    def test_valid_camera_model_accepted(self) -> None:
        camera = _camera()
        self.assertEqual(camera.width, 4)
        self.assertEqual(camera.distortion_model, "none")

    def test_invalid_intrinsics_rejected(self) -> None:
        K = _K()
        K[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "K"):
            CameraModel(
                width=4,
                height=3,
                K=K,
                distortion_model="none",
                distortion_params=None,
                rolling_shutter_row_time_s=None,
                confidence=1.0,
                source="calibrated",
            )

    def test_valid_pose_estimate_accepted(self) -> None:
        pose = _pose()
        np.testing.assert_allclose(pose.camera_center_world_m, np.array([1.0, 2.0, 3.0]))

    def test_non_4x4_transform_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "T_world_camera"):
            _pose(np.eye(3, dtype=np.float32))

    def test_wrong_bottom_row_rejected(self) -> None:
        T = _T()
        T[3, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "T_world_camera"):
            _pose(T)

    def test_valid_frame_prediction_and_dense_matches_accepted(self) -> None:
        camera = _camera()
        height, width = camera.height, camera.width
        dense_matches = DenseMatchSet(
            source_frame_id=0,
            target_frame_id=1,
            source_pixels_uv=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            target_pixels_uv=np.array([[0.5, 0.0], [1.5, 1.0]], dtype=np.float32),
            confidence=np.array([1.0, 0.5], dtype=np.float32),
        )
        prediction = FramePrediction(
            pose=_pose(),
            camera=camera,
            depth_m=np.ones((height, width), dtype=np.float32),
            depth_sigma_m=np.ones((height, width), dtype=np.float32) * 0.1,
            normal_camera=np.zeros((height, width, 3), dtype=np.float32),
            point_world=np.zeros((height, width, 3), dtype=np.float32),
            confidence=np.ones((height, width), dtype=np.float32),
            static_mask=np.ones((height, width), dtype=bool),
            object_embeddings=None,
            object_mask_logits=None,
            dense_matches=dense_matches,
        )
        self.assertIs(prediction.dense_matches, dense_matches)

    def test_mesh_chunk_rejects_missing_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_frame_ids"):
            _mesh_chunk(source_frame_ids=[])
        with self.assertRaisesRegex(ValueError, "scale_source"):
            _mesh_chunk(scale_source="")

    def test_mesh_chunk_rejects_out_of_range_face_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "faces"):
            _mesh_chunk(faces=np.array([[0, 1, 3]], dtype=np.int32))

    def test_surface_source_numeric_values_match_docs(self) -> None:
        self.assertEqual(int(SurfaceSource.OBSERVED_SURFACE), 0)
        self.assertEqual(int(SurfaceSource.SINGLE_VIEW_PRIOR), 1)
        self.assertEqual(int(SurfaceSource.COMPLETED_SURFACE), 2)
        self.assertEqual(int(SurfaceSource.DYNAMIC_SURFACE), 3)
        self.assertEqual(int(SurfaceSource.LOW_CONFIDENCE), 4)

    def test_object_instance_rejects_invalid_bounds_and_ordering(self) -> None:
        with self.assertRaisesRegex(ValueError, "observed_coverage_ratio"):
            _object_instance(observed_coverage_ratio=1.5)
        with self.assertRaisesRegex(ValueError, "confidence"):
            _object_instance(confidence=-0.1)
        with self.assertRaisesRegex(ValueError, "first_seen_frame_id"):
            _object_instance(first_seen_frame_id=8, last_seen_frame_id=7)
        with self.assertRaisesRegex(ValueError, "oriented_bbox_extents_m"):
            _object_instance(oriented_bbox_extents_m=np.array([1.0, 0.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
