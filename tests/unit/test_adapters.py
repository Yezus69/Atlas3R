import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas3r.api import (
    CameraModel,
    FramePacket,
    FramePrediction,
    PoseEstimate,
    ScaleSource,
    TrackingState,
)
from atlas3r.models.adapters import (
    AdapterAvailability,
    AdapterCapabilities,
    AdapterDependencyError,
    AdapterStatus,
    DepthProAdapter,
    FrameBatch,
    GeometryTeacherAdapter,
    TeacherPrediction,
    VGGTAdapter,
    list_adapters,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _K() -> np.ndarray:
    return np.array(
        [
            [80.0, 0.0, 1.5],
            [0.0, 80.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _frame_packet(frame_id: int = 0) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        rgb_u8=np.zeros((2, 3, 3), dtype=np.uint8),
        rgb_model=np.zeros((3, 2, 3), dtype=np.float32),
        K_original=None,
        K_model=_K(),
        distortion=None,
        resize_transform=np.eye(3, dtype=np.float32),
        camera_metadata={"source": "unit-test"},
    )


def _camera() -> CameraModel:
    return CameraModel(
        width=3,
        height=2,
        K=_K(),
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=1.0,
        source="calibrated",
    )


def _pose(frame_id: int = 0) -> PoseEstimate:
    T_world_camera = np.eye(4, dtype=np.float32)
    return PoseEstimate(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        T_world_camera=T_world_camera,
        q_world_camera_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        camera_center_world_m=T_world_camera[:3, 3].copy(),
        covariance_6x6=np.eye(6, dtype=np.float32) * 0.05,
        confidence=0.8,
        tracking_state=TrackingState.OK.value,
        scale_source=ScaleSource.CALIBRATED_RGB.value,
        diagnostics={},
    )


def _frame_prediction(frame_id: int = 0) -> FramePrediction:
    camera = _camera()
    height, width = camera.height, camera.width
    return FramePrediction(
        pose=_pose(frame_id),
        camera=camera,
        depth_m=np.ones((height, width), dtype=np.float32),
        depth_sigma_m=np.ones((height, width), dtype=np.float32) * 0.1,
        normal_camera=np.zeros((height, width, 3), dtype=np.float32),
        point_world=np.zeros((height, width, 3), dtype=np.float32),
        confidence=np.ones((height, width), dtype=np.float32),
        static_mask=np.ones((height, width), dtype=bool),
        object_embeddings=None,
        object_mask_logits=None,
        dense_matches=None,
    )


def _capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        predicts_camera=True,
        predicts_pose=True,
        predicts_depth=True,
        predicts_normals=False,
        predicts_points=True,
        predicts_dense_matches=False,
        predicts_objects=False,
        supports_batch=True,
        supports_streaming=False,
        notes=("unit-test",),
    )


class _StaticAdapter:
    def __init__(self, prediction: TeacherPrediction) -> None:
        self._prediction = prediction

    def predict(self, frames: FrameBatch) -> TeacherPrediction:
        self._last_batch = frames
        return self._prediction


class AdapterContractsTest(unittest.TestCase):
    def test_frame_batch_teacher_prediction_and_protocol_shape(self) -> None:
        capabilities = _capabilities()
        batch = FrameBatch(frames=(_frame_packet(),), batch_id="batch-0", metadata={"fps": 30})
        prediction = TeacherPrediction(
            adapter_name="unit-teacher",
            frame_predictions=(_frame_prediction(),),
            capabilities=capabilities,
            metadata={"coordinate_frame": "world"},
        )
        adapter: GeometryTeacherAdapter = _StaticAdapter(prediction)

        self.assertEqual(batch.frames[0].frame_id, 0)
        self.assertEqual(prediction.capabilities, capabilities)
        self.assertIs(adapter.predict(batch), prediction)

    def test_contracts_reject_invalid_status_and_empty_payloads(self) -> None:
        capabilities = _capabilities()
        with self.assertRaisesRegex(ValueError, "frames"):
            FrameBatch(frames=())
        with self.assertRaisesRegex(ValueError, "frame_predictions"):
            TeacherPrediction(
                adapter_name="unit-teacher",
                frame_predictions=(),
                capabilities=capabilities,
            )
        with self.assertRaisesRegex(ValueError, "availability"):
            AdapterStatus(
                name="unit",
                display_name="Unit",
                availability="maybe",
                capabilities=capabilities,
            )

    def test_stub_modules_import_without_optional_dependencies(self) -> None:
        modules = [
            "atlas3r.models.adapters.vggt_adapter",
            "atlas3r.models.adapters.depth_pro_adapter",
            "atlas3r.models.adapters.registry",
        ]
        for module_name in modules:
            self.assertIsNotNone(importlib.import_module(module_name))

    def test_missing_dependency_errors_include_adapter_name_and_hint(self) -> None:
        with patch("atlas3r.models.adapters._dependency.find_spec", return_value=None):
            with self.assertRaisesRegex(AdapterDependencyError, "VGGT.*Install VGGT"):
                VGGTAdapter()
            with self.assertRaisesRegex(AdapterDependencyError, "Depth Pro.*Install Depth Pro"):
                DepthProAdapter()

    def test_registry_reports_known_adapter_statuses(self) -> None:
        with patch("atlas3r.models.adapters._dependency.find_spec", return_value=None):
            statuses = list_adapters()

        by_name = {status.name: status for status in statuses}
        self.assertEqual(
            by_name["vggt"].availability,
            AdapterAvailability.UNAVAILABLE.value,
        )
        self.assertEqual(
            by_name["depth-pro"].availability,
            AdapterAvailability.UNAVAILABLE.value,
        )
        self.assertIn("missing optional dependency", by_name["vggt"].reason or "")

    def test_cli_adapter_listing_succeeds(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)

        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "adapters", "list"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("name\tstatus\tdetails", result.stdout)
        self.assertIn("vggt", result.stdout)
        self.assertIn("depth-pro", result.stdout)
        self.assertRegex(result.stdout, r"(available|unavailable|stub-only)")


if __name__ == "__main__":
    unittest.main()
