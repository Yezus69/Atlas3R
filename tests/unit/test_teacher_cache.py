import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.api import (
    CameraModel,
    FramePrediction,
    PoseEstimate,
    ScaleSource,
    TrackingState,
)
from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.io.teacher_cache import (
    CACHE_FORMAT_NAME,
    load_teacher_prediction_cache,
    validate_teacher_prediction_cache,
    write_teacher_prediction_cache,
)
from atlas3r.models.adapters import (
    AdapterAvailability,
    AdapterCapabilities,
    AdapterStatus,
    TeacherPrediction,
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


def _camera() -> CameraModel:
    return CameraModel(
        width=3,
        height=2,
        K=_K(),
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=0.95,
        source="calibrated",
    )


def _pose(frame_id: int) -> PoseEstimate:
    T_world_camera = np.eye(4, dtype=np.float32)
    T_world_camera[0, 3] = float(frame_id) * 0.1
    return PoseEstimate(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        T_world_camera=T_world_camera,
        q_world_camera_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        camera_center_world_m=T_world_camera[:3, 3].copy(),
        covariance_6x6=np.eye(6, dtype=np.float32) * (0.01 + frame_id * 0.01),
        confidence=0.75,
        tracking_state=TrackingState.OK.value,
        scale_source=ScaleSource.CALIBRATED_RGB.value,
        diagnostics={},
    )


def _frame_prediction(frame_id: int) -> FramePrediction:
    camera = _camera()
    height, width = camera.height, camera.width
    confidence = np.linspace(0.4, 0.9, height * width, dtype=np.float32).reshape(height, width)
    return FramePrediction(
        pose=_pose(frame_id),
        camera=camera,
        depth_m=np.ones((height, width), dtype=np.float32) * (1.0 + frame_id),
        depth_sigma_m=np.ones((height, width), dtype=np.float32) * (0.05 + frame_id * 0.01),
        normal_camera=np.zeros((height, width, 3), dtype=np.float32),
        point_world=np.zeros((height, width, 3), dtype=np.float32),
        confidence=confidence,
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


def _adapter_status(capabilities: AdapterCapabilities) -> AdapterStatus:
    return AdapterStatus(
        name="unit-teacher",
        display_name="Unit Teacher",
        availability=AdapterAvailability.AVAILABLE.value,
        capabilities=capabilities,
        install_hint="No install required for the unit-test teacher.",
        reason=None,
    )


def _prediction() -> tuple[TeacherPrediction, AdapterStatus]:
    capabilities = _capabilities()
    prediction = TeacherPrediction(
        adapter_name="unit-teacher",
        frame_predictions=(_frame_prediction(1), _frame_prediction(0)),
        capabilities=capabilities,
        metadata={"coordinate_frame": "world", "source": "unit-test"},
    )
    return prediction, _adapter_status(capabilities)


class TeacherPredictionCacheTest(unittest.TestCase):
    def test_cache_writer_output_is_deterministic_and_preserves_summaries(self) -> None:
        prediction, status = _prediction()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "cache_a"
            second = root / "cache_b"

            write_teacher_prediction_cache(prediction, first, status)
            write_teacher_prediction_cache(prediction, second, status)

            self.assertEqual(
                (first / "metadata.json").read_text(encoding="utf-8"),
                (second / "metadata.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (first / "frame_summaries.jsonl").read_text(encoding="utf-8"),
                (second / "frame_summaries.jsonl").read_text(encoding="utf-8"),
            )

            cache = load_teacher_prediction_cache(first)
            self.assertEqual(cache.metadata["format_name"], CACHE_FORMAT_NAME)
            self.assertEqual(cache.metadata["frame_ids"], [0, 1])
            self.assertEqual(cache.adapter_status.name, "unit-teacher")
            self.assertEqual(cache.frame_summaries[0]["coordinate_frame"], "world")
            self.assertEqual(cache.frame_summaries[0]["scale_source"], "calibrated_rgb")
            self.assertIn("confidence_summary", cache.frame_summaries[0])
            self.assertIn("uncertainty_summary", cache.frame_summaries[0])
            self.assertGreater(cache.frame_summaries[0]["uncertainty_summary"]["mean"], 0.0)

    def test_cache_reader_validates_required_metadata_and_frame_summaries(self) -> None:
        prediction, status = _prediction()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            write_teacher_prediction_cache(prediction, cache_dir, status)

            metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
            del metadata["adapter"]
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter"):
                validate_teacher_prediction_cache(cache_dir)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            write_teacher_prediction_cache(prediction, cache_dir, status)
            lines = (cache_dir / "frame_summaries.jsonl").read_text(encoding="utf-8").splitlines()
            first_summary = json.loads(lines[0])
            del first_summary["confidence_summary"]
            lines[0] = json.dumps(first_summary)
            (cache_dir / "frame_summaries.jsonl").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "confidence_summary"):
                validate_teacher_prediction_cache(cache_dir)


class AdapterRunnerCliTest(unittest.TestCase):
    def test_cli_runner_reports_clear_known_adapter_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            write_synthetic_cube_room_session(session)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "adapters",
                    "run",
                    "--adapter",
                    "depth-pro",
                    "--input",
                    str(session),
                    "--output",
                    str(cache),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("adapter=depth-pro", result.stderr)
            self.assertRegex(result.stderr, r"status=(unavailable|stub-only)")
            self.assertIn("guidance=", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_runner_reports_stub_only_adapter_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_modules = root / "fake_modules"
            fake_modules.mkdir()
            (fake_modules / "vggt.py").write_text("# fake optional dependency\n", encoding="utf-8")
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            write_synthetic_cube_room_session(session)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{fake_modules}{os.pathsep}{SRC}"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "adapters",
                    "run",
                    "--adapter",
                    "vggt",
                    "--input",
                    str(session),
                    "--output",
                    str(cache),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("adapter=vggt", result.stderr)
            self.assertIn("status=stub-only", result.stderr)
            self.assertIn("Implement the adapter", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
