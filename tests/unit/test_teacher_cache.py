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
    load_teacher_prediction_array_payload,
    load_teacher_prediction_cache,
    validate_teacher_prediction_cache,
    write_teacher_prediction_cache,
)
from atlas3r.models.adapters import (
    AdapterAvailability,
    AdapterCapabilities,
    AdapterRunError,
    AdapterStatus,
    FixtureCubeRoomTeacherAdapter,
    TeacherPrediction,
    run_adapter_to_cache,
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
            self.assertFalse(cache.metadata["arrays"]["stored"])
            self.assertIsNone(cache.metadata["arrays"]["directory"])
            self.assertEqual(cache.adapter_status.name, "unit-teacher")
            self.assertEqual(cache.frame_summaries[0]["coordinate_frame"], "world")
            self.assertEqual(cache.frame_summaries[0]["scale_source"], "calibrated_rgb")
            self.assertIsNone(cache.frame_summaries[0]["arrays_path"])
            self.assertEqual(cache.frame_summaries[0]["camera"]["K"], _K().tolist())
            self.assertEqual(
                cache.frame_summaries[0]["pose"]["T_world_camera"],
                _pose(0).T_world_camera.tolist(),
            )
            self.assertIn("confidence_summary", cache.frame_summaries[0])
            self.assertIn("uncertainty_summary", cache.frame_summaries[0])
            self.assertGreater(cache.frame_summaries[0]["uncertainty_summary"]["mean"], 0.0)

    def test_cache_writer_stores_and_round_trips_array_payloads_when_requested(self) -> None:
        prediction, status = _prediction()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"

            written = write_teacher_prediction_cache(
                prediction,
                cache_dir,
                status,
                store_arrays=True,
            )

            self.assertIn(cache_dir / "arrays" / "frame_000000.npz", written)
            cache = load_teacher_prediction_cache(cache_dir)
            self.assertTrue(cache.metadata["arrays"]["stored"])
            self.assertEqual(cache.metadata["arrays"]["directory"], "arrays")
            self.assertEqual(cache.frame_summaries[0]["arrays_path"], "arrays/frame_000000.npz")

            payload = load_teacher_prediction_array_payload(cache_dir, 0)
            self.assertEqual(
                set(payload),
                {
                    "depth_m",
                    "depth_sigma_m",
                    "normal_camera",
                    "point_world",
                    "confidence",
                    "static_mask",
                },
            )
            np.testing.assert_array_equal(payload["depth_m"], _frame_prediction(0).depth_m)
            np.testing.assert_array_equal(payload["confidence"], _frame_prediction(0).confidence)

    def test_cache_reader_reports_missing_and_corrupt_array_payload_paths(self) -> None:
        prediction, status = _prediction()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            write_teacher_prediction_cache(
                prediction,
                cache_dir,
                status,
                store_arrays=True,
            )

            payload_path = cache_dir / "arrays" / "frame_000000.npz"
            payload_path.unlink()
            with self.assertRaisesRegex(ValueError, r"frame_000000\.npz.*missing array payload"):
                validate_teacher_prediction_cache(cache_dir)

            write_teacher_prediction_cache(
                prediction,
                cache_dir,
                status,
                store_arrays=True,
            )
            payload_path.write_bytes(b"not a valid npz payload")
            with self.assertRaisesRegex(ValueError, r"frame_000000\.npz.*invalid NumPy"):
                validate_teacher_prediction_cache(cache_dir)

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
    def test_fixture_adapter_reconstructs_prediction_from_synthetic_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session.atlas3r"
            write_synthetic_cube_room_session(session)

            prediction = FixtureCubeRoomTeacherAdapter().predict_from_session(session)

            self.assertEqual(prediction.adapter_name, "fixture-cube-room")
            self.assertEqual(len(prediction.frame_predictions), 3)
            self.assertEqual(prediction.metadata["coordinate_frame"], "synthetic_world")
            first = prediction.frame_predictions[0]
            self.assertEqual(first.pose.frame_id, 0)
            self.assertEqual(first.pose.scale_source, "known_anchor")
            self.assertEqual(first.camera.source, "synthetic_calibrated")
            self.assertEqual(first.depth_m.shape, (24, 32))
            self.assertEqual(first.depth_sigma_m.shape, first.depth_m.shape)
            self.assertEqual(first.confidence.shape, first.depth_m.shape)
            self.assertTrue(np.all(first.depth_sigma_m >= 0.0))
            self.assertTrue(np.all((first.confidence >= 0.0) & (first.confidence <= 1.0)))
            self.assertEqual(first.normal_camera.shape, (24, 32, 3))
            self.assertEqual(first.point_world.shape, (24, 32, 3))
            self.assertIsNotNone(first.object_mask_logits)
            self.assertIn(
                "Synthetic analytic fixture", prediction.metadata["geometry_truth_boundary"]
            )

    def test_fixture_runner_writes_deterministic_validated_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            first_cache = root / "cache_a"
            second_cache = root / "cache_b"
            write_synthetic_cube_room_session(session)

            first_result = run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=first_cache,
            )
            second_result = run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=second_cache,
            )

            self.assertEqual(first_result.adapter_status.availability, "available")
            self.assertEqual(second_result.output_cache, second_cache)
            self.assertEqual(
                (first_cache / "metadata.json").read_text(encoding="utf-8"),
                (second_cache / "metadata.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (first_cache / "frame_summaries.jsonl").read_text(encoding="utf-8"),
                (second_cache / "frame_summaries.jsonl").read_text(encoding="utf-8"),
            )

            cache = load_teacher_prediction_cache(first_cache)
            self.assertEqual(cache.adapter_status.name, "fixture-cube-room")
            self.assertEqual(cache.adapter_status.availability, "available")
            self.assertEqual(cache.metadata["frame_ids"], [0, 1, 2])
            self.assertEqual(cache.metadata["coordinate_frame"], "synthetic_world")
            self.assertEqual(cache.metadata["scale_sources"], ["known_anchor"])
            self.assertFalse(cache.metadata["arrays"]["stored"])
            self.assertEqual(cache.frame_summaries[0]["frame_id"], 0)
            self.assertEqual(cache.frame_summaries[0]["coordinate_frame"], "synthetic_world")
            self.assertEqual(cache.frame_summaries[0]["scale_source"], "known_anchor")
            self.assertIsNone(cache.frame_summaries[0]["arrays_path"])
            self.assertEqual(cache.frame_summaries[0]["camera"]["source"], "synthetic_calibrated")
            self.assertIn("confidence_summary", cache.frame_summaries[0])
            self.assertIn("uncertainty_summary", cache.frame_summaries[0])
            self.assertGreater(cache.frame_summaries[0]["confidence_summary"]["mean"], 0.0)
            self.assertGreater(cache.frame_summaries[0]["uncertainty_summary"]["mean"], 0.0)

    def test_fixture_runner_can_store_validated_array_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache_dir = root / "cache"
            write_synthetic_cube_room_session(session)

            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache_dir,
                store_arrays=True,
            )

            cache = load_teacher_prediction_cache(cache_dir)
            self.assertTrue(cache.metadata["arrays"]["stored"])
            self.assertEqual(cache.frame_summaries[0]["arrays_path"], "arrays/frame_000000.npz")
            payload = load_teacher_prediction_array_payload(cache_dir, 0)
            self.assertIn("object_mask_logits", payload)
            self.assertEqual(payload["depth_m"].shape, (24, 32))
            self.assertEqual(payload["depth_sigma_m"].dtype, np.float32)
            self.assertEqual(payload["static_mask"].dtype, np.bool_)
            self.assertTrue(np.all(payload["depth_sigma_m"] >= 0.0))
            self.assertTrue(np.all((payload["confidence"] >= 0.0) & (payload["confidence"] <= 1.0)))

    def test_cli_fixture_runner_writes_cache(self) -> None:
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
                    "fixture-cube-room",
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

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Wrote teacher prediction cache", result.stdout)
            self.assertTrue((cache / "metadata.json").is_file())
            self.assertTrue((cache / "frame_summaries.jsonl").is_file())
            loaded_cache = load_teacher_prediction_cache(cache)
            self.assertEqual(loaded_cache.metadata["frame_ids"], [0, 1, 2])
            self.assertFalse(loaded_cache.metadata["arrays"]["stored"])
            self.assertFalse((cache / "arrays").exists())

    def test_cli_fixture_runner_arrays_and_cache_inspection_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            write_synthetic_cube_room_session(session)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            run_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "adapters",
                    "run",
                    "--adapter",
                    "fixture-cube-room",
                    "--input",
                    str(session),
                    "--output",
                    str(cache),
                    "--store-arrays",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)
            self.assertTrue((cache / "arrays" / "frame_000000.npz").is_file())

            inspect_args = [
                sys.executable,
                "-m",
                "atlas3r",
                "inspect",
                "teacher-cache",
                "--input",
                str(cache),
            ]
            first = subprocess.run(
                inspect_args,
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            second = subprocess.run(
                inspect_args,
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            inspection = json.loads(first.stdout)
            self.assertEqual(inspection["adapter"]["name"], "fixture-cube-room")
            self.assertEqual(inspection["adapter"]["status"], "available")
            self.assertEqual(inspection["frames"]["frame_ids"], [0, 1, 2])
            self.assertEqual(inspection["coordinate_frame"], "synthetic_world")
            self.assertEqual(inspection["scale_sources"], ["known_anchor"])
            self.assertTrue(inspection["arrays"]["stored"])
            self.assertFalse(inspection["cache"]["accuracy_report"])
            self.assertIn("not an accuracy report", inspection["cache"]["accuracy_note"])
            self.assertIn("confidence_summaries", inspection)
            self.assertIn("uncertainty_summaries", inspection)

    def test_fixture_runner_rejects_non_synthetic_and_malformed_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "not_synthetic.atlas3r"
            write_synthetic_cube_room_session(session)
            metadata_path = session / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["session_type"] = "captured_rgb"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(AdapterRunError, "session_type.*synthetic_cube_room"):
                run_adapter_to_cache(
                    adapter_name="fixture-cube-room",
                    input_session=session,
                    output_cache=root / "cache",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "malformed.atlas3r"
            write_synthetic_cube_room_session(session)
            depth_path = session / "depth" / "frame_000000.npz"
            with np.load(depth_path) as depth_file:
                depth_arrays = {name: depth_file[name] for name in depth_file.files}
            del depth_arrays["depth_sigma_m"]
            np.savez(depth_path, **depth_arrays)

            with self.assertRaisesRegex(AdapterRunError, "missing depth sidecar fields"):
                run_adapter_to_cache(
                    adapter_name="fixture-cube-room",
                    input_session=session,
                    output_cache=root / "cache",
                )

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
