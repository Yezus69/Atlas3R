import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas3r.teachers.external import (
    DepthProExternalTeacherRunner,
    DepthProFramePrediction,
    ExternalTeacherDependencyError,
    ExternalTeacherRunConfig,
    VGGTExternalTeacherRunner,
    VGGTLocalIngestConfig,
    VGGTRunConfig,
    get_depth_pro_status,
    get_vggt_status,
    ingest_vggt_local_teacher_signal_cache,
)
from atlas3r.teachers.map_eval import TeacherSignalMapConfig, map_teacher_signals
from atlas3r.teachers.signals import load_teacher_signal_manifest, read_teacher_signal_payload
from tests.unit.test_teacher_signals import _clip_payload, _write_clip_cache

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class ExternalTeacherTest(unittest.TestCase):
    def test_depth_pro_status_does_not_import_optional_model_package(self) -> None:
        before = "depth_pro" in sys.modules
        with patch("atlas3r.models.adapters._dependency.find_spec", return_value=None):
            status = get_depth_pro_status()

        self.assertEqual(before, "depth_pro" in sys.modules)
        self.assertEqual(status.availability, "unavailable")
        self.assertFalse(status.can_run_locally)
        self.assertIn("Install Depth Pro", status.install_hint)

    def test_depth_pro_unavailable_path_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ExternalTeacherRunConfig(
                clip_cache=root / "missing_clip_cache",
                output=root / "teacher_cache",
            )

            with patch("atlas3r.models.adapters._dependency.find_spec", return_value=None):
                with self.assertRaisesRegex(
                    ExternalTeacherDependencyError,
                    "Depth Pro.*Install Depth Pro",
                ):
                    DepthProExternalTeacherRunner().run(config)

    def test_vggt_status_does_not_import_optional_model_package(self) -> None:
        before = "vggt" in sys.modules
        with patch("atlas3r.teachers.external.vggt.find_spec", return_value=None):
            status = get_vggt_status()

        self.assertEqual(before, "vggt" in sys.modules)
        self.assertEqual(status.availability, "unavailable")
        self.assertFalse(status.can_run_locally)
        self.assertIn("ATLAS3R_VGGT_REPO", status.reason or "")

    def test_vggt_unavailable_path_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = VGGTRunConfig(
                clip_cache=root / "missing_clip_cache",
                output=root / "teacher_cache",
            )

            with patch("atlas3r.teachers.external.vggt.find_spec", return_value=None):
                with self.assertRaisesRegex(
                    ExternalTeacherDependencyError,
                    "VGGT.*ATLAS3R_VGGT_REPO",
                ):
                    VGGTExternalTeacherRunner().run(config)

    def test_fake_depth_pro_writes_valid_cache_and_map_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            output = root / "depth_pro_teacher"
            inspect_output = root / "depth_pro_inspect"

            result = DepthProExternalTeacherRunner(_fake_depth_pro_predictor).run(
                ExternalTeacherRunConfig(
                    clip_cache=clip_manifest,
                    output=output,
                    max_clips=1,
                    inspect_output=inspect_output,
                )
            )
            manifest = load_teacher_signal_manifest(output, validate_payloads=True)
            payload = read_teacher_signal_payload(output)
            inspect_summary_written = inspect_output.joinpath("summary.json").is_file()
            map_result = map_teacher_signals(
                TeacherSignalMapConfig(
                    teacher_cache=output,
                    output=root / "depth_pro_map",
                    max_clips=1,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                )
            )

        self.assertEqual(result["teacher_name"], "depth_pro")
        self.assertEqual(result["signal_count"], 1)
        self.assertTrue(inspect_summary_written)
        self.assertEqual(manifest["teacher_source_type"], "single_frame_metric_depth_teacher")
        self.assertFalse(manifest["truth_boundary"]["measured_geometry"])  # type: ignore[index]
        self.assertTrue(manifest["truth_boundary"]["pseudo_label"])  # type: ignore[index]
        self.assertGreater(float(payload["depth_sigma_m"][0, 0, 0]), 0.0)
        self.assertEqual(float(payload["confidence"][0, 0, 0]), 0.75)
        self.assertGreater(int(map_result["map_summary"]["surface_point_count"]), 0)  # type: ignore[index]

    def test_depth_pro_reuses_unique_frame_predictions_for_overlapping_clips(self) -> None:
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        def predictor(rgb_u8: np.ndarray, K: np.ndarray) -> DepthProFramePrediction:
            calls.append((rgb_u8.copy(), K.copy()))
            depth_value = float(len(calls))
            height, width = rgb_u8.shape[:2]
            return DepthProFramePrediction(
                depth_m=np.full((height, width), depth_value, dtype=np.float32),
                confidence=np.full((height, width), 0.8, dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache", clip_count=2, overlap=True)
            output = root / "depth_pro_teacher"
            result = DepthProExternalTeacherRunner(predictor).run(
                ExternalTeacherRunConfig(
                    clip_cache=clip_manifest,
                    output=output,
                    max_clips=2,
                    run_inspect=False,
                )
            )
            first_payload = read_teacher_signal_payload(output, signal_index=0)
            second_payload = read_teacher_signal_payload(output, signal_index=1)

        self.assertEqual(len(calls), 3)
        self.assertEqual(result["clip_count"], 2)
        self.assertEqual(result["clip_frame_slot_count"], 4)
        self.assertEqual(result["unique_frame_prediction_count"], 3)
        self.assertEqual(result["duplicate_prediction_reuse_count"], 1)
        self.assertEqual(float(first_payload["depth_m"][1, 0, 0]), 2.0)
        self.assertEqual(float(second_payload["depth_m"][0, 0, 0]), 2.0)

    def test_depth_pro_resizes_prediction_arrays_to_clip_resolution(self) -> None:
        def low_res_predictor(_rgb_u8: np.ndarray, _K: np.ndarray) -> DepthProFramePrediction:
            return DepthProFramePrediction(
                depth_m=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                confidence=np.full((2, 2), 0.8, dtype=np.float32),
                depth_sigma_m=np.full((2, 2), 0.2, dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            output = root / "depth_pro_teacher"
            result = DepthProExternalTeacherRunner(low_res_predictor).run(
                ExternalTeacherRunConfig(
                    clip_cache=clip_manifest,
                    output=output,
                    max_clips=1,
                    run_inspect=False,
                )
            )
            manifest = load_teacher_signal_manifest(output, validate_payloads=True)
            payload = read_teacher_signal_payload(output)

        source_metadata = manifest["source_metadata"]  # type: ignore[index]
        resize_events = source_metadata["prediction_resize_events"]  # type: ignore[index]
        self.assertEqual(tuple(payload["depth_m"].shape), (2, 4, 4))
        self.assertEqual(tuple(payload["confidence"].shape), (2, 4, 4))
        self.assertGreater(result["resized_prediction_array_count"], 0)
        self.assertIn("depth_m", {event["array"] for event in resize_events})  # type: ignore[index]
        self.assertIn("confidence", {event["array"] for event in resize_events})  # type: ignore[index]

    def test_vggt_local_ingest_writes_valid_cache_and_map_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            local_input = root / "vggt_local"
            local_input.mkdir()
            clip_payload = _clip_payload()
            np.savez_compressed(
                local_input / "clip_000000.npz",
                depth_m=clip_payload["depth_m"],
                confidence=clip_payload["valid_depth_mask"].astype(np.float32),
                pointmap_camera_m=np.zeros((2, 4, 4, 3), dtype=np.float32),
                pointmap_world_m=np.zeros((2, 4, 4, 3), dtype=np.float32),
            )
            output = root / "vggt_teacher"

            result = ingest_vggt_local_teacher_signal_cache(
                VGGTLocalIngestConfig(
                    clip_cache=clip_manifest,
                    input=local_input,
                    output=output,
                )
            )
            manifest = load_teacher_signal_manifest(output, validate_payloads=True)
            payload = read_teacher_signal_payload(output)
            map_result = map_teacher_signals(
                TeacherSignalMapConfig(
                    teacher_cache=output,
                    output=root / "vggt_map",
                    max_clips=1,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                )
            )

        self.assertEqual(result["teacher_name"], "vggt")
        self.assertEqual(manifest["teacher_source_type"], "local_external_geometry_teacher")
        self.assertTrue(manifest["truth_boundary"]["pseudo_label"])  # type: ignore[index]
        self.assertFalse(manifest["truth_boundary"]["measured_geometry"])  # type: ignore[index]
        self.assertIn("pointmap_camera_m", payload)
        self.assertAlmostEqual(float(payload["depth_m"][0, 0, 1]), 1.0)
        self.assertGreater(float(payload["depth_sigma_m"][0, 0, 1]), 0.0)
        self.assertGreater(int(map_result["map_summary"]["surface_point_count"]), 0)  # type: ignore[index]

    def test_fake_vggt_writes_valid_cache_alignment_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            output = root / "vggt_teacher"
            inspect_output = root / "vggt_eval"

            result = VGGTExternalTeacherRunner(_fake_vggt_predictor).run(
                VGGTRunConfig(
                    clip_cache=clip_manifest,
                    output=output,
                    max_clips=1,
                    inspect_output=inspect_output,
                    align_to_source_pose="diagnostic_sim3",
                )
            )
            manifest = load_teacher_signal_manifest(output, validate_payloads=True)
            payload = read_teacher_signal_payload(output)
            source_metadata = manifest["source_metadata"]  # type: ignore[index]
            signal_entry = manifest["signals"][0]  # type: ignore[index]
            summary_written = (inspect_output / "summary.json").is_file()
            report_written = (inspect_output / "report.md").is_file()

        self.assertEqual(result["teacher_name"], "vggt")
        self.assertEqual(manifest["teacher_source_type"], "external_multiview_geometry_teacher")
        self.assertTrue(manifest["truth_boundary"]["pseudo_label"])  # type: ignore[index]
        self.assertFalse(manifest["truth_boundary"]["measured_geometry"])  # type: ignore[index]
        self.assertEqual(source_metadata["alignment_policy"], "diagnostic_sim3")  # type: ignore[index]
        self.assertEqual(source_metadata["model_source"], "injected_test_predictor")  # type: ignore[index]
        self.assertIn("world_points", source_metadata["output_fields_found"])  # type: ignore[index]
        self.assertTrue(summary_written)
        self.assertTrue(report_written)
        self.assertFalse(Path(signal_entry["payload_path"]).is_absolute())  # type: ignore[index]
        self.assertNotIn("..", Path(signal_entry["payload_path"]).parts)  # type: ignore[index]
        self.assertEqual(tuple(payload["depth_m"].shape), (2, 4, 4))
        self.assertIn("pointmap_world_m", payload)
        self.assertGreater(float(payload["depth_sigma_m"][0, 0, 1]), 0.0)
        self.assertGreater(float(payload["confidence"][0, 0, 1]), 0.0)

    def test_external_teacher_cli_help_works(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        commands = [
            ["teachers", "run-depth-pro", "--help"],
            ["teachers", "run-vggt", "--help"],
            ["teachers", "ingest-vggt-local", "--help"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", "atlas3r", *command],
                    check=False,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)
                if command[1] in {"run-depth-pro", "run-vggt"}:
                    self.assertIn("--device", result.stdout)
                if command[1] == "run-vggt":
                    self.assertIn("--align-to-source-pose", result.stdout)


def _fake_depth_pro_predictor(
    rgb_u8: np.ndarray,
    _K: np.ndarray,
) -> DepthProFramePrediction:
    height, width = rgb_u8.shape[:2]
    return DepthProFramePrediction(
        depth_m=np.full((height, width), 1.25, dtype=np.float32),
        confidence=np.full((height, width), 0.75, dtype=np.float32),
    )


def _fake_vggt_predictor(clip_payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = clip_payload["depth_m"].copy()
    confidence = np.where(clip_payload["valid_depth_mask"], 2.0, 0.0).astype(np.float32)
    pointmap = np.zeros((*depth.shape, 3), dtype=np.float32)
    pointmap[..., 2] = depth
    extrinsic = np.linalg.inv(clip_payload["T_world_camera"]).astype(np.float32)
    return {
        "depth": depth[..., np.newaxis],
        "depth_conf": confidence,
        "extrinsic": extrinsic,
        "intrinsic": clip_payload["K"],
        "world_points": pointmap,
    }


if __name__ == "__main__":
    unittest.main()
