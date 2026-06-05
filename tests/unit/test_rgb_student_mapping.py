import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.mesh_chunks import load_mesh_chunk_npz
from atlas3r.mapping.observations import DepthObservation
from atlas3r.models.smgt import SMGTTiny, SMGTTinyConfig, save_smgt_tiny_checkpoint
from atlas3r.models.smgt.checkpoint import smgt_tiny_truth_boundary
from atlas3r.runtime.rgb_student_eval import _depth_eval, _pose_eval
from atlas3r.runtime.rgb_student_mapping import RGBStudentMapConfig, run_rgb_student_mapping
from atlas3r.training.torch_runtime import torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class RGBStudentMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        self.torch = torch

    def test_fixture_checkpoint_maps_nonzero_mesh_and_truth_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_npz_image_folder(root / "images", frame_count=4)
            checkpoint = _write_checkpoint(root / "checkpoint_best.pt", self.torch)
            output = root / "student_map"

            summary = run_rgb_student_mapping(
                RGBStudentMapConfig(
                    input=input_dir,
                    output=output,
                    checkpoint=checkpoint,
                    device="cpu",
                    max_frames=4,
                    frame_stride=1,
                    clip_length=2,
                    clip_overlap=1,
                    image_size=(8, 8),
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    pixel_stride=1,
                    export_mesh_chunks=True,
                    mesh_format="ply",
                    rgb_only=True,
                )
            )
            manifest = json.loads(
                (output / "mesh_chunks" / "mesh_chunk_manifest.json").read_text("utf-8")
            )
            first = manifest["chunks"][0]
            chunk = load_mesh_chunk_npz(output / "mesh_chunks" / first["payload_npz"])
            ply_exists = (output / "mesh_chunks" / first["payload_ply"]).is_file()

        self.assertGreater(summary["mesh_chunk_count"], 0)
        self.assertGreater(summary["total_vertex_count"], 0)
        self.assertGreater(summary["total_triangle_count"], 0)
        self.assertGreater(chunk.vertex_count, 0)
        self.assertTrue(ply_exists)
        self.assertTrue(summary["truth_boundary"]["student_rgb_only_used"])
        self.assertTrue(summary["truth_boundary"]["learned_inference"])
        self.assertFalse(summary["truth_boundary"]["teacher_geometry_used"])
        self.assertFalse(summary["truth_boundary"]["measured_depth_used"])
        self.assertFalse(summary["truth_boundary"]["measured_pose_used"])
        self.assertFalse(summary["truth_boundary"]["rgb_only_mapping_ready"])
        self.assertEqual(
            summary["truth_boundary"]["metric_scale_source"], "student_rgb_prior_unverified"
        )
        self.assertIn("student_mapping_gate", summary)
        self.assertEqual(summary["student_map_valid_policy"], "confidence")
        self.assertGreater(summary["confidence_gated_valid_pixel_ratio"], 0.0)
        self.assertGreater(summary["mapped_pixel_ratio"], 0.0)

    def test_runtime_does_not_import_vggt_modules(self) -> None:
        removed = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name.startswith("atlas3r.teachers.external.vggt")
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_dir = _write_npz_image_folder(root / "images", frame_count=2)
                checkpoint = _write_checkpoint(root / "checkpoint_best.pt", self.torch)
                run_rgb_student_mapping(
                    RGBStudentMapConfig(
                        input=input_dir,
                        output=root / "out",
                        checkpoint=checkpoint,
                        device="cpu",
                        max_frames=2,
                        clip_length=2,
                        clip_overlap=1,
                        image_size=(8, 8),
                        voxel_size_m=0.25,
                        pixel_stride=1,
                        rgb_only=True,
                    )
                )
                imported = [
                    name
                    for name in sys.modules
                    if name.startswith("atlas3r.teachers.external.vggt")
                ]
        finally:
            sys.modules.update(removed)

        self.assertEqual(imported, [])

    def test_strict_confidence_gate_reports_zero_mesh_without_artifact_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_npz_image_folder(root / "images", frame_count=2)
            checkpoint = _write_checkpoint(root / "checkpoint_best.pt", self.torch)
            output = root / "student_map"

            with self.assertRaisesRegex(RuntimeError, "no observed mesh chunks"):
                run_rgb_student_mapping(
                    RGBStudentMapConfig(
                        input=input_dir,
                        output=output,
                        checkpoint=checkpoint,
                        device="cpu",
                        max_frames=2,
                        clip_length=2,
                        clip_overlap=1,
                        image_size=(8, 8),
                        voxel_size_m=0.25,
                        pixel_stride=1,
                        export_mesh_chunks=True,
                        student_confidence_threshold=0.99,
                        rgb_only=True,
                    )
                )

            self.assertTrue((output / "sparse_tsdf" / "sparse_tsdf_state.npz").is_file())
            metadata = json.loads((output / "sparse_tsdf" / "metadata.json").read_text("utf-8"))

        self.assertEqual(metadata["surface_count"], 0)
        self.assertEqual(metadata["voxel_size_m"], 0.25)
        self.assertIn("empty_surface_reason", metadata)

    def test_cli_help_and_missing_checkpoint_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_npz_image_folder(root / "images", frame_count=1)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            help_result = subprocess.run(
                [sys.executable, "-m", "atlas3r", "runtime", "map-rgb-student", "--help"],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            run_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "runtime",
                    "map-rgb-student",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(root / "out"),
                    "--checkpoint",
                    str(root / "missing.pt"),
                    "--rgb-only",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--checkpoint", help_result.stdout)
        self.assertIn("--rgb-only", help_result.stdout)
        self.assertEqual(run_result.returncode, 2)
        self.assertIn("checkpoint file does not exist", run_result.stderr)

    def test_eval_reports_constant_depth_and_no_motion_baselines(self) -> None:
        matched = tuple(
            (_observation(frame_id=index, center_x=float(index)),) * 2 for index in range(3)
        )

        depth = _depth_eval(matched)
        pose, _ = _pose_eval(matched)

        self.assertEqual(depth["absrel"], 0.0)
        self.assertGreater(depth["constant_depth_baseline_absrel"], 0.0)
        self.assertTrue(depth["student_beats_constant_depth_baseline"])
        self.assertEqual(pose["ate_rmse_m"], 0.0)
        self.assertGreater(pose["no_motion_pose_baseline_ate_rmse_m"], 0.0)
        self.assertTrue(pose["student_beats_no_motion_pose_baseline"])

    def test_eval_resizes_measured_depth_to_student_resolution(self) -> None:
        measured = _observation(frame_id=0, center_x=0.0)
        student = _observation(
            frame_id=0,
            center_x=0.0,
            depth=np.repeat(np.repeat(measured.depth_m, 2, axis=0), 2, axis=1),
        )

        depth = _depth_eval(((student, measured),))

        self.assertEqual(depth["overlap_valid_pixel_count"], 16)
        self.assertEqual(depth["absrel"], 0.0)


def _write_checkpoint(path: Path, torch: object) -> Path:
    model = SMGTTiny(
        SMGTTinyConfig(
            image_height=8,
            image_width=8,
            clip_length=2,
            hidden_dim=8,
            feature_dim=12,
            memory_dim=16,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        model.confidence_head.bias.fill_(2.0)
        model.dynamic_head.bias.fill_(-2.0)
    save_smgt_tiny_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_record={"unit": True},
        metrics={"loss_total": 1.0},
        validation_metrics={"loss_total": 1.0},
        truth_boundary=smgt_tiny_truth_boundary(),
    )
    return path


def _observation(
    *,
    frame_id: int,
    center_x: float,
    depth: np.ndarray | None = None,
) -> DepthObservation:
    depth = (
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        if depth is None
        else depth.astype(np.float32, copy=True)
    )
    height, width = depth.shape
    K = np.asarray(
        [
            [float(width), 0.0, (width - 1.0) * 0.5],
            [0.0, float(height), (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    T_world_camera = np.eye(4, dtype=np.float32)
    T_world_camera[0, 3] = np.float32(center_x)
    confidence = np.ones((height, width), dtype=np.float32)
    return DepthObservation(
        frame_id=frame_id,
        camera=CameraModel(
            width=width,
            height=height,
            K=K,
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="unit_test",
        ),
        pose=PoseEstimate(
            frame_id=frame_id,
            timestamp_ns=frame_id,
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            camera_center_world_m=T_world_camera[:3, 3].copy(),
            covariance_6x6=None,
            confidence=1.0,
            tracking_state="OK",
            scale_source="external_pose",
            diagnostics={},
        ),
        depth_m=depth,
        depth_sigma_m=np.full((height, width), 0.01, dtype=np.float32),
        confidence=confidence,
        static_mask=confidence.astype(bool),
        object_id=None,
        rgb_u8=np.zeros((height, width, 3), dtype=np.uint8),
        source="unit_test",
    )


def _write_npz_image_folder(path: Path, *, frame_count: int) -> Path:
    path.mkdir(parents=True)
    K = np.asarray([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.savez(path / "intrinsics.npz", K=K)
    for frame_id in range(frame_count):
        rgb = np.full((8, 8, 3), frame_id * 30, dtype=np.uint8)
        np.savez(path / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
    return path


if __name__ == "__main__":
    unittest.main()
