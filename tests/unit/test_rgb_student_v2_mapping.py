import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.mapping.mesh_chunks import load_mesh_chunk_npz
from atlas3r.models.smgt import (
    SMGTSmallV2,
    SMGTSmallV2Config,
    save_smgt_small_v2_checkpoint,
    smgt_small_v2_truth_boundary,
)
from atlas3r.runtime.rgb_student_v2_mapping import RGBStudentV2MapConfig, run_rgb_student_v2_mapping
from atlas3r.training.torch_runtime import torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class RGBStudentV2MappingTest(unittest.TestCase):
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
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps({"confidence_threshold": 0.30, "max_sigma_m": 5.0}),
                encoding="utf-8",
            )
            output = root / "student_map"

            summary = run_rgb_student_v2_mapping(
                RGBStudentV2MapConfig(
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
                    calibration=calibration,
                )
            )
            manifest = json.loads(
                (output / "mesh_chunks" / "mesh_chunk_manifest.json").read_text("utf-8")
            )
            first = manifest["chunks"][0]
            chunk = load_mesh_chunk_npz(output / "mesh_chunks" / first["payload_npz"])

        self.assertEqual(summary["student_model_family"], "SMGTSmallV2")
        self.assertGreater(summary["mesh_chunk_count"], 0)
        self.assertGreater(summary["total_vertex_count"], 0)
        self.assertGreater(summary["total_triangle_count"], 0)
        self.assertGreater(chunk.vertex_count, 0)
        self.assertTrue(summary["truth_boundary"]["student_rgb_only_used"])
        self.assertTrue(summary["truth_boundary"]["learned_inference"])
        self.assertFalse(summary["truth_boundary"]["teacher_geometry_used"])
        self.assertFalse(summary["truth_boundary"]["measured_depth_used"])
        self.assertFalse(summary["truth_boundary"]["measured_pose_used"])
        self.assertTrue(summary["truth_boundary"]["measured_training_used"])
        self.assertEqual(summary["student_map_valid_policy"], "confidence_sigma")
        self.assertGreater(summary["mapped_pixel_ratio"], 0.0)

    def test_runtime_v2_does_not_import_vggt_modules(self) -> None:
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
                run_rgb_student_v2_mapping(
                    RGBStudentV2MapConfig(
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

    def test_cli_help_works(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)

        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "map-rgb-student-v2", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--calibration", result.stdout)
        self.assertIn("--rgb-only", result.stdout)


def _write_checkpoint(path: Path, torch: object) -> Path:
    model = SMGTSmallV2(
        SMGTSmallV2Config(
            image_height=8,
            image_width=8,
            clip_length=2,
            stem_dim=8,
            hidden_dim=16,
            feature_dim=128,
            memory_dim=128,
            pose_hidden_dim=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        model.confidence_head.bias.fill_(2.0)
        model.dynamic_head.bias.fill_(-4.0)
        model.sigma_head.bias.fill_(-2.0)
    save_smgt_small_v2_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_record={"unit": True},
        metrics={"loss_total": 1.0},
        validation_metrics={"loss_total": 1.0},
        truth_boundary=smgt_small_v2_truth_boundary(
            measured_training_used=True,
            pseudo_training_used=True,
        ),
    )
    return path


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
