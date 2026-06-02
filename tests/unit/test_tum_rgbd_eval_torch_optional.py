import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.data.image_runtime import pillow_available
from atlas3r.data.tum_rgbd import prepare_tum_rgbd_manifest
from atlas3r.eval.tum_rgbd_checkpoint import (
    TumRgbdCheckpointEvalConfig,
    run_tum_rgbd_checkpoint_eval,
)
from atlas3r.training.torch_runtime import torch_available
from atlas3r.training.tum_rgbd_artifacts import write_tum_rgbd_checkpoint
from atlas3r.training.tum_rgbd_train import real_rgbd_truth_boundary

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TumRgbdCheckpointEvalTorchOptionalTest(unittest.TestCase):
    def _require_optional_deps(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        if not pillow_available():
            self.skipTest("optional Pillow dependency is not installed")

    def test_evaluator_writes_summary_metrics_preview_and_sample(self) -> None:
        self._require_optional_deps()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_fixture_manifest(root, frame_count=4)
            checkpoint = _write_real_tum_checkpoint(root / "checkpoint.pt")
            output = root / "eval"

            summary = run_tum_rgbd_checkpoint_eval(
                TumRgbdCheckpointEvalConfig(
                    checkpoint=checkpoint,
                    manifest=manifest,
                    output=output,
                    width=4,
                    height=3,
                    device="cpu",
                    max_frames=1,
                    num_workers=0,
                )
            )

            self.assertEqual(summary["metric_family"], "real_rgbd_debug_eval")
            self.assertFalse(summary["accuracy_report"])
            self.assertTrue(summary["diagnostic_only"])
            self.assertEqual(summary["frame_count"], 1)
            for filename in [
                "summary.json",
                "per_frame_metrics.jsonl",
                "prediction_sample.npz",
                "prediction_preview.html",
                "prediction_preview.svg",
                "trajectory_estimate_tum.txt",
                "trajectory_groundtruth_tum.txt",
            ]:
                self.assertTrue((output / filename).is_file(), filename)
            per_frame = (output / "per_frame_metrics.jsonl").read_text(encoding="utf-8")
            self.assertIn("depth_within_1mm_percent", per_frame)

    def test_evaluator_write_tsdf_writes_map_comparison(self) -> None:
        self._require_optional_deps()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_fixture_manifest(root, frame_count=4)
            checkpoint = _write_real_tum_checkpoint(root / "checkpoint.pt")
            output = root / "eval_tsdf"

            run_tum_rgbd_checkpoint_eval(
                TumRgbdCheckpointEvalConfig(
                    checkpoint=checkpoint,
                    manifest=manifest,
                    output=output,
                    width=4,
                    height=3,
                    device="cpu",
                    max_frames=1,
                    num_workers=0,
                    write_tsdf=True,
                    voxel_size_m=0.25,
                    map_max_points=128,
                )
            )

            comparison = json.loads((output / "map_comparison.json").read_text(encoding="utf-8"))
            self.assertFalse(comparison["accuracy_report"])
            self.assertIn("point_set_metrics", comparison)
            self.assertTrue((output / "predicted_tsdf" / "metadata.json").is_file())
            self.assertTrue((output / "target_tsdf" / "metadata.json").is_file())

    def test_evaluator_rejects_checkpoint_with_wrong_truth_boundary(self) -> None:
        self._require_optional_deps()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_fixture_manifest(root, frame_count=4)
            truth_boundary = real_rgbd_truth_boundary()
            truth_boundary["accuracy_report"] = True
            checkpoint = _write_real_tum_checkpoint(
                root / "bad_checkpoint.pt",
                truth_boundary=truth_boundary,
            )

            with self.assertRaisesRegex(ValueError, "truth_boundary.accuracy_report"):
                run_tum_rgbd_checkpoint_eval(
                    TumRgbdCheckpointEvalConfig(
                        checkpoint=checkpoint,
                        manifest=manifest,
                        output=root / "eval_bad",
                        width=4,
                        height=3,
                        device="cpu",
                        max_frames=1,
                        num_workers=0,
                    )
                )

    def test_v2_forward_pass_shapes_are_finite(self) -> None:
        self._require_optional_deps()
        import torch

        from atlas3r.training.tiny_depth_pose_model import TinyMetricDepthNetV2

        model = TinyMetricDepthNetV2(hidden_channels=4)
        images = torch.rand((2, 3, 5, 6), dtype=torch.float32)
        intrinsics = torch.tensor(
            [[[4.0, 0.0, 2.5], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).repeat(2, 1, 1)

        output = model(images, intrinsics)

        self.assertEqual(tuple(output["depth_m"].shape), (2, 1, 5, 6))
        self.assertEqual(tuple(output["depth_sigma_m"].shape), (2, 1, 5, 6))
        self.assertEqual(tuple(output["confidence"].shape), (2, 1, 5, 6))
        self.assertEqual(tuple(output["camera_center_world_m"].shape), (2, 3))
        for value in output.values():
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_cli_help_for_eval_and_new_train_flags(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        eval_help = subprocess.run(
            [sys.executable, "-m", "atlas3r", "eval", "tum-rgbd-checkpoint", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        train_help = subprocess.run(
            [sys.executable, "-m", "atlas3r", "train", "tum-rgbd-depth-pose", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(eval_help.returncode, 0, eval_help.stderr)
        self.assertIn("--write-tsdf", eval_help.stdout)
        self.assertEqual(train_help.returncode, 0, train_help.stderr)
        self.assertIn("--model", train_help.stdout)
        self.assertIn("--depth-loss", train_help.stdout)


def _write_real_tum_checkpoint(
    path: Path,
    *,
    truth_boundary: dict[str, object] | None = None,
) -> Path:
    import torch

    from atlas3r.training.tiny_depth_pose_model import TinyDepthPoseNet

    model = TinyDepthPoseNet(hidden_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    write_tum_rgbd_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_record={"model": "tiny-v1", "hidden_channels": 4},
        metrics={"depth_rmse_m": 1.0, "depth_mae_m": 1.0, "depth_absrel": 1.0},
        validation_metrics={"depth_rmse_m": 1.0, "depth_mae_m": 1.0, "depth_absrel": 1.0},
        truth_boundary=truth_boundary or real_rgbd_truth_boundary(),
    )
    return path


def _write_fixture_manifest(root: Path, *, frame_count: int) -> Path:
    sequence = root / "rgbd_dataset_freiburg1_xyz"
    _write_png_tum_sequence(sequence, frame_count=frame_count)
    manifest_path = root / "manifest.json"
    prepare_tum_rgbd_manifest(
        sequence,
        manifest_path,
        max_frames=frame_count,
        split_policy="block",
        val_fraction=0.5,
    )
    return manifest_path


def _write_png_tum_sequence(root: Path, *, frame_count: int) -> None:
    from PIL import Image

    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_lines = ["# rgb"]
    depth_lines = ["# depth"]
    gt_lines = ["# groundtruth"]
    base_depth = np.array(
        [
            [5000, 5000, 5000, 5000],
            [5000, 5000, 5000, 5000],
            [5000, 5000, 5000, 5000],
        ],
        dtype=np.uint16,
    )
    for index in range(frame_count):
        timestamp = 1.0 + index * 0.033
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 32 + index
        rgb[..., 1] = np.arange(4, dtype=np.uint8)[np.newaxis, :] * 20
        rgb[..., 2] = np.arange(3, dtype=np.uint8)[:, np.newaxis] * 30
        rgb_name = f"rgb/{index:06d}.png"
        depth_name = f"depth/{index:06d}.png"
        Image.fromarray(rgb).save(root / rgb_name)
        Image.fromarray(base_depth + index * 10).save(root / depth_name)
        rgb_lines.append(f"{timestamp:.6f} {rgb_name}")
        depth_lines.append(f"{timestamp:.6f} {depth_name}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.1:.6f} 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
