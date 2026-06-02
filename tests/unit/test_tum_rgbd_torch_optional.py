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
from atlas3r.training.torch_runtime import torch_available
from atlas3r.training.tum_rgbd_dataset import TumRgbdDepthDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TumRgbdTorchOptionalTest(unittest.TestCase):
    def _require_optional_deps(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        if not pillow_available():
            self.skipTest("optional Pillow dependency is not installed")

    def test_dataset_depth_conversion_mask_and_scaled_intrinsics(self) -> None:
        self._require_optional_deps()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_fixture_manifest(Path(tmp), frame_count=2)
            dataset = TumRgbdDepthDataset(
                manifest_path,
                width=4,
                height=3,
                split="val",
            )

            sample = dataset[0]

        depth_m = sample["target"]["depth_m"].numpy()
        valid_mask = sample["target"]["valid_depth_mask"].numpy()
        intrinsics = sample["intrinsics"].numpy()
        self.assertEqual(tuple(sample["images_rgb"].shape), (3, 3, 4))
        self.assertEqual(float(depth_m[0, 0, 0]), 0.0)
        self.assertFalse(bool(valid_mask[0, 0, 0]))
        self.assertEqual(float(depth_m[0, 0, 1]), 1.0)
        self.assertTrue(bool(valid_mask[0, 0, 1]))
        self.assertAlmostEqual(float(intrinsics[0, 0]), 525.0 * 4.0 / 640.0)
        self.assertAlmostEqual(float(intrinsics[1, 1]), 525.0 * 3.0 / 480.0)

    def test_masked_loss_ignores_invalid_pixels_and_rejects_all_invalid(self) -> None:
        self._require_optional_deps()
        import torch

        from atlas3r.training.losses import masked_rgbd_depth_pose_loss

        prediction = {
            "depth_m": torch.tensor([[[[1.0, 50.0]]]], dtype=torch.float32),
            "depth_sigma_m": torch.ones((1, 1, 1, 2), dtype=torch.float32),
            "confidence": torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
            "camera_center_world_m": torch.zeros((1, 3), dtype=torch.float32),
        }
        target = {
            "depth_m": torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float32),
            "valid_depth_mask": torch.tensor([[[[True, False]]]]),
            "camera_center_world_m": torch.zeros((1, 3), dtype=torch.float32),
        }

        loss, metrics = masked_rgbd_depth_pose_loss(prediction, target)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics["depth_mae_m"], 0.0)
        self.assertEqual(metrics["valid_depth_pixels"], 1.0)
        target["valid_depth_mask"] = torch.zeros((1, 1, 1, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "no valid depth"):
            masked_rgbd_depth_pose_loss(prediction, target)

    def test_cli_help_and_one_step_cpu_training_smoke(self) -> None:
        self._require_optional_deps()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fixture_manifest(root, frame_count=3)
            output = root / "run"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            help_result = subprocess.run(
                [sys.executable, "-m", "atlas3r", "train", "tum-rgbd-depth-pose", "--help"],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--manifest", help_result.stdout)

            train_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "train",
                    "tum-rgbd-depth-pose",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output),
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--width",
                    "4",
                    "--height",
                    "3",
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--log-every",
                    "1",
                    "--val-every",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--preview-every",
                    "1",
                    "--hidden-channels",
                    "4",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(train_result.returncode, 0, train_result.stderr)
            for filename in [
                "config.json",
                "metrics.jsonl",
                "validation_metrics.jsonl",
                "summary.json",
                "checkpoint_last.pt",
                "checkpoint_best.pt",
                "prediction_sample.npz",
                "prediction_preview.html",
                "prediction_preview.svg",
            ]:
                self.assertTrue((output / filename).is_file(), filename)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["truth_boundary"]["synthetic_only"])
            self.assertTrue(summary["truth_boundary"]["trained_on_real_rgbd"])


def _write_fixture_manifest(root: Path, *, frame_count: int) -> Path:
    sequence = root / "rgbd_dataset_freiburg1_xyz"
    _write_png_tum_sequence(sequence, frame_count=frame_count)
    manifest_path = root / "manifest.json"
    prepare_tum_rgbd_manifest(sequence, manifest_path, max_frames=frame_count)
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
            [0, 5000, 5000, 5000],
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
        Image.fromarray(base_depth + index).save(root / depth_name)
        rgb_lines.append(f"{timestamp:.6f} {rgb_name}")
        depth_lines.append(f"{timestamp:.6f} {depth_name}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.1:.6f} 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
