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
from atlas3r.forge.tum_rgbd_clips import TumRgbdClipForgeConfig, forge_tum_rgbd_clip_cache
from atlas3r.training.torch_runtime import torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TumClipTemporalOptionalTest(unittest.TestCase):
    def test_forge_cli_help_works(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)

        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "forge", "tum-rgbd-clips", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--write-pointmaps", result.stdout)

    def test_fixture_sequence_forges_train_and_val_clips(self) -> None:
        self._require_pillow()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fixture_manifest(root, frame_count=8)
            train_result = forge_tum_rgbd_clip_cache(
                TumRgbdClipForgeConfig(
                    manifest=manifest_path,
                    output=root / "train_cache",
                    split="train",
                    clip_length=3,
                    width=4,
                    height=3,
                    max_clips=2,
                    write_pointmaps=True,
                    write_normals=True,
                )
            )
            val_result = forge_tum_rgbd_clip_cache(
                TumRgbdClipForgeConfig(
                    manifest=manifest_path,
                    output=root / "val_cache",
                    split="val",
                    clip_length=3,
                    width=4,
                    height=3,
                    max_clips=1,
                )
            )

            train_manifest = json.loads(
                (root / "train_cache" / "atlas3r_clip_cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            with np.load(root / "train_cache" / "clips" / "clip_000000.npz") as payload:
                payload_files = set(payload.files)

        self.assertEqual(train_result["clip_count"], 2)
        self.assertEqual(val_result["clip_count"], 1)
        self.assertEqual(
            train_manifest["truth_boundary"]["teacher_source"], "tum_rgbd_sensor_depth_pose"
        )
        self.assertIn("pointmap_camera_m", payload_files)
        self.assertIn("normal_camera", payload_files)

    def test_torch_dataset_model_loss_and_cpu_training_smoke(self) -> None:
        self._require_pillow()
        self._require_torch()
        import torch

        from atlas3r.training.temporal_losses import temporal_geometry_loss
        from atlas3r.training.tiny_temporal_geometry_model import TinyTemporalMetricNetV0
        from atlas3r.training.tum_clip_dataset import TumRgbdClipCacheDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fixture_manifest(root, frame_count=8)
            train_cache = root / "train_cache"
            val_cache = root / "val_cache"
            forge_tum_rgbd_clip_cache(
                TumRgbdClipForgeConfig(
                    manifest=manifest_path,
                    output=train_cache,
                    split="train",
                    clip_length=3,
                    width=4,
                    height=3,
                    max_clips=2,
                )
            )
            forge_tum_rgbd_clip_cache(
                TumRgbdClipForgeConfig(
                    manifest=manifest_path,
                    output=val_cache,
                    split="val",
                    clip_length=3,
                    width=4,
                    height=3,
                    max_clips=1,
                )
            )
            dataset = TumRgbdClipCacheDataset(train_cache)
            sample = dataset[0]
            model = TinyTemporalMetricNetV0(hidden_channels=4)
            prediction = model(
                sample["images_rgb"].unsqueeze(0),
                sample["intrinsics"].unsqueeze(0),
            )
            loss, metrics = temporal_geometry_loss(
                prediction,
                {key: value.unsqueeze(0) for key, value in sample["target"].items()},
            )
            run_dir = root / "temporal_run"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "train",
                    "tum-rgbd-temporal",
                    "--clip-cache",
                    str(train_cache),
                    "--val-clip-cache",
                    str(val_cache),
                    "--output",
                    str(run_dir),
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
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
                    "--max-runtime-minutes",
                    "5",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
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
                self.assertTrue((run_dir / filename).is_file(), filename)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(tuple(prediction["center_depth_m"].shape), (1, 1, 3, 4))
        self.assertEqual(
            tuple(prediction["relative_translation_center_from_camera"].shape), (1, 3, 3)
        )
        self.assertEqual(metrics["valid_depth_pixels"], 12.0)

    def _require_pillow(self) -> None:
        if not pillow_available():
            self.skipTest("optional Pillow dependency is not installed")

    def _require_torch(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")


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
    base_depth = np.full((3, 4), 5000, dtype=np.uint16)
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
