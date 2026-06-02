import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.training.synthetic_depth_dataset import generate_synthetic_depth_samples
from atlas3r.training.synthetic_overfit import SyntheticOverfitConfig, run_synthetic_overfit
from atlas3r.training.torch_runtime import torch_available


class SyntheticOverfitTorchOptionalTest(unittest.TestCase):
    def _torch(self) -> object:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        return torch

    def test_tiny_model_forward_and_loss_shapes_are_valid(self) -> None:
        torch = self._torch()
        from atlas3r.training.losses import synthetic_depth_pose_loss
        from atlas3r.training.tiny_depth_pose_model import TinyDepthPoseNet

        samples = generate_synthetic_depth_samples(count=2, width=16, height=12, seed=31)
        images = np.stack([sample.rgb_model for sample in samples], axis=0).astype(np.float32)
        depth = np.stack([sample.depth_m for sample in samples], axis=0)[:, np.newaxis, :, :]
        confidence = np.stack([sample.confidence for sample in samples], axis=0)[
            :, np.newaxis, :, :
        ]
        centers = np.stack([sample.camera_center_world_m for sample in samples], axis=0)
        model = TinyDepthPoseNet()

        prediction = model(torch.from_numpy(images))
        loss, metrics = synthetic_depth_pose_loss(
            prediction,
            {
                "depth_m": torch.from_numpy(depth.astype(np.float32)),
                "confidence": torch.from_numpy(confidence.astype(np.float32)),
                "camera_center_world_m": torch.from_numpy(centers.astype(np.float32)),
            },
        )

        self.assertEqual(tuple(prediction["depth_m"].shape), (2, 1, 12, 16))
        self.assertEqual(tuple(prediction["depth_sigma_m"].shape), (2, 1, 12, 16))
        self.assertEqual(tuple(prediction["confidence"].shape), (2, 1, 12, 16))
        self.assertEqual(tuple(prediction["camera_center_world_m"].shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(prediction["depth_m"]).all()))
        self.assertTrue(bool(torch.all(prediction["depth_m"] > 0.0)))
        self.assertTrue(bool(torch.all(prediction["depth_sigma_m"] > 0.0)))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("depth_mae_m", metrics)

    def test_cpu_training_smoke_writes_required_artifacts_and_checkpoint(self) -> None:
        torch = self._torch()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = run_synthetic_overfit(
                SyntheticOverfitConfig(
                    output=output,
                    steps=3,
                    batch_size=2,
                    num_samples=8,
                    width=32,
                    height=24,
                    seed=3,
                    device="cpu",
                    log_every=1,
                )
            )

            self.assertEqual(result["format_name"], "atlas3r_synthetic_overfit_train_run")
            for filename in [
                "config.json",
                "metrics.jsonl",
                "summary.json",
                "checkpoint_last.pt",
                "prediction_preview.html",
                "prediction_preview.svg",
                "prediction_sample.npz",
            ]:
                self.assertTrue((output / filename).is_file(), filename)

            metrics_lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metrics_lines), 3)
            final_metric = json.loads(metrics_lines[-1])
            self.assertIn("loss_total", final_metric)
            self.assertIn("depth_mae_m", final_metric)

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            truth_boundary = summary["truth_boundary"]
            self.assertIs(truth_boundary["training_mvp"], True)
            self.assertIs(truth_boundary["synthetic_only"], True)
            self.assertIs(truth_boundary["real_capture_model"], False)
            self.assertIs(truth_boundary["usable_for_realtime_mapping"], False)
            self.assertIs(truth_boundary["accuracy_report"], False)
            self.assertIs(truth_boundary["performance_report"], False)

            checkpoint = torch.load(output / "checkpoint_last.pt", map_location="cpu")
            for key in [
                "format_name",
                "format_version",
                "step",
                "model_state_dict",
                "optimizer_state_dict",
                "config",
                "metrics",
                "truth_boundary",
            ]:
                self.assertIn(key, checkpoint)
            self.assertEqual(checkpoint["format_name"], "atlas3r_tiny_depth_pose_checkpoint")
            self.assertEqual(checkpoint["step"], 3)

            with np.load(output / "prediction_sample.npz") as sample_arrays:
                self.assertEqual(sample_arrays["target_depth_m"].shape, (24, 32))
                self.assertEqual(sample_arrays["predicted_depth_m"].shape, (24, 32))


if __name__ == "__main__":
    unittest.main()
