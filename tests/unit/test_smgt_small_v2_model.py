import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.models.smgt import (
    SMGTSmallV2,
    SMGTSmallV2Config,
    load_smgt_small_v2_checkpoint,
    save_smgt_small_v2_checkpoint,
    smgt_small_v2_prediction_to_student_output,
    smgt_small_v2_truth_boundary,
)
from atlas3r.training.torch_runtime import torch_available


class SMGTSmallV2ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        self.torch = torch

    def test_forward_shapes_ranges_and_pose_transforms(self) -> None:
        model = SMGTSmallV2(_config())
        images, K = _inputs(self.torch)

        prediction = model(images, K)
        output = smgt_small_v2_prediction_to_student_output(
            prediction,
            frame_ids=(0, 1, 2, 3),
            truth_boundary=smgt_small_v2_truth_boundary(measured_training_used=True),
        )

        self.assertEqual(output.depth_m.shape, (1, 4, 64, 96))
        self.assertEqual(output.depth_sigma_m.shape, (1, 4, 64, 96))
        self.assertEqual(output.confidence.shape, (1, 4, 64, 96))
        self.assertEqual(output.dynamic_probability.shape, (1, 4, 64, 96))
        self.assertEqual(output.normals_camera.shape, (1, 4, 3, 64, 96))
        self.assertEqual(output.pointmap_camera_m.shape, (1, 4, 3, 64, 96))
        self.assertEqual(output.T_world_camera.shape, (1, 4, 4, 4))
        self.assertTrue(np.all(output.depth_m >= 0.05))
        self.assertTrue(np.all(output.depth_sigma_m >= 0.0))
        self.assertTrue(np.all((output.confidence >= 0.0) & (output.confidence <= 1.0)))
        self.assertTrue(
            np.all((output.dynamic_probability >= 0.0) & (output.dynamic_probability <= 1.0))
        )
        self.assertTrue(np.all(np.isfinite(output.T_world_camera)))
        np.testing.assert_allclose(
            output.T_world_camera[0, :, 3, :],
            np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
            atol=1e-5,
        )

    def test_streaming_step_matches_clip_forward(self) -> None:
        model = SMGTSmallV2(_config())
        model.eval()
        images, K = _inputs(self.torch)

        with self.torch.no_grad():
            clip = model(images, K)
            state = model.init_state(1)
            per_frame = []
            for index in range(4):
                prediction, state = model.step_frame(images[:, index], K[:, index], state)
                per_frame.append(prediction)
            stream = {
                key: self.torch.cat([item[key] for item in per_frame], dim=1)
                for key in per_frame[0]
            }

        for key in ("depth_m", "depth_sigma_m", "confidence", "T_world_camera"):
            self.torch.testing.assert_close(clip[key], stream[key], rtol=1e-5, atol=1e-5)

    def test_phase_pose_prior_tracks_horizontal_rgb_shift(self) -> None:
        config = SMGTSmallV2Config(
            image_height=64,
            image_width=96,
            clip_length=2,
            stem_dim=16,
            hidden_dim=32,
            feature_dim=128,
            memory_dim=128,
            pose_hidden_dim=64,
            learned_pose_residual_scale=0.0,
        )
        model = SMGTSmallV2(config)
        model.eval()
        image = self.torch.rand((1, 1, 3, 64, 96), dtype=self.torch.float32)
        shifted = image.roll(shifts=3, dims=-1)
        images = self.torch.cat([image, shifted], dim=1)
        K = self.torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
        K[:, :, 0, 0] = 96.0
        K[:, :, 1, 1] = 96.0

        with self.torch.no_grad():
            prediction = model(images, K)

        tx = float(prediction["relative_translation_m"][0, 1, 0].cpu().item())
        self.assertLess(tx, -1e-5)

    def test_checkpoint_save_load_and_wrong_format(self) -> None:
        model = SMGTSmallV2(_config())
        optimizer = self.torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint_best.pt"
            save_smgt_small_v2_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=7,
                config_record={"unit": True},
                metrics={"loss_total": 1.0},
                validation_metrics={"val_depth_absrel": 0.2},
                truth_boundary=smgt_small_v2_truth_boundary(
                    measured_training_used=True,
                    pseudo_training_used=True,
                ),
            )
            loaded = load_smgt_small_v2_checkpoint(checkpoint, device="cpu")
            wrong = root / "wrong.pt"
            self.torch.save({"format_name": "atlas3r_smgt_tiny_checkpoint"}, wrong)

            with self.assertRaisesRegex(ValueError, "unsupported SMGT-small-v2"):
                load_smgt_small_v2_checkpoint(wrong, device="cpu")

        self.assertEqual(loaded.checkpoint["step"], 7)
        self.assertEqual(loaded.checkpoint["model_name"], "SMGTSmallV2")


def _config() -> SMGTSmallV2Config:
    return SMGTSmallV2Config(
        image_height=64,
        image_width=96,
        clip_length=4,
        stem_dim=16,
        hidden_dim=32,
        feature_dim=128,
        memory_dim=128,
        pose_hidden_dim=64,
    )


def _inputs(torch: object) -> tuple[object, object]:
    images = torch.rand((1, 4, 3, 64, 96), dtype=torch.float32)
    K = torch.eye(3).view(1, 1, 3, 3).repeat(1, 4, 1, 1)
    K[:, :, 0, 0] = 96.0
    K[:, :, 1, 1] = 96.0
    K[:, :, 0, 2] = 47.5
    K[:, :, 1, 2] = 31.5
    return images, K


if __name__ == "__main__":
    unittest.main()
