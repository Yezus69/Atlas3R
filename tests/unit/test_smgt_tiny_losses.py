import unittest

from atlas3r.models.smgt import SMGTTiny, SMGTTinyConfig
from atlas3r.training.smgt_tiny_losses import SMGTTinyLossConfig, smgt_tiny_loss
from atlas3r.training.torch_runtime import torch_available


class SMGTTinyLossTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        self.torch = torch

    def test_losses_are_finite_and_weights_apply(self) -> None:
        model = SMGTTiny(
            SMGTTinyConfig(
                image_height=16,
                image_width=16,
                clip_length=3,
                hidden_dim=8,
                feature_dim=12,
                memory_dim=16,
            )
        )
        images = self.torch.rand((2, 3, 3, 16, 16), dtype=self.torch.float32)
        K = self.torch.eye(3).view(1, 1, 3, 3).repeat(2, 3, 1, 1)
        K[:, :, 0, 0] = 16.0
        K[:, :, 1, 1] = 16.0
        target_T = self.torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
        target_T[:, 1, 0, 3] = 0.05
        target_T[:, 2, 0, 3] = 0.10
        valid = self.torch.ones((2, 3, 16, 16), dtype=self.torch.bool)
        valid[:, :, :4, :] = False
        batch = {
            "images_rgb": images,
            "K": K,
            "depth_target_weight": self.torch.tensor([0.25, 0.10], dtype=self.torch.float32),
            "pose_target_weight": self.torch.tensor([0.25, 0.10], dtype=self.torch.float32),
            "target": {
                "depth_m": self.torch.ones((2, 3, 16, 16), dtype=self.torch.float32),
                "depth_sigma_m": self.torch.full((2, 3, 16, 16), 0.05),
                "confidence": self.torch.ones((2, 3, 16, 16), dtype=self.torch.float32),
                "valid_mask": valid,
                "T_world_camera": target_T,
            },
        }

        prediction = model(images, K)
        loss, metrics = smgt_tiny_loss(prediction, batch, config=SMGTTinyLossConfig())

        self.assertTrue(bool(self.torch.isfinite(loss)))
        self.assertGreater(metrics["loss_total"], 0.0)
        self.assertGreater(metrics["loss_depth_log"], 0.0)
        self.assertGreaterEqual(metrics["loss_pose_relative"], 0.0)
        self.assertLess(metrics["valid_pixel_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
