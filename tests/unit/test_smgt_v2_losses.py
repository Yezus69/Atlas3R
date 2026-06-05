import unittest

from atlas3r.models.smgt import SMGTSmallV2, SMGTSmallV2Config
from atlas3r.training.smgt_v2_losses import (
    SMGTV2LossConfig,
    good_pixel_confidence_target,
    smgt_v2_candidate_can_be_best,
    smgt_v2_loss,
)
from atlas3r.training.torch_runtime import torch_available


class SMGTV2LossTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        self.torch = torch

    def test_losses_are_finite_and_weights_apply(self) -> None:
        model = SMGTSmallV2(
            SMGTSmallV2Config(
                image_height=16,
                image_width=16,
                clip_length=3,
                stem_dim=8,
                hidden_dim=16,
                feature_dim=128,
                memory_dim=128,
                pose_hidden_dim=32,
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
        batch = {
            "images_rgb": images,
            "K": K,
            "target_source_id": self.torch.tensor([1.0, 0.0], dtype=self.torch.float32),
            "depth_target_weight": self.torch.tensor([1.0, 0.25], dtype=self.torch.float32),
            "pose_target_weight": self.torch.tensor([1.0, 0.25], dtype=self.torch.float32),
            "confidence_target_weight": self.torch.tensor([1.0, 0.25], dtype=self.torch.float32),
            "target": {
                "depth_m": self.torch.ones((2, 3, 16, 16), dtype=self.torch.float32),
                "depth_sigma_m": self.torch.full((2, 3, 16, 16), 0.05),
                "confidence": self.torch.ones((2, 3, 16, 16), dtype=self.torch.float32),
                "valid_mask": valid,
                "T_world_camera": target_T,
            },
        }

        prediction = model(images, K)
        loss, metrics = smgt_v2_loss(prediction, batch, config=SMGTV2LossConfig())

        self.assertTrue(bool(self.torch.isfinite(loss)))
        self.assertGreater(metrics["loss_total"], 0.0)
        self.assertEqual(batch["depth_target_weight"][0].item(), 1.0)
        self.assertLessEqual(batch["depth_target_weight"][1].item(), 0.25)
        self.assertIn("loss_depth_silog", metrics)
        self.assertIn("loss_temporal_depth_consistency", metrics)
        self.assertIn("confidence_brier", metrics)
        self.assertIn("pose_relative_translation_mean_m", metrics)

    def test_good_pixel_target_uses_measured_error_and_pseudo_confidence(self) -> None:
        pred = self.torch.tensor([[[[1.0, 2.0]]], [[[5.0, 5.0]]]], dtype=self.torch.float32)
        target = self.torch.tensor([[[[1.05, 4.0]]], [[[1.0, 1.0]]]], dtype=self.torch.float32)
        valid = self.torch.ones_like(target, dtype=self.torch.bool)
        teacher_conf = self.torch.tensor([[[[0.2, 0.8]]], [[[0.3, 0.7]]]], dtype=self.torch.float32)
        source_id = self.torch.tensor([1.0, 0.0], dtype=self.torch.float32)

        confidence_target = good_pixel_confidence_target(
            pred, target, valid, teacher_conf, source_id
        )

        self.assertEqual(confidence_target[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(confidence_target[0, 0, 0, 1].item(), 0.0)
        self.assertAlmostEqual(confidence_target[1, 0, 0, 0].item(), 0.3, places=6)
        self.assertAlmostEqual(confidence_target[1, 0, 0, 1].item(), 0.7, places=6)

    def test_checkpoint_selection_rejects_zero_mesh_candidates(self) -> None:
        self.assertFalse(
            smgt_v2_candidate_can_be_best(
                {
                    "mesh_success": 0.0,
                    "mesh_chunk_count": 0.0,
                    "val_mapped_pixel_ratio_at_selected_threshold": 0.4,
                }
            )
        )
        self.assertTrue(
            smgt_v2_candidate_can_be_best(
                {
                    "mesh_success": 1.0,
                    "mesh_chunk_count": 1.0,
                    "val_mapped_pixel_ratio_at_selected_threshold": 0.4,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
