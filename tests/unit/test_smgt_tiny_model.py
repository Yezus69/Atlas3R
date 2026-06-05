import unittest

import numpy as np

from atlas3r.models.smgt import SMGTTiny, SMGTTinyConfig, smgt_prediction_to_student_output
from atlas3r.models.smgt.checkpoint import smgt_tiny_truth_boundary
from atlas3r.training.torch_runtime import torch_available


class SMGTTinyModelTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        self.torch = torch

    def test_forward_shapes_ranges_and_student_output_contract(self) -> None:
        config = SMGTTinyConfig(
            image_height=32,
            image_width=48,
            clip_length=4,
            hidden_dim=16,
            feature_dim=24,
            memory_dim=32,
        )
        model = SMGTTiny(config)
        images = self.torch.rand((1, 4, 3, 32, 48), dtype=self.torch.float32)
        K = self.torch.eye(3).view(1, 1, 3, 3).repeat(1, 4, 1, 1)
        K[:, :, 0, 0] = 48.0
        K[:, :, 1, 1] = 48.0
        K[:, :, 0, 2] = 23.5
        K[:, :, 1, 2] = 15.5

        prediction = model(images, K)
        output = smgt_prediction_to_student_output(
            prediction,
            frame_ids=(0, 1, 2, 3),
            truth_boundary=smgt_tiny_truth_boundary(),
        )

        self.assertEqual(output.depth_m.shape, (1, 4, 32, 48))
        self.assertEqual(output.depth_sigma_m.shape, (1, 4, 32, 48))
        self.assertEqual(output.confidence.shape, (1, 4, 32, 48))
        self.assertEqual(output.dynamic_probability.shape, (1, 4, 32, 48))
        self.assertEqual(output.normals_camera.shape, (1, 4, 3, 32, 48))
        self.assertEqual(output.pointmap_camera_m.shape, (1, 4, 3, 32, 48))
        self.assertEqual(output.T_world_camera.shape, (1, 4, 4, 4))
        self.assertTrue(np.all(output.depth_m >= 0.0))
        self.assertTrue(np.all(output.depth_sigma_m >= 0.0))
        self.assertTrue(np.all((output.confidence >= 0.0) & (output.confidence <= 1.0)))
        self.assertTrue(
            np.all((output.dynamic_probability >= 0.0) & (output.dynamic_probability <= 1.0))
        )
        np.testing.assert_allclose(
            output.T_world_camera[0, :, 3, :],
            np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
        )

    def test_streaming_memory_state_steps_one_frame(self) -> None:
        config = SMGTTinyConfig(
            image_height=32,
            image_width=48,
            clip_length=4,
            hidden_dim=16,
            feature_dim=24,
            memory_dim=32,
        )
        model = SMGTTiny(config)
        K = self.torch.eye(3).view(1, 3, 3)
        K[:, 0, 0] = 48.0
        K[:, 1, 1] = 48.0
        image = self.torch.rand((1, 3, 32, 48), dtype=self.torch.float32)

        prediction_a, state_a = model.forward_step(image, K)
        prediction_b, state_b = model.forward_step(image, K, memory_state=state_a.detach())

        self.assertEqual(prediction_a["depth_m"].shape, (1, 1, 32, 48))
        self.assertEqual(prediction_b["T_world_camera"].shape, (1, 1, 4, 4))
        self.assertEqual(state_b.hidden.shape[1], config.memory_dim)


if __name__ == "__main__":
    unittest.main()
