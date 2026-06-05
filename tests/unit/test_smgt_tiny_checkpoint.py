import tempfile
import unittest
from pathlib import Path

from atlas3r.models.smgt import (
    SMGTTiny,
    SMGTTinyConfig,
    load_smgt_tiny_checkpoint,
    save_smgt_tiny_checkpoint,
)
from atlas3r.models.smgt.checkpoint import smgt_tiny_truth_boundary
from atlas3r.training.torch_runtime import torch_available


class SMGTTinyCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        import torch

        self.torch = torch

    def test_save_load_checkpoint_and_truth_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_best.pt"
            model = SMGTTiny(
                SMGTTinyConfig(
                    image_height=8,
                    image_width=8,
                    hidden_dim=8,
                    feature_dim=12,
                    memory_dim=16,
                )
            )
            optimizer = self.torch.optim.AdamW(model.parameters(), lr=1e-3)
            truth = smgt_tiny_truth_boundary()
            save_smgt_tiny_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=3,
                config_record={"unit": True},
                metrics={"loss_total": 1.0},
                validation_metrics={"loss_total": 0.5},
                truth_boundary=truth,
            )
            loaded = load_smgt_tiny_checkpoint(path, device="cpu")

        self.assertEqual(loaded.checkpoint["format_name"], "atlas3r_smgt_tiny_checkpoint")
        self.assertFalse(loaded.checkpoint["truth_boundary"]["usable_for_mapping"])
        self.assertTrue(loaded.checkpoint["truth_boundary"]["learned_inference"])
        self.assertEqual(loaded.device, "cpu")

    def test_wrong_checkpoint_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.pt"
            self.torch.save({"format_name": "wrong"}, path)

            with self.assertRaisesRegex(ValueError, "unsupported SMGTTiny checkpoint format"):
                load_smgt_tiny_checkpoint(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
